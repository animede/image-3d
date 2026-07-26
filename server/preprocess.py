"""画像前処理 (SPEC.md §3.1 / §3.2, FR-1, FR-2)。

- アップロード画像の検証(フォーマット・サイズ)
- リサイズ・正方形化
- 背景除去。単色背景は色キー、それ以外は rembg。未導入環境では自動スキップ(NFR-5)。
"""
from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import binary_dilation, binary_fill_holes, label

logger = logging.getLogger(__name__)

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_DIMENSION = 2048
TARGET_SIZE = 1024


class InvalidImageError(ValueError):
    """アップロードされたデータが有効な画像でない場合に送出する。"""


def load_and_validate_image(data: bytes, max_bytes: int) -> Image.Image:
    """バイト列を検証し、PIL Imageとして読み込む。"""
    if len(data) == 0:
        raise InvalidImageError("空のファイルです。")
    if len(data) > max_bytes:
        raise InvalidImageError(
            f"ファイルサイズが上限({max_bytes // (1024 * 1024)}MB)を超えています。"
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("画像として読み込めませんでした。PNG/JPEG/WebPを使用してください。") from exc

    if image.format not in ALLOWED_FORMATS:
        raise InvalidImageError(
            f"対応していない画像形式です({image.format})。PNG/JPEG/WebPを使用してください。"
        )

    return image.convert("RGBA") if image.mode != "RGBA" else image


def resize_to_square(image: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    """アスペクト比を保ってリサイズし、透明背景の正方形にレターボックスする。"""
    image = image.copy()
    image.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image, offset, image if image.mode == "RGBA" else None)
    return canvas


def remove_background(image: Image.Image) -> tuple[Image.Image, bool]:
    """rembgで背景除去する。未導入・失敗時は元画像をそのまま返し、
    適用できたかどうかを bool で示す(NFR-5: mock/軽量環境でも動作継続)。
    """
    try:
        from rembg import remove  # 遅延import: Phase1では未導入でもOK
    except ImportError:
        logger.info("rembg is not installed; skipping background removal.")
        return image, False

    try:
        result = remove(image)
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        return result, True
    except Exception:
        logger.exception("Background removal failed; using original image.")
        return image, False


# --- 単色背景の色キー -------------------------------------------------------
# 画像生成で作った素材は背景が完全な単色のことが多い。そこに rembg をかけると
# **背景と同系色の部位を丸ごと落とす**ことがある(実測: 白背景・白い毛のT字
# キャラで両腕が消失し、前景率 37.8%→28.6%)。単色と判定できる場合は推定より
# 確実な色キーを使う。

# 枠画素の背景色からの平均ずれ(max channel)がこれ以下なら「単色背景」とみなす。
UNIFORM_BG_SPREAD = 8.0
# 背景色とみなす色差。背景が単色であることは判定済みなので、JPEGノイズを吸収
# する程度に狭く取る。広げると背景と色の近い被写体(白背景の白い毛など)を
# 削ってしまう。アンチエイリアスのふちはこの下限から線形に不透明化する。
UNIFORM_BG_TOLERANCE = 12
UNIFORM_BG_SOFT_FLOOR = 4
# 色キー結果がこの範囲外なら破綻とみなし rembg にフォールバックする。
UNIFORM_BG_MIN_FOREGROUND = 0.02
UNIFORM_BG_MAX_FOREGROUND = 0.95


def detect_uniform_background(rgb: np.ndarray) -> np.ndarray | None:
    """枠1画素の色から単色背景を推定する。単色でなければ None。"""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    spread = float(np.abs(border - bg).max(axis=1).mean())
    if spread > UNIFORM_BG_SPREAD:
        logger.info("Background is not uniform (spread=%.1f); colour key skipped.", spread)
        return None
    return bg


def remove_uniform_background(image: Image.Image) -> tuple[Image.Image, bool]:
    """単色背景を色キーで抜く。単色でない・結果が破綻した場合は (元画像, False)。

    枠から連結している背景色領域だけを塗りつぶすため、キャラ内部にある背景と
    同色の部分(白い服など)は残る。
    """
    rgba = image.convert("RGBA")
    rgb = np.asarray(rgba)[..., :3].astype(np.int16)
    bg = detect_uniform_background(rgb)
    if bg is None:
        return image, False

    distance = np.abs(rgb - bg).max(axis=2)
    components, _ = label(distance <= UNIFORM_BG_TOLERANCE)
    touching = set(components[0]) | set(components[-1])
    touching |= set(components[:, 0]) | set(components[:, -1])
    touching.discard(0)
    foreground = binary_fill_holes(~np.isin(components, list(touching)))

    ratio = float(foreground.mean())
    if not UNIFORM_BG_MIN_FOREGROUND <= ratio <= UNIFORM_BG_MAX_FOREGROUND:
        logger.warning("Colour key produced an implausible foreground (%.1f%%); falling back.", ratio * 100)
        return image, False

    # アンチエイリアスのふちは色キーでは背景側に落ちる。前景を少し広げた帯に
    # 限って色差から半透明を復元し、ジャギーを避ける。
    soft = np.clip(
        (distance - UNIFORM_BG_SOFT_FLOOR) / (UNIFORM_BG_TOLERANCE - UNIFORM_BG_SOFT_FLOOR),
        0.0,
        1.0,
    )
    fringe = binary_dilation(foreground, iterations=2)
    alpha = np.where(foreground, 1.0, np.where(fringe, soft, 0.0))

    result = np.dstack([np.asarray(rgba)[..., :3], (alpha * 255).astype(np.uint8)])
    logger.info("Removed a uniform background %s by colour key (foreground %.1f%%).", tuple(bg.astype(int)), ratio * 100)
    return Image.fromarray(result, "RGBA"), True


# 既に背景が抜かれていると判断する透明画素の割合。ふちの薄いアンチエイリアス
# だけで誤判定しないよう、ある程度まとまった透明領域を要求する。
TRANSPARENT_PIXEL_RATIO = 0.05


def has_removed_background(image: Image.Image) -> bool:
    """入力が既に背景除去済み(実質的な透明領域を持つ)かどうか。

    透明PNGにさらに rembg をかけても得るものは無く、**前景を削るだけ**になる
    (実測: 背景除去済み画像に再適用して前景の1.8%が消失)。特に背景と同系色の
    細い部位は丸ごと欠落しうる。
    """
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return False
    alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    return float((alpha < 128).mean()) >= TRANSPARENT_PIXEL_RATIO


def preprocess_image(
    data: bytes,
    max_bytes: int,
    remove_bg: bool = True,
    size: int = TARGET_SIZE,
) -> tuple[Image.Image, Image.Image, bool]:
    """アップロードデータから (元画像RGBA, 前処理後画像, 背景除去適用有無) を返す。"""
    original = load_and_validate_image(data, max_bytes)

    processed = original
    bg_removed = False
    if remove_bg:
        if has_removed_background(original):
            # 既に抜けている画像に rembg をかけると前景を削るだけになる
            logger.info("Input already has a transparent background; skipping rembg.")
        else:
            # 単色背景は色キーの方が確実。推定に頼る rembg は最後の手段。
            processed, bg_removed = remove_uniform_background(original)
            if not bg_removed:
                processed, bg_removed = remove_background(original)

    processed = resize_to_square(processed, size=size)
    return original, processed, bg_removed
