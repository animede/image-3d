"""画像前処理 (SPEC.md §3.1 / §3.2, FR-1, FR-2)。

- アップロード画像の検証(フォーマット・サイズ)
- リサイズ・正方形化
- 背景除去(rembg)。未導入環境では自動スキップ(NFR-5)。
"""
from __future__ import annotations

import io
import logging

from PIL import Image, UnidentifiedImageError

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
        processed, bg_removed = remove_background(original)

    processed = resize_to_square(processed, size=size)
    return original, processed, bg_removed
