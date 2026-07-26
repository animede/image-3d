"""texgen のテクスチャを元画像の全解像度で上書きする (SPEC.md §3.9)。

texgen(Hunyuan3D-2 paint)のテクスチャは、参照画像がどれだけ精細でも
**512px の情報しか運べない**:

    参照 1024 → delight で 512 (dehighlight_utils.py:70)
             → マルチビュー生成 512 (multiview_utils.py:28 `view_size = 512`)
             → ×4 に単純拡大 (pipelines.py:226。超解像は上流でコメントアウト)
             → 2048 のアトラスへベイク

体全体を 512 に詰めるので顔は実質 150px 程度になり、毛並みやステッチのような
細部は原理的に残らない(実測: 帽子の赤が (175,50,70)→(216,1,20) と平坦化、
腕の毛は色こそ合うが質感が消える)。

一方で正面・背面の参照画像は元の解像度でそのまま手元にある。そこで
**カメラを正対して向いているテクセルだけ**を参照画像から直接サンプリングし直し、
texgen の結果へ上書きする。texgen は「参照が無い/浅い角度でしか見えない範囲」の
担保に回る。

視線に対し浅い角度の面を上書きしないのが肝で、そこはフチ画素を引き伸ばして
サンプリングしてしまい黒い筋の原因になる(colorproc の
`_LOW_CONFIDENCE_DOT_THRESHOLD` と同じ理由)。境目が出ないよう、法線の向きに
応じて texgen 側へなめらかに戻す。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt

from . import colorproc

logger = logging.getLogger(__name__)

# 参照画像を全面的に信用する法線しきい値(視線との内積)。
# 0.75 ≒ 視線から41°以内。正対する顔・胸・腕の前面がここに入る。
FULL_CONFIDENCE_DOT = 0.75

# 上書きを完全にやめる法線しきい値。0.35 ≒ 視線から70°。colorproc が
# 「投影を信用しない」と判断するのと同じ値。ここから上へ向けて線形に
# 重みを上げ、texgen との境目が線にならないようにする。
MIN_CONFIDENCE_DOT = 0.35

# 面あたりのサンプル数の目安(テクセル面積の何倍を撒くか)。
# ランダムなバリセントリックサンプリングなので取りこぼしが出るが、
# 残った穴は `HOLE_FILL_RADIUS_TEXELS` で埋める。
SAMPLES_PER_TEXEL = 4.0

# 1面あたりのサンプル数の上限(極端に大きい面でメモリが跳ねるのを防ぐ)。
MAX_SAMPLES_PER_FACE = 4096

# メモリを一定に保つための面のチャンクサイズ。
FACE_CHUNK = 20000

# サンプルの取りこぼしと UV チャート外周(ガター)を埋める距離。
# ここを埋めておかないと、バイリニア補間がチャートの外から texgen の色を
# 拾って細い継ぎ目になる。
HOLE_FILL_RADIUS_TEXELS = 4


@dataclass
class RefineStats:
    """精細化の結果。ログと job の警告に使う。"""

    applied: bool = False
    views: list[str] = field(default_factory=list)
    texture_size: tuple[int, int] = (0, 0)
    refined_texel_ratio: float = 0.0
    mean_blend_weight: float = 0.0
    reason: Optional[str] = None


def _extract_texture_image(visual: Any) -> Optional[Image.Image]:
    material = getattr(visual, "material", None)
    if material is not None:
        for attr in ("image", "baseColorTexture"):
            image = getattr(material, attr, None)
            if image is not None:
                return image
    return getattr(visual, "image", None)


def _set_texture_image(visual: Any, image: Image.Image) -> bool:
    """テクスチャ画像を差し替える。SimpleMaterial / PBRMaterial の両方に対応。"""
    material = getattr(visual, "material", None)
    if material is not None:
        if getattr(material, "image", None) is not None:
            material.image = image
            return True
        if getattr(material, "baseColorTexture", None) is not None:
            material.baseColorTexture = image
            return True
    if getattr(visual, "image", None) is not None:
        visual.image = image
        return True
    return False


def _face_sample_counts(uv_texels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """各面のテクセル面積から、撒くサンプル数を決める。"""
    tri = uv_texels[faces]  # (F, 3, 2)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    counts = np.ceil(area * SAMPLES_PER_TEXEL).astype(np.int64)
    return np.clip(counts, 1, MAX_SAMPLES_PER_FACE)


def _barycentric(n: int, rng: np.random.Generator) -> np.ndarray:
    """三角形内に一様な重心座標を n 個返す (n, 3)。"""
    r1, r2 = rng.random(n), rng.random(n)
    su = np.sqrt(r1)
    return np.column_stack([1.0 - su, su * (1.0 - r2), su * r2]).astype(np.float32)


def _blend_weight(dot: np.ndarray) -> np.ndarray:
    """法線と視線の内積 -> 参照画像を信用する重み (0..1)。"""
    span = max(FULL_CONFIDENCE_DOT - MIN_CONFIDENCE_DOT, 1e-9)
    return np.clip((dot - MIN_CONFIDENCE_DOT) / span, 0.0, 1.0).astype(np.float32)


def refine_texture_with_references(
    mesh: trimesh.Trimesh,
    references: dict[str, Image.Image],
) -> RefineStats:
    """texgen 済みメッシュのテクスチャを、参照画像の全解像度で上書きする。

    メッシュは **その場で** 書き換える(テクスチャ画像のみ差し替え、頂点・UVは不変)。

    Args:
        mesh: UV とテクスチャを持つ texgen の出力。Z-up・正面が -Y。
        references: ビュー名(`colorproc.VIEW_NAMES`)-> 背景除去済み参照画像。

    Returns:
        RefineStats。適用できなかった場合は `applied=False` と `reason`。
    """
    stats = RefineStats(views=sorted(references))

    views = [v for v in references if v in colorproc.VIEW_NAMES]
    if not views:
        stats.reason = "参照画像がありません。"
        return stats

    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    texture = _extract_texture_image(visual)
    if uv is None or texture is None:
        stats.reason = "メッシュにUVまたはテクスチャがありません。"
        return stats

    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) != len(mesh.vertices):
        stats.reason = f"UV数({len(uv)})が頂点数({len(mesh.vertices)})と一致しません。"
        return stats

    base = np.asarray(texture.convert("RGB"), dtype=np.float32)  # (h, w, 3)
    h, w = base.shape[:2]
    stats.texture_size = (w, h)

    # UV(v=0が下端) -> テクセル座標(行0が上端)
    uv_texels = np.column_stack(
        [np.clip(uv[:, 0], 0.0, 1.0) * (w - 1), (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (h - 1)]
    )

    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if normals.shape != vertices.shape:
        mesh_fixed = mesh.copy()
        mesh_fixed.fix_normals()
        normals = np.asarray(mesh_fixed.vertex_normals, dtype=np.float64)

    # ビューごとに、参照画像の画素と「フチを除いた信頼できる画素」を用意する。
    view_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image]] = {}
    for view in views:
        image = references[view]
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        trusted = colorproc._trusted_pixel_mask(alpha)
        view_data[view] = (rgb, trusted, np.asarray(colorproc._VIEW_NORMALS[view]), image)

    accum_rgb = np.zeros((h * w, 3), dtype=np.float32)
    accum_weight = np.zeros(h * w, dtype=np.float32)
    accum_count = np.zeros(h * w, dtype=np.float32)

    counts_all = _face_sample_counts(uv_texels, faces)
    rng = np.random.default_rng(0)

    for start in range(0, len(faces), FACE_CHUNK):
        chunk = faces[start : start + FACE_CHUNK]
        counts = counts_all[start : start + FACE_CHUNK]
        total = int(counts.sum())
        if total == 0:
            continue

        face_idx = np.repeat(np.arange(len(chunk)), counts)
        bary = _barycentric(total, rng)[:, :, None]  # (S, 3, 1)

        tri_v = vertices[chunk[face_idx]]  # (S, 3, 3)
        tri_n = normals[chunk[face_idx]]
        tri_uv = uv_texels[chunk[face_idx]]

        points = (tri_v * bary).sum(axis=1)  # (S, 3)
        sample_normals = (tri_n * bary).sum(axis=1)
        norm = np.linalg.norm(sample_normals, axis=1, keepdims=True)
        sample_normals /= np.maximum(norm, 1e-9)
        sample_uv = (tri_uv * bary).sum(axis=1)  # (S, 2)

        tx = np.clip(sample_uv[:, 0].astype(np.int64), 0, w - 1)
        ty = np.clip(sample_uv[:, 1].astype(np.int64), 0, h - 1)
        flat = ty * w + tx

        # 各サンプルを、最も正対して見えるビューに割り当てる。
        dots = np.stack([sample_normals @ view_data[v][2] for v in views], axis=1)  # (S, V)
        best = np.argmax(dots, axis=1)
        best_dot = dots[np.arange(len(dots)), best]

        weight = _blend_weight(best_dot)
        np.add.at(accum_count, flat, 1.0)

        for vi, view in enumerate(views):
            sel = (best == vi) & (weight > 0.0)
            if not sel.any():
                continue
            rgb, trusted, _, image = view_data[view]
            px, py = colorproc.project_points_to_pixels(mesh, image, points[sel], view=view)
            # フチ・透明画素に落ちたサンプルは信用しない(黒筋の原因)。
            ok = trusted[py, px]
            if not ok.any():
                continue
            idx = np.flatnonzero(sel)[ok]
            wgt = weight[idx]
            np.add.at(accum_rgb, flat[idx], rgb[py[ok], px[ok]] * wgt[:, None])
            np.add.at(accum_weight, flat[idx], wgt)

    covered = accum_weight > 0.0
    if not covered.any():
        stats.reason = "参照画像から上書きできるテクセルがありませんでした。"
        return stats

    refined = np.zeros_like(accum_rgb)
    refined[covered] = accum_rgb[covered] / accum_weight[covered, None]
    # テクセルごとの混合率 = そのテクセルに落ちたサンプルの平均重み。
    # 正対する面ほど 1 に近づき、浅い角度では texgen 側が残る。
    blend = np.zeros(h * w, dtype=np.float32)
    counted = accum_count > 0
    blend[counted] = accum_weight[counted] / accum_count[counted]

    refined = refined.reshape(h, w, 3)
    blend = blend.reshape(h, w)
    covered_2d = covered.reshape(h, w)

    # 取りこぼしと UV チャート外周を、最近傍の上書き済みテクセルで埋める。
    # 埋めないとバイリニア補間がチャート外の texgen 色を拾って継ぎ目になる。
    if not covered_2d.all():
        distance, (iy, ix) = distance_transform_edt(
            ~covered_2d, return_distances=True, return_indices=True
        )
        fill = (distance > 0) & (distance <= HOLE_FILL_RADIUS_TEXELS)
        refined[fill] = refined[iy[fill], ix[fill]]
        blend[fill] = blend[iy[fill], ix[fill]]

    result = base * (1.0 - blend[..., None]) + refined * blend[..., None]
    new_texture = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")

    if not _set_texture_image(visual, new_texture):
        stats.reason = "テクスチャ画像を差し替えられませんでした。"
        return stats

    stats.applied = True
    stats.refined_texel_ratio = float((blend > 0).mean())
    stats.mean_blend_weight = float(blend[blend > 0].mean())
    logger.info(
        "Refined the texgen atlas from %s references at full resolution: "
        "%.1f%% of %dx%d texels touched (mean blend %.2f).",
        "+".join(views),
        stats.refined_texel_ratio * 100,
        w,
        h,
        stats.mean_blend_weight,
    )
    return stats
