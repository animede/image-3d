"""4色カラープリント対応 (SPEC.md §3.7 / FR-8)。

テクスチャ生成AIは使用せず、以下の簡易パイプラインで頂点カラーを付与する:

1. `project_multiview_colors`: 背景除去済み入力画像をメッシュ正面軸に沿って
    直交投影し、正面側の頂点にRGBAカラーを割り当てる。背面画像がある場合は
    背面側の頂点に背面画像を投影し、無い場合はベース色にする。
2. `quantize`: 頂点カラーを scipy.cluster.vq.kmeans2 で `n_colors` (2〜4) 色に
   量子化する。
3. `split_by_color`: 面ごとの多数決で色ラベルを決め、色ごとにサブメッシュへ
   分割する(全サブメッシュの面の合併 = 元メッシュ)。
4. `palette_stats`: パレット(HEX)と色ごとの面数比率を返す。

座標系の重要事実 (server/generators/hunyuan3d.py 参照):
    メッシュは Z-up (高さ=Z、床=z=0)。hy3dgen 自体の出力は Y-up・カメラ視線
    方向 +Z だが、hunyuan3d.py で X軸まわり +90° 回転して Z-up に変換して
    いるため、変換後は **キャラクターの正面は -Y 方向を向く**
    (カメラは -Y 側から +Y 方向を見て撮影したとみなせる)。
    よって画像→メッシュの投影対応は:
        画像の横方向 u (0=左 .. 1=右) → メッシュ X (増加方向は実生成検証で確定)
        画像の縦方向 v (0=上 .. 1=下) → メッシュ Z (上下反転、v=0が高いZに対応)
    メッシュのXZバウンディングボックスを画像の被写体バウンディングボックス
    (アルファ>0領域、なければ画像全体)にフィットさせる。
    `project_colors` は互換用に従来通り全頂点へ正面投影する。
    実ジョブのカラーモードでは `project_multiview_colors` を使い、正面色が
    背面全面へ回り込まないよう、頂点法線で正面/背面を分ける。

    実生成検証(momo.png, hunyuan3d, GPU実機。README/報告参照)の結果、
    画像の u=0(左端)がメッシュの -X 側、u=1(右端)が +X 側に対応する
    ことを確認した(_U_TO_X_SIGN=+1)。逆に見える場合はこの符号を反転すること。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

# 画像u(横, 0=左..1=右)からメッシュX座標への符号。
# 実生成検証(momo.png, hunyuan3d, GPU実機)の結果、u=0(画像左端)が
# メッシュの -X 側に、u=1(画像右端)が +X 側に対応することを確認した(+1)。
# 逆に見える場合はここを -1 に反転する。
_U_TO_X_SIGN = 1.0

# 正面/背面判定に使う頂点法線Y成分のしきい値。
# 正面は -Y 方向を向くため normal_y < -threshold、背面は normal_y > threshold。
_VIEW_NORMAL_THRESHOLD = 0.10

# 背面画像が無い場合や側面/上下など明確に正面・背面でない頂点へ使うベース色。
_DEFAULT_BASE_COLOR = np.array([220, 220, 220], dtype=np.uint8)


def _subject_bbox_uv(image: Image.Image) -> tuple[float, float, float, float]:
    """画像内の被写体(アルファ>0領域)のバウンディングボックスを
    正規化uv座標 (u_min, u_max, v_min, v_max) (0..1) で返す。
    アルファチャンネルが無い、または全域が不透明/透明な場合は画像全体を返す。
    """
    w, h = image.size
    if image.mode == "RGBA":
        alpha = np.asarray(image.getchannel("A"))
        ys, xs = np.where(alpha > 0)
        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            # 1pxの被写体などの退化ケースを避けるため最低限の幅を確保
            if x1 > x0 and y1 > y0:
                return (x0 / w, (x1 + 1) / w, y0 / h, (y1 + 1) / h)
    return (0.0, 1.0, 0.0, 1.0)


def _project_image_colors(mesh: trimesh.Trimesh, image: Image.Image, *, view: str) -> np.ndarray:
    """単一ビュー画像をメッシュXZへ直交投影し、全頂点分のRGBAカラーを返す。

    `view="front"` は -Y 側から見た画像、`view="back"` は +Y 側から見た画像として
    扱う。背面ビューはカメラ方向が反対になるため、X→u の対応を反転する。
    """
    if view not in ("front", "back"):
        raise ValueError(f"viewは'front'または'back'である必要があります(got {view})。")
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    w, h = image.size
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)  # (h, w, 3)
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)  # (h, w)

    u_min, u_max, v_min, v_max = _subject_bbox_uv(image)

    vertices = mesh.vertices
    bounds = mesh.bounds
    x_min, x_max = bounds[0][0], bounds[1][0]
    z_min, z_max = bounds[0][2], bounds[1][2]
    x_extent = max(x_max - x_min, 1e-9)
    z_extent = max(z_max - z_min, 1e-9)

    # メッシュX -> 正規化u (被写体bbox基準)。
    # front: _U_TO_X_SIGN=+1 (実生成検証済み): u=0(左端)が-X側、u=1(右端)が+X側。
    # back: カメラが反対側(+Y)のため、左右対応を反転する。
    x_norm = (vertices[:, 0] - x_min) / x_extent  # 0..1, 0=-X側, 1=+X側
    u_sign = _U_TO_X_SIGN if view == "front" else -_U_TO_X_SIGN
    if u_sign < 0:
        u_norm = 1.0 - x_norm  # +X側(x_norm=1) -> u=0(画像左端) (反転版)
    else:
        u_norm = x_norm  # -X側(x_norm=0) -> u=0(画像左端)
    u = u_min + u_norm * (u_max - u_min)

    # メッシュZ -> 正規化v (上下反転: Z最大=画像上端 v=0)
    z_norm = (vertices[:, 2] - z_min) / z_extent  # 0..1, 0=床, 1=頭頂
    v = v_min + (1.0 - z_norm) * (v_max - v_min)

    px = np.clip((u * w).astype(np.int64), 0, w - 1)
    py = np.clip((v * h).astype(np.int64), 0, h - 1)

    sampled_rgb = rgb[py, px]  # (N, 3)
    sampled_alpha = alpha[py, px]  # (N,)

    # 透明画素に投影された頂点は最近傍の不透明画素の色で埋める
    opaque_mask = sampled_alpha > 0
    if opaque_mask.any() and not opaque_mask.all():
        opaque_ys, opaque_xs = np.where(alpha > 0)
        tree = cKDTree(np.column_stack([opaque_ys, opaque_xs]))
        missing_idx = np.where(~opaque_mask)[0]
        _, nn_idx = tree.query(np.column_stack([py[missing_idx], px[missing_idx]]))
        sampled_rgb[missing_idx] = rgb[opaque_ys[nn_idx], opaque_xs[nn_idx]]
    elif not opaque_mask.any():
        # 完全に透明(アルファ情報が無い画像等)な場合は投影色をそのまま使う
        pass

    colors = np.empty((len(vertices), 4), dtype=np.uint8)
    colors[:, :3] = sampled_rgb
    colors[:, 3] = 255
    return colors


def project_colors(mesh: trimesh.Trimesh, image: Image.Image) -> np.ndarray:
    """背景除去済み入力画像をメッシュ正面軸に沿って全頂点へ直交投影する。

    互換用の従来方式。実ジョブの `color_mode=color4` では、背面への正面色の
    回り込みを避けるため `project_multiview_colors` を使用する。

    Args:
        mesh: Z-up・正面が-Y方向を向くメッシュ(hunyuan3d.py の出力座標系)。
        image: 背景除去済みの入力画像(RGBA推奨。RGBの場合は不透明として扱う)。

    Returns:
        (N, 4) uint8 の頂点カラー配列(RGBA)。
    """
    return _project_image_colors(mesh, image, view="front")


def _front_back_vertex_masks(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """頂点法線から正面側・背面側の頂点マスクを返す。"""
    normals = np.asarray(mesh.vertex_normals)
    if normals.shape != (len(mesh.vertices), 3):
        mesh = mesh.copy()
        mesh.fix_normals()
        normals = np.asarray(mesh.vertex_normals)
    y = normals[:, 1]
    front_mask = y < -_VIEW_NORMAL_THRESHOLD
    back_mask = y > _VIEW_NORMAL_THRESHOLD
    return front_mask, back_mask


def project_multiview_colors(
    mesh: trimesh.Trimesh,
    front_image: Image.Image,
    back_image: Optional[Image.Image] = None,
    base_color: Optional[np.ndarray] = None,
) -> np.ndarray:
    """正面/背面を分けて頂点カラーを投影する。

    - 正面側の頂点: 正面画像を投影する。
    - 背面側の頂点: 背面画像があれば背面画像を投影し、無ければベース色にする。
    - 側面/上下など正面・背面判定が曖昧な頂点: ベース色にする。

    これにより、正面画像が背面全面へ薄く回り込む従来の簡易投影を避ける。
    """
    if base_color is None:
        base_rgb = _DEFAULT_BASE_COLOR
    else:
        base_rgb = np.asarray(base_color[:3], dtype=np.uint8)

    colors = np.empty((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, :3] = base_rgb
    colors[:, 3] = 255

    front_mask, back_mask = _front_back_vertex_masks(mesh)

    front_colors = _project_image_colors(mesh, front_image, view="front")
    colors[front_mask] = front_colors[front_mask]

    if back_image is not None:
        back_colors = _project_image_colors(mesh, back_image, view="back")
        colors[back_mask] = back_colors[back_mask]

    return colors


def quantize(colors: np.ndarray, n_colors: int) -> tuple[np.ndarray, np.ndarray]:
    """頂点カラーをk-meansで `n_colors` (2〜4) 色に量子化する。

    Args:
        colors: (N, 3) or (N, 4) uint8 カラー配列(RGBまたはRGBA)。
        n_colors: 量子化後の色数(2〜4)。

    Returns:
        (palette, labels):
            palette: (K, 3) uint8 量子化パレット(空クラスタは除去済み、K<=n_colors)。
            labels: (N,) int クラスタラベル(0..K-1)。
    """
    if n_colors < 2 or n_colors > 4:
        raise ValueError(f"n_colorsは2〜4である必要があります(got {n_colors})。")

    rgb = colors[:, :3].astype(np.float64)

    n_unique = len(np.unique(rgb.reshape(-1, 3), axis=0))
    k = max(1, min(n_colors, n_unique))

    if k == 1:
        palette = np.round(rgb.mean(axis=0)).astype(np.uint8).reshape(1, 3)
        labels = np.zeros(len(rgb), dtype=np.int64)
        return palette, labels

    centroids, labels = kmeans2(rgb, k, seed=0, minit="++", missing="warn")

    # 空クラスタの除去 + ラベル振り直し
    used = np.unique(labels)
    remap = {old: new for new, old in enumerate(used)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)
    palette = np.clip(np.round(centroids[used]), 0, 255).astype(np.uint8)

    return palette, labels


def _vertex_labels_to_face_labels(mesh: trimesh.Trimesh, vertex_labels: np.ndarray) -> np.ndarray:
    """頂点ラベルから面ラベルを多数決で決定する。"""
    face_vertex_labels = vertex_labels[mesh.faces]  # (F, 3)
    face_labels = np.empty(len(mesh.faces), dtype=np.int64)
    for i in range(len(mesh.faces)):
        vals, counts = np.unique(face_vertex_labels[i], return_counts=True)
        face_labels[i] = vals[np.argmax(counts)]
    return face_labels


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(v) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def split_by_color(
    mesh: trimesh.Trimesh, labels_per_vertex: np.ndarray, palette: np.ndarray
) -> list[tuple[trimesh.Trimesh, str]]:
    """面ごとの多数決で色ラベルを決め、色ごとのサブメッシュに分割する。

    Args:
        mesh: 元メッシュ(頂点カラー投影済みである必要はない)。
        labels_per_vertex: (N,) 各頂点のクラスタラベル(quantizeの出力)。
        palette: (K, 3) uint8 パレット。

    Returns:
        [(サブメッシュ, "#rrggbb"), ...] のリスト(パレット順、空クラスタは含まない)。
        全サブメッシュの面数合計は元メッシュの面数と一致する。
    """
    face_labels = _vertex_labels_to_face_labels(mesh, labels_per_vertex)

    result: list[tuple[trimesh.Trimesh, str]] = []
    for label in range(len(palette)):
        face_mask = face_labels == label
        if not face_mask.any():
            continue
        sub_faces = mesh.faces[face_mask]
        sub = mesh.submesh([np.where(face_mask)[0]], append=True, repair=False)
        if isinstance(sub, list):
            # append=True なら通常単一メッシュが返るが、念のためフォールバック
            sub = trimesh.util.concatenate(sub) if len(sub) > 1 else sub[0]
        hex_color = _rgb_to_hex(palette[label])
        rgba = np.array(
            [*palette[label][:3], 255], dtype=np.uint8
        )
        sub.visual = trimesh.visual.ColorVisuals(
            mesh=sub, vertex_colors=np.tile(rgba, (len(sub.vertices), 1))
        )
        sub.visual.face_colors = np.tile(rgba, (len(sub.faces), 1))
        result.append((sub, hex_color))

    return result


def palette_stats(
    labels_per_vertex: np.ndarray, palette: np.ndarray, mesh: Optional[trimesh.Trimesh] = None
) -> list[dict]:
    """SPEC.md §5 `stats.palette` 形式の統計を返す(face_ratio降順)。

    面数ベースの比率を返すため `mesh` が必要(未指定時は頂点数ベースにフォールバック)。
    """
    if mesh is not None:
        face_labels = _vertex_labels_to_face_labels(mesh, labels_per_vertex)
        total = len(face_labels)
        counts = np.array([(face_labels == label).sum() for label in range(len(palette))])
    else:
        total = len(labels_per_vertex)
        counts = np.array(
            [(labels_per_vertex == label).sum() for label in range(len(palette))]
        )

    total = total or 1
    stats = []
    for label in range(len(palette)):
        if counts[label] == 0:
            continue
        stats.append(
            {
                "hex": _rgb_to_hex(palette[label]),
                "face_ratio": float(counts[label]) / float(total),
            }
        )
    stats.sort(key=lambda d: d["face_ratio"], reverse=True)
    return stats
