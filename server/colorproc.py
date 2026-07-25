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

# 画像u(横, 0=左..1=右)からメッシュY座標への符号(左右側面ビュー用)。
# カメラの右手方向から決まる: 左側面(カメラ +X 側)は u が +Y、
# 右側面(カメラ -X 側)は u が -Y に対応する。逆に見える場合はここを反転する。
_U_TO_Y_SIGN = 1.0

# 各ビューで見える頂点の法線方向(=そのビューのカメラがある側)。
# メッシュは Z-up、正面が -Y、キャラクターの左が +X。
# したがって「左側面図」はキャラの左側 (+X) から見た画像を指す。
_VIEW_NORMALS = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (1.0, 0.0, 0.0),
    "right": (-1.0, 0.0, 0.0),
}

# 画像の横方向 u が対応するメッシュ軸 (0=X, 1=Y) と向き。
_VIEW_U_AXIS = {
    "front": (0, 1.0),
    "back": (0, -1.0),
    "left": (1, 1.0),
    "right": (1, -1.0),
}

VIEW_NAMES = tuple(_VIEW_NORMALS)

# 正面・背面を優先し、左右側面は補完に使う。
#
# 4ビューを同格に扱うと、丸い顔は頬の法線が横を向くため**顔まわりの55%が側面画像に
# 置き換わり**、目や口元がぼやけて崩れた(実測: 顔まわり28857頂点のうち front は30%
# だけ)。側面画像は見切れをシルエット照合で外挿しており位置精度も落ちる。
_PRIMARY_VIEWS = ("front", "back")
_SECONDARY_VIEWS = ("left", "right")

# 側面画像がある場合に、正面・背面が担当する範囲を狭めるしきい値。
# 既定の 0.10 は法線が視線から84°ずれていても前面扱いにするため、ほぼ真横の面まで
# 前後の色が浅い角度で引き伸ばされ、側面に黒い継ぎ目ができる(実測: ほぼ真横を向く
# 18100頂点のうち79%が前後担当のままだった)。0.5(=60°以内)なら真横の98%が側面へ
# 回り、正面を正対して向く顔は前面のまま残る。
_PRIMARY_VIEW_MIN_DOT_WITH_SIDES = 0.5

# ビュー判定に使う法線しきい値。どのビューにも十分向いていない頂点
# (真上・真下を向く面など)はベース色にする。
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


def _view_u_axis(view: str) -> tuple[int, float]:
    """ビューに対する (画像uが対応するメッシュ軸, 向き) を返す。"""
    axis, sign = _VIEW_U_AXIS[view]
    return axis, sign * (_U_TO_X_SIGN if axis == 0 else _U_TO_Y_SIGN)


# シルエット照合に使う分割数と、採用に必要な相関の下限。
_SILHOUETTE_BINS = 96
_SILHOUETTE_MIN_CORRELATION = 0.5


def _subject_touches_vertical_edges(image: Image.Image) -> bool:
    """被写体が画像の上端または下端に接しているか(=見切れの疑い)。"""
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8) > 0
    if not alpha.any():
        return False
    return bool(alpha[0].any() or alpha[-1].any())


def _mesh_silhouette_profile(
    mesh: trimesh.Trimesh, axis: int, bins: int = _SILHOUETTE_BINS
) -> np.ndarray:
    """メッシュを高さ方向に等分し、各段の横幅(指定軸の広がり)を返す。"""
    vertices = mesh.vertices
    z = vertices[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    edges = np.linspace(z_min, z_max, bins + 1)
    index = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, bins - 1)

    widths = np.zeros(bins)
    filled = np.flatnonzero(np.bincount(index, minlength=bins))
    for i in filled:
        values = vertices[index == i, axis]
        widths[i] = float(values.max() - values.min())

    # 頂点が疎なメッシュでは空の段ができ、そこが幅0の切れ込みに見えてしまう。
    # 実在する段から補間して埋める。
    if len(filled) >= 2 and len(filled) < bins:
        widths = np.interp(np.arange(bins), filled, widths[filled])
    return widths


def _image_silhouette_profile(image: Image.Image) -> np.ndarray:
    """画像の各行について、不透明画素の横方向の広がりを返す。"""
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8) > 0
    widths = np.zeros(alpha.shape[0])
    rows = np.flatnonzero(alpha.any(axis=1))
    for y in rows:
        xs = np.flatnonzero(alpha[y])
        widths[y] = float(xs[-1] - xs[0])
    return widths


def _align_vertical_by_silhouette(
    mesh: trimesh.Trimesh, image: Image.Image, axis: int
) -> Optional[tuple[float, float]]:
    """メッシュ上端・下端が画像のどの行に対応するかをシルエット照合で求める。

    被写体が枠からはみ出している(見切れている)画像では、被写体バウンディング
    ボックスにメッシュ全高を合わせる従来の方法が破綻する。実測: 側面図の
    縦占有が 767/768 で上下とも見切れており、そのまま使うと色が縦に約4割ずれる。

    横方向は見切れていないことが多いので、**横幅から倍率を決め、縦のオフセット
    だけを探索する**。メッシュの段ごとの横幅と画像の行ごとの横幅を相関で
    突き合わせ、最も一致する位置を採る。

    Returns:
        (上端の行, 下端の行) を画像高さで正規化した値。決められなければ None。
    """
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8) > 0
    if not alpha.any():
        return None
    height, width = alpha.shape

    columns = np.flatnonzero(alpha.any(axis=0))
    subject_width = float(columns[-1] - columns[0])
    mesh_width = float(mesh.vertices[:, axis].max() - mesh.vertices[:, axis].min())
    if subject_width <= 0 or mesh_width <= 0:
        return None

    pixels_per_unit = subject_width / mesh_width
    mesh_height_px = float(
        mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min()
    ) * pixels_per_unit
    if not np.isfinite(mesh_height_px) or mesh_height_px < 8:
        return None

    mesh_profile = _mesh_silhouette_profile(mesh, axis)
    image_profile = _image_silhouette_profile(image)

    # メッシュ側プロファイルを画像の行ピッチへ引き伸ばす(上端=最大Z)
    sample_count = int(round(mesh_height_px))
    positions = np.linspace(0, len(mesh_profile) - 1, sample_count)
    stretched = np.interp(positions, np.arange(len(mesh_profile)), mesh_profile[::-1])

    best_score, best_top = -np.inf, None
    for top in range(-sample_count + 8, height - 8):
        lo, hi = max(top, 0), min(top + sample_count, height)
        if hi - lo < max(16, sample_count * 0.3):
            continue
        a = stretched[lo - top : hi - top]
        b = image_profile[lo:hi]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        score = float(np.corrcoef(a, b)[0, 1])
        if score > best_score:
            best_score, best_top = score, top

    if best_top is None or best_score < _SILHOUETTE_MIN_CORRELATION:
        return None
    return best_top / height, (best_top + sample_count) / height


def _project_image_colors(mesh: trimesh.Trimesh, image: Image.Image, *, view: str) -> np.ndarray:
    """単一ビュー画像を直交投影し、全頂点分のRGBAカラーを返す。

    正面(-Y側)/背面(+Y側)は メッシュXZ 平面へ、左側面(+X側)/右側面(-X側)は
    メッシュYZ 平面へ投影する。カメラの右手方向が変わるので、ビューごとに
    横方向の対応軸と符号が変わる(`_VIEW_U_AXIS`)。
    """
    if view not in _VIEW_U_AXIS:
        raise ValueError(f"viewは{sorted(_VIEW_U_AXIS)}のいずれかである必要があります(got {view})。")
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    w, h = image.size
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)  # (h, w, 3)
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)  # (h, w)
    axis, u_sign = _view_u_axis(view)

    u_min, u_max, v_min, v_max = _subject_bbox_uv(image)

    # 被写体が枠の上下に接している=見切れている可能性がある。その場合は
    # 被写体bboxに全高を合わせると縦がずれるため、シルエット照合で
    # メッシュ上端・下端に対応する行を求め直す。
    if _subject_touches_vertical_edges(image):
        aligned = _align_vertical_by_silhouette(mesh, image, axis)
        if aligned is not None:
            v_min, v_max = aligned

    vertices = mesh.vertices
    bounds = mesh.bounds
    a_min, a_max = bounds[0][axis], bounds[1][axis]
    z_min, z_max = bounds[0][2], bounds[1][2]
    a_extent = max(a_max - a_min, 1e-9)
    z_extent = max(z_max - z_min, 1e-9)

    # 横方向の軸 -> 正規化u (被写体bbox基準)。
    # 例) front は _U_TO_X_SIGN=+1 (実生成検証済み) で u=0(左端)が-X側、
    #     back はカメラが反対側(+Y)にあるため左右対応が反転する。
    a_norm = (vertices[:, axis] - a_min) / a_extent  # 0..1
    u_norm = a_norm if u_sign > 0 else 1.0 - a_norm
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


def _vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    normals = np.asarray(mesh.vertex_normals)
    if normals.shape != (len(mesh.vertices), 3):
        mesh = mesh.copy()
        mesh.fix_normals()
        normals = np.asarray(mesh.vertex_normals)
    return normals


def _view_vertex_masks(
    mesh: trimesh.Trimesh,
    views: list[str],
    threshold: float = _VIEW_NORMAL_THRESHOLD,
) -> dict[str, np.ndarray]:
    """各頂点を、法線が最もよく向いているビューへ割り当てる。

    どのビューにも `_VIEW_NORMAL_THRESHOLD` を超えて向いていない頂点
    (真上・真下を向く面など)はどのマスクにも入らず、ベース色のままになる。

    正面・背面だけを渡した場合は従来の
    `normal_y < -threshold` / `normal_y > threshold` と同じ結果になる。
    """
    normals = _vertex_normals(mesh)
    scores = np.stack([normals @ np.asarray(_VIEW_NORMALS[v]) for v in views])
    best = scores.argmax(axis=0)
    visible = scores.max(axis=0) > threshold
    return {view: (best == i) & visible for i, view in enumerate(views)}


def _front_back_vertex_masks(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """頂点法線から正面側・背面側の頂点マスクを返す(後方互換用)。"""
    masks = _view_vertex_masks(mesh, ["front", "back"])
    return masks["front"], masks["back"]


def project_multiview_colors(
    mesh: trimesh.Trimesh,
    front_image: Image.Image,
    back_image: Optional[Image.Image] = None,
    left_image: Optional[Image.Image] = None,
    right_image: Optional[Image.Image] = None,
    base_color: Optional[np.ndarray] = None,
) -> np.ndarray:
    """複数ビューの画像を頂点法線で振り分けて投影する。

    - まず正面・背面のうち、正対して向いている頂点へそれぞれの画像を当てる。
    - 残りを左右側面の画像で埋める。
    - それでも残る頂点は、ゆるいしきい値で再度前後に拾わせる。
    - どこにも当てはまらない頂点(真上・真下など)はベース色にする。

    左右を正面・背面と同格に扱うと丸い顔が側面画像に置き換わって崩れるため、
    側面はあくまで補完に使う(`_SECONDARY_VIEWS` / `_PRIMARY_VIEW_MIN_DOT_WITH_SIDES`)。
    """
    if base_color is None:
        base_rgb = _DEFAULT_BASE_COLOR
    else:
        base_rgb = np.asarray(base_color[:3], dtype=np.uint8)

    colors = np.empty((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, :3] = base_rgb
    colors[:, 3] = 255

    images = {
        "front": front_image,
        "back": back_image,
        "left": left_image,
        "right": right_image,
    }
    available = [view for view in VIEW_NAMES if images[view] is not None]
    primary = [view for view in available if view in _PRIMARY_VIEWS]
    secondary = [view for view in available if view in _SECONDARY_VIEWS]

    # 側面で補完できるなら、前後は正対している面だけを担当する
    threshold = _PRIMARY_VIEW_MIN_DOT_WITH_SIDES if secondary else _VIEW_NORMAL_THRESHOLD
    masks = _view_vertex_masks(mesh, primary, threshold) if primary else {}
    claimed = np.zeros(len(mesh.vertices), dtype=bool)
    for mask in masks.values():
        claimed |= mask

    if secondary:
        for view, mask in _view_vertex_masks(mesh, secondary).items():
            masks[view] = mask & ~claimed
            claimed |= masks[view]
        # 側面でも埋まらなかった分は、当初のゆるいしきい値で前後に拾わせる
        for view, mask in _view_vertex_masks(mesh, primary).items():
            masks[view] |= mask & ~claimed

    for view in available:
        view_colors = _project_image_colors(mesh, images[view], view=view)
        colors[masks[view]] = view_colors[masks[view]]

    return colors


def _bbox_normalize(vertices: np.ndarray) -> np.ndarray:
    """頂点集合をバウンディングボックス基準で0..1に正規化する(退化軸は0)。"""
    v = np.asarray(vertices, dtype=np.float64)
    v_min = v.min(axis=0)
    extent = v.max(axis=0) - v_min
    extent = np.where(extent > 1e-12, extent, 1.0)
    return (v - v_min) / extent


def transfer_vertex_colors_nearest(
    src_vertices: np.ndarray,
    src_colors: np.ndarray,
    dst_vertices: np.ndarray,
    align_bbox: bool = False,
) -> np.ndarray:
    """最近傍頂点で頂点カラーを転写する(GPU不要の純関数)。

    Pixal3Dジェネレータ等、生成直後のrawメッシュにテクスチャ由来の頂点カラーを
    付与した後、`meshproc.process` が浮遊小部品除去・簡略化等で頂点集合を
    再構築してしまい元の頂点カラーが失われるため、後処理後メッシュの各頂点に
    対し raw メッシュの最近傍頂点の色を転写する(scipy cKDTree使用)。

    Args:
        src_vertices: (N, 3) rawメッシュの頂点座標。
        src_colors: (N, 3) or (N, 4) rawメッシュの頂点カラー(uint8推奨)。
        dst_vertices: (M, 3) 後処理後メッシュの頂点座標。
        align_bbox: Trueの場合、両頂点集合をそれぞれのバウンディングボックスで
            0..1に正規化してから最近傍探索する。`meshproc.process` はスケール
            (mm化)・接地・センタリングを行うため raw/後処理後メッシュは座標系が
            異なるが、バウンディングボックス正規化でこの相似変換を吸収する
            (浮遊小部品除去によるbboxのわずかな差は許容誤差とする)。

    Returns:
        (M, C) 転写後の頂点カラー配列(src_colorsと同じdtype・チャンネル数)。
    """
    if len(src_vertices) == 0:
        raise ValueError("src_verticesが空です。")
    if len(src_vertices) != len(src_colors):
        raise ValueError(
            f"src_verticesとsrc_colorsの長さが一致しません({len(src_vertices)} != {len(src_colors)})。"
        )

    if align_bbox:
        src = _bbox_normalize(src_vertices)
        dst = _bbox_normalize(dst_vertices)
    else:
        src = np.asarray(src_vertices, dtype=np.float64)
        dst = np.asarray(dst_vertices, dtype=np.float64)

    tree = cKDTree(src)
    _, nn_idx = tree.query(dst)
    return np.asarray(src_colors)[nn_idx]


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
