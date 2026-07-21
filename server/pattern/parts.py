"""パーツ自動分解 (SPEC.md §3.12 / FR-13 の「2段階構成 (4c)」1段目)。

実際のぬいぐるみは頭・胴・腕・脚・耳等を別々に縫製・詰め物して後で
縫い付ける構造のため、型紙生成の前段としてメッシュを部位単位の
ボリュームへ自動分解する。

アルゴリズム概要:
    1. **局所肉厚(Shape Diameter Function風)**: 各面についてサンプリング
       (メッシュが大きい場合は一部の面のみ)し、その重心から内向き
       (-法線方向、±30°程度のコーン内の数本)にレイキャストして対面までの
       距離の中央値を局所肉厚とする。rtree/pyembreeなど追加依存なしの
       ブルートフォースMöller-Trumboreをscipy.spatial.cKDTreeで絞り込んだ
       近傍三角形にのみ適用することで実用速度に収める(全面×全面の
       総当たりは低速すぎるため)。
    2. サンプルされなかった面へは、双対グラフ上の最短路で最も近い
       サンプル面の値を伝播(拡散)させて全面の肉厚を補間する。
    3. 肉厚を対数スケールでk-means(2クラスタ)し、「細い部位」
       (腕・脚・耳等)と「太い部位」(胴・頭等)に分ける。
    4. 双対グラフ上でクラスタごとの連結成分をパーツ候補とする。
    5. 極小パーツ(全面積2%未満)は隣接パーツへ吸収する。
       (連結成分抽出の副作用で稀に生じる、面積0に近い完全孤立した
       断片(縮退三角形等)は最近傍パーツへ吸収する。)
    6. `n_parts_hint` が指定され、現在のパーツ数がヒントに満たない場合は、
       最大パーツを「くびれ(凹二面角)」に沿うよう誘導した
       farthest-point + 多始点最短路(segment.pyと同様のVoronoi風手法)で
       2分割し、ヒントに達するまで繰り返す(=肉厚だけでは分離できない
       頭と胴の境目等を、面積順マージ/階層分割で補う)。
       ヒントより多い場合は面積の小さいパーツから順に隣接統合する。

画像誘導分解(`image_rgba` 指定時、任意):
    くびれの浅い頭・胴・脚は肉厚ベースの分解だけでは1パーツに統合されて
    しまうことがある(3D形状だけでは不十分)。入力画像(背景除去済み、
    RGBA)の色領域を手がかりに、そのようなパーツを追加分割する。
    1. `extract_image_regions`: アルファ>0画素をk-means量子化 →
       連結成分ラベリング → 極小領域マージで2D色領域ラベル画像を作る。
    2. `project_labels_to_faces`: colorproc.pyと同一の投影規約
       (正面=-Y、画像u→メッシュ+X、画像v→Z上下反転、メッシュXZ bboxを
       被写体bboxにフィット)を自前実装し、面重心を画像座標へ投影して
       2Dラベルを取得する(境界規約の重複はモジュール独立性のため。
       `server/colorproc.py` を pattern モジュールから import してはならない)。
    3. 既存のジオメトリ分解結果のうち大きいパーツ(全面積30%超)について、
       投影ラベルがそのパーツを2つ以上の有意なチャンク(各15%以上)に
       分割する場合のみ、そのチャンク境界に沿ってサブ分割する
       (`_split_subregion` の凹エッジ誘導Voronoi分割を、画像ラベルの
       重心をシードにして流用)。
    `image_rgba` が None の場合は上記をすべてスキップし、従来通りの
    ジオメトリのみの分解になる(回帰なし)。

このモジュールは純粋モジュール(server/DEVELOPMENT_POLICY.md §3.5)。
依存は numpy / scipy / trimesh のみ。PIL等の画像ライブラリはアダプタ側
(main.py)でのみ使用し、ここでは (H, W, 4) uint8 の ndarray として受け取る。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.cluster.vq import kmeans2
from scipy import ndimage as _ndimage

# 画像誘導分解のパラメータ (colorproc.py の投影規約と同一にする。
# server/colorproc.py の docstring 参照。pattern モジュールからは
# import できないため、同じ定数・ロジックをここに複製する)。
_IMG_U_TO_X_SIGN = 1.0
_IMG_MAX_REGIONS = 8
_IMG_MIN_REGION_AREA_FRACTION = 0.01  # 全画素数に対するこの比率未満は極小領域としてマージ
_IMG_LARGE_PART_AREA_FRACTION = 0.30  # サブ分割対象とする「大パーツ」の閾値(全面積比)
# サブ分割を採用する条件: 各チャンクがパーツ面積のこの比率以上。
# 実写ジョブ(momo.png系、頭部の顔色領域が投影後パーツ面積の約13.4〜13.9%)
# での実測に基づき0.12に調整(0.15だと僅かに届かず頭胴分離が働かない
# ケースを実データ検証で確認したため)。合成テストの色境界(45/55分割)は
# この値でも十分「有意」の余裕を持って上回る。
_IMG_SIGNIFICANT_CHUNK_FRACTION = 0.12

# LLMバウンディングボックス由来のラベルは互いに重なり合う(bbox同士が
# 重複する)ため、色領域ラベルより有意チャンク判定を緩める
# (帽子・耳等の小パーツがサブ分割で棄却されないように)。
_IMG_SIGNIFICANT_CHUNK_FRACTION_LLM = 0.08

_MIN_AREA_FRACTION = 0.02  # 極小パーツとみなす全面積に対する比率の閾値
_DEFAULT_N_SAMPLES = 1500  # SDFサンプリング面数の上限
_RAYS_PER_SAMPLE = 5
_CONE_DEG = 25.0
_RAY_EPS = 1e-9
# 肉厚の太いクラスタ/細いクラスタの重心比がこの値未満なら「実質一様」と
# みなしクラスタリングを行わない(球のような単一パーツ形状で、ノイズだけを
# 理由に無意味な2分割をしてしまうのを防ぐ)。
_THICKNESS_BIMODAL_RATIO_THRESHOLD = 1.5


# --------------------------------------------------------------------------
# 画像誘導分解: 2D色領域抽出
# --------------------------------------------------------------------------
def extract_image_regions(
    image_rgba: np.ndarray, max_regions: int = _IMG_MAX_REGIONS, seed: int = 0
) -> tuple[np.ndarray, int]:
    """背景除去済み画像(RGBA)から色領域ラベル画像を抽出する。

    アルファ>0の画素をk-means(色数4〜6、`scipy.cluster.vq.kmeans2`)で
    量子化し、`scipy.ndimage.label` で連結成分化した後、全画素数の
    `_IMG_MIN_REGION_AREA_FRACTION` 未満の領域を最近傍の大領域へマージし、
    最終的に `max_regions` 以下の領域数へ絞る。

    Args:
        image_rgba: (H, W, 4) uint8 ndarray(アルファチャンネル必須、
            背景=alpha 0 を仮定)。
        max_regions: 領域数の上限。
        seed: k-meansのシード(決定的な結果にするため固定推奨)。

    Returns:
        (label_img, n_regions):
            label_img: (H, W) int64。透明画素は -1。不透明画素は
                0始まりの連番領域ラベル。
            n_regions: 実際の領域数(0の場合は不透明画素が無い)。
    """
    image_rgba = np.asarray(image_rgba)
    if image_rgba.ndim != 3 or image_rgba.shape[2] < 4:
        raise ValueError("image_rgbaは(H, W, 4)のndarrayである必要があります。")

    h, w = image_rgba.shape[:2]
    alpha = image_rgba[:, :, 3]
    opaque_mask = alpha > 0
    label_img = np.full((h, w), -1, dtype=np.int64)

    n_opaque = int(np.sum(opaque_mask))
    if n_opaque == 0:
        return label_img, 0

    rgb = image_rgba[:, :, :3].astype(np.float64)
    opaque_pixels = rgb[opaque_mask]  # (n_opaque, 3)

    n_unique = len(np.unique(opaque_pixels, axis=0))
    # 色数4〜6を狙うが、ユニーク色数がそれ未満ならその数に合わせる。
    k = max(1, min(max(4, min(6, max_regions)), n_unique))

    if k == 1:
        color_labels = np.zeros(n_opaque, dtype=np.int64)
    else:
        try:
            _centroids, color_labels = kmeans2(
                opaque_pixels, k, seed=seed, minit="++", missing="warn"
            )
        except Exception:
            color_labels = np.zeros(n_opaque, dtype=np.int64)

    color_label_img = np.full((h, w), -1, dtype=np.int64)
    color_label_img[opaque_mask] = color_labels

    # 色クラスタごとに連結成分化(同じ色でも離れた領域は別領域として扱う)。
    region_label_img = np.full((h, w), -1, dtype=np.int64)
    next_region = 0
    structure = _ndimage.generate_binary_structure(2, 2)  # 8近傍
    for c in range(int(color_labels.max()) + 1 if n_opaque else 0):
        mask_c = color_label_img == c
        if not np.any(mask_c):
            continue
        comp_labels, n_comp = _ndimage.label(mask_c, structure=structure)
        for comp_id in range(1, n_comp + 1):
            region_label_img[comp_labels == comp_id] = next_region
            next_region += 1

    if next_region == 0:
        # フォールバック: 全不透明画素を単一領域とする
        region_label_img[opaque_mask] = 0
        next_region = 1

    region_label_img = _merge_tiny_image_regions(
        region_label_img, next_region, n_opaque, _IMG_MIN_REGION_AREA_FRACTION
    )
    region_label_img = _limit_image_regions(region_label_img, max_regions)

    return region_label_img, int(region_label_img.max()) + 1 if np.any(region_label_img >= 0) else 0


def _merge_tiny_image_regions(
    label_img: np.ndarray, n_regions: int, n_opaque: int, min_area_fraction: float
) -> np.ndarray:
    """全画素数に対する面積比が閾値未満の領域を、最近傍の大領域へマージする。

    実写画像はアンチエイリアシング・ノイズ由来の1〜数画素の孤立成分を
    大量(数百〜数千)に生みうるため、1領域ずつcKDTreeを再構築して処理すると
    非常に遅い(O(領域数 × 画素数 log 画素数))。ここでは反復のたびに
    「その時点の全ての極小領域」をまとめて1回のcKDTreeクエリで大領域へ
    再割当するバッチ処理にし、実用速度に収める。
    """
    return _batch_merge_by_area_threshold(
        label_img, n_opaque=n_opaque, min_area_fraction=min_area_fraction, max_regions=None
    )


def _limit_image_regions(label_img: np.ndarray, max_regions: int) -> np.ndarray:
    """領域数が `max_regions` を超える場合、面積の小さい領域から順に
    最近傍の大領域へマージして上限内に収める(バッチ処理、`_merge_tiny_image_regions`
    と同じ性能上の理由)。"""
    return _batch_merge_by_area_threshold(
        label_img, n_opaque=None, min_area_fraction=None, max_regions=max_regions
    )


def _batch_merge_by_area_threshold(
    label_img: np.ndarray,
    n_opaque: Optional[int],
    min_area_fraction: Optional[float],
    max_regions: Optional[int],
) -> np.ndarray:
    """極小領域(または超過分の領域)を隣接する大領域へまとめてマージする。

    各反復で「マージ対象となる全領域」を一括判定し、対象外(=残す)領域の
    画素のみでcKDTreeを1回構築して、対象領域の全画素を最近傍探索で
    再割当する(領域ごとにcKDTreeを再構築する素朴な実装は実写画像の
    大量の極小領域(数百〜数千)で非常に遅くなるため避ける)。

    `min_area_fraction`/`n_opaque` を指定すると「全画素数に対する面積比が
    閾値未満」の領域をすべて対象にする。`max_regions` を指定すると
    「領域数がその上限を超えている間、面積の小さい領域から」対象にする
    (収束のため1反復あたり最大でも領域数の半分程度を対象にする)。
    """
    label_img = label_img.copy()
    ys, xs = np.where(label_img >= 0)
    if len(ys) == 0:
        return label_img
    labels_flat = label_img[ys, xs]
    pts = np.column_stack([ys, xs])

    for _ in range(40):  # 収束保証のための安全な反復上限(通常1〜3回で収束)
        unique_labels, counts = np.unique(labels_flat, return_counts=True)
        if len(unique_labels) <= 1:
            break

        if max_regions is not None:
            if len(unique_labels) <= max(1, max_regions):
                break
            n_to_remove = len(unique_labels) - max(1, max_regions)
            order = np.argsort(counts)
            target_labels = set(int(lbl) for lbl in unique_labels[order[:n_to_remove]])
        else:
            areas_fraction = counts / max(n_opaque or len(labels_flat), 1)
            target_labels = set(
                int(lbl) for lbl, frac in zip(unique_labels, areas_fraction) if frac < (min_area_fraction or 0.0)
            )
            if not target_labels:
                break
            # 全領域が対象(=残す大領域が無い)場合は最大の領域だけ残す
            if len(target_labels) >= len(unique_labels):
                keep = int(unique_labels[np.argmax(counts)])
                target_labels.discard(keep)

        if not target_labels:
            break

        target_mask = np.array([lbl in target_labels for lbl in labels_flat])
        keep_mask = ~target_mask
        if not np.any(keep_mask):
            break

        keep_pts = pts[keep_mask]
        keep_labels = labels_flat[keep_mask]
        tree = cKDTree(keep_pts)
        _, nn_idx = tree.query(pts[target_mask], k=1)
        labels_flat[target_mask] = keep_labels[nn_idx]

    label_img[ys, xs] = labels_flat
    return _relabel_image_contiguous(label_img)


def _relabel_image_contiguous(label_img: np.ndarray) -> np.ndarray:
    label_img = label_img.copy()
    mask = label_img >= 0
    if not np.any(mask):
        return label_img
    unique_labels = np.unique(label_img[mask])
    mapping = {int(old): new for new, old in enumerate(unique_labels)}
    flat = label_img[mask]
    label_img[mask] = np.array([mapping[int(v)] for v in flat], dtype=np.int64)
    return label_img


# --------------------------------------------------------------------------
# LLM誘導分解: bbox→2Dラベル画像 (純粋関数、HTTP/PIL非依存)
# --------------------------------------------------------------------------
def labels_from_bboxes(
    bboxes: list[tuple[float, float, float, float]], alpha_mask: np.ndarray
) -> np.ndarray:
    """正規化バウンディングボックス群を2Dラベル画像に変換する。

    マルチモーダルLLMが返すbboxは互いに重なり合う(例: 帽子のbboxが
    頭・胴のbboxに包含される)。このため面積の大きいbboxから順に描画し、
    小さいbboxで後から上書きすることで、小パーツ(帽子・耳等)が大パーツ
    (胴体等)に飲み込まれないようにする。

    Args:
        bboxes: 正規化座標 (x0, y0, x1, y1) (0..1) のリスト。描画順序は
            呼び出し側で気にしなくてよい(本関数内で面積降順に並べ替える)。
        alpha_mask: (H, W) 配列。0より大きい画素のみラベルを割り当てる
            (アルファ0=背景の画素は常にラベル0のまま)。

    Returns:
        (H, W) int64 のラベル画像。`extract_image_regions` と同じ規約
        (透明画素/どのbboxにも含まれない不透明画素は -1、それ以外は
        0始まりの連番ラベル=入力 `bboxes` のインデックス)。
        `project_labels_to_faces` にそのまま渡せる形式。
    """
    alpha_mask = np.asarray(alpha_mask)
    h, w = alpha_mask.shape[:2]
    opaque = alpha_mask > 0

    label_img = np.full((h, w), -1, dtype=np.int64)
    if not np.any(opaque) or len(bboxes) == 0:
        return label_img

    # 面積の大きい順に描画(小さいbboxが後から上書きするように)。
    areas = [
        max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1])) for b in bboxes
    ]
    order = sorted(range(len(bboxes)), key=lambda i: areas[i], reverse=True)

    ys = np.arange(h).reshape(-1, 1)
    xs = np.arange(w).reshape(1, -1)
    u = (xs + 0.5) / w
    v = (ys + 0.5) / h

    for i in order:
        x0, y0, x1, y1 = bboxes[i]
        box_mask = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1) & opaque
        label_img[box_mask] = i

    return label_img


# --------------------------------------------------------------------------
# 画像誘導分解: 3D→2D投影 (colorproc.py と同一規約の自前実装)
# --------------------------------------------------------------------------
def _image_subject_bbox_uv(alpha: np.ndarray) -> tuple[float, float, float, float]:
    """アルファ>0領域のバウンディングボックスを正規化uv座標
    (u_min, u_max, v_min, v_max) (0..1) で返す(colorproc.py `_subject_bbox_uv` と同一規約)。"""
    h, w = alpha.shape[:2]
    ys, xs = np.where(alpha > 0)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        if x1 > x0 and y1 > y0:
            return (x0 / w, (x1 + 1) / w, y0 / h, (y1 + 1) / h)
    return (0.0, 1.0, 0.0, 1.0)


def project_labels_to_faces(mesh: trimesh.Trimesh, label_img: np.ndarray) -> np.ndarray:
    """面重心をメッシュ正面軸(-Y)から画像へ投影し、各面の2D領域ラベルを返す。

    colorproc.py の `project_colors`/`_project_image_colors` と同一の投影規約
    (正面=-Y、画像u→メッシュ+X(`_U_TO_X_SIGN=+1`と同じ符号)、画像v→Z上下
    反転、メッシュXZバウンディングボックスを画像の被写体bbox(アルファ>0)に
    フィット)を自前実装する(pattern モジュールはserver/colorproc.pyを
    importしない境界規約のため、ロジックを複製する)。

    透明画素に落ちた面は `scipy.spatial.cKDTree` で最近傍の不透明画素の
    ラベルに割り当てる。

    Args:
        mesh: 対象メッシュ(Z-up、正面が-Y方向を向く想定)。
        label_img: (H, W) int64。`extract_image_regions` の出力
            (透明画素は-1)。

    Returns:
        (F,) int64。各面の2D領域ラベル(すべて>=0。全画素透明等で
        ラベルが取得できない場合は全面 -1)。
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros((0,), dtype=np.int64)

    h, w = label_img.shape[:2]
    opaque_mask = label_img >= 0
    if not np.any(opaque_mask):
        return np.full(n_faces, -1, dtype=np.int64)

    u_min, u_max, v_min, v_max = _image_subject_bbox_uv(opaque_mask.astype(np.uint8))

    centers = mesh.triangles_center
    bounds = mesh.bounds
    x_min, x_max = bounds[0][0], bounds[1][0]
    z_min, z_max = bounds[0][2], bounds[1][2]
    x_extent = max(x_max - x_min, 1e-9)
    z_extent = max(z_max - z_min, 1e-9)

    x_norm = (centers[:, 0] - x_min) / x_extent  # 0..1, 0=-X側, 1=+X側
    if _IMG_U_TO_X_SIGN < 0:
        u_norm = 1.0 - x_norm
    else:
        u_norm = x_norm
    u = u_min + u_norm * (u_max - u_min)

    z_norm = (centers[:, 2] - z_min) / z_extent  # 0..1, 0=床, 1=頭頂
    v = v_min + (1.0 - z_norm) * (v_max - v_min)  # 上下反転

    px = np.clip((u * w).astype(np.int64), 0, w - 1)
    py = np.clip((v * h).astype(np.int64), 0, h - 1)

    sampled_labels = label_img[py, px]
    missing = sampled_labels < 0
    if np.any(missing) and not np.all(missing):
        opaque_ys, opaque_xs = np.where(opaque_mask)
        tree = cKDTree(np.column_stack([opaque_ys, opaque_xs]))
        missing_idx = np.where(missing)[0]
        _, nn_idx = tree.query(np.column_stack([py[missing_idx], px[missing_idx]]))
        sampled_labels[missing_idx] = label_img[opaque_ys[nn_idx], opaque_xs[nn_idx]]

    return sampled_labels.astype(np.int64)


# --------------------------------------------------------------------------
# ブルートフォース Möller-Trumbore (単一レイ vs 複数三角形)
# --------------------------------------------------------------------------
def _ray_hit_distances(
    origin: np.ndarray,
    direction: np.ndarray,
    tri_v0: np.ndarray,
    tri_v1: np.ndarray,
    tri_v2: np.ndarray,
) -> np.ndarray:
    """単一レイと三角形集合の交差判定。三角形ごとのヒット距離を返す
    (未ヒットはinf)。`_ray_vs_triangles` の全ヒット版(背面仮想シードの
    first exit 判定でヒット面のインデックスが必要なため)。"""
    n = len(tri_v0)
    result = np.full(n, np.inf)
    if n == 0:
        return result
    e1 = tri_v1 - tri_v0
    e2 = tri_v2 - tri_v0
    pvec = np.cross(direction, e2)
    det = np.sum(e1 * pvec, axis=1)
    mask = np.abs(det) > _RAY_EPS
    if not np.any(mask):
        return result
    inv_det = np.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]
    tvec = origin - tri_v0
    u = np.sum(tvec * pvec, axis=1) * inv_det
    qvec = np.cross(tvec, e1)
    v = np.sum(direction * qvec, axis=1) * inv_det
    t = np.sum(e2 * qvec, axis=1) * inv_det
    valid = (
        mask
        & (u >= -1e-6)
        & (u <= 1 + 1e-6)
        & (v >= -1e-6)
        & (u + v <= 1 + 1e-6)
        & (t > 1e-6)
    )
    result[valid] = t[valid]
    return result


def _ray_vs_triangles(
    origin: np.ndarray,
    direction: np.ndarray,
    tri_v0: np.ndarray,
    tri_v1: np.ndarray,
    tri_v2: np.ndarray,
) -> float:
    """単一レイと三角形集合の交差判定。最も近いヒットの距離(なければinf)。"""
    t = _ray_hit_distances(origin, direction, tri_v0, tri_v1, tri_v2)
    return float(np.min(t)) if len(t) else np.inf


# --------------------------------------------------------------------------
# 局所肉厚 (SDF風)
# --------------------------------------------------------------------------
def _cone_directions(normal: np.ndarray, n_rays: int, cone_deg: float, rng: np.random.Generator) -> list[np.ndarray]:
    dirs = [normal]
    for _ in range(max(0, n_rays - 1)):
        rand_vec = rng.normal(size=3)
        rand_vec = rand_vec - np.dot(rand_vec, normal) * normal
        norm_len = np.linalg.norm(rand_vec)
        if norm_len < 1e-9:
            continue
        rand_vec = rand_vec / norm_len
        angle = np.radians(cone_deg) * rng.uniform(0.0, 1.0)
        d = np.cos(angle) * normal + np.sin(angle) * rand_vec
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-9:
            continue
        dirs.append(d / d_norm)
    return dirs


def _sample_local_thickness(
    mesh: trimesh.Trimesh,
    n_samples: int,
    rays_per_sample: int,
    cone_deg: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """一部の面についてのみ局所肉厚(内向きレイキャストの中央値)を計算する。

    Returns:
        (thickness: (F,) ndarray, NaNは未計算面, sample_idx: 計算した面indices)
    """
    n_faces = len(mesh.faces)
    rng = np.random.default_rng(seed)
    n_samples = min(max(1, n_samples), n_faces)
    sample_idx = rng.choice(n_faces, size=n_samples, replace=False)

    centers = mesh.triangles_center
    normals = mesh.face_normals
    tri = mesh.triangles
    tri_v0, tri_v1, tri_v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]

    tree = cKDTree(centers)
    cell = max(float(mesh.scale) / 20.0, 1e-6)

    thickness = np.full(n_faces, np.nan)

    for fi in sample_idx:
        origin_face = centers[fi]
        inward_normal = -normals[fi]
        if not np.all(np.isfinite(inward_normal)) or np.linalg.norm(inward_normal) < 1e-9:
            continue
        dirs = _cone_directions(inward_normal, rays_per_sample, cone_deg, rng)

        # 近傍三角形のみを候補にすることで全面総当たりを回避する(rtree等
        # 追加依存なしの高速化)。ヒットが得られなければ半径を拡大し、
        # 最終手段として全面総当たりにフォールバックする。
        radius = cell * 3.0
        idx = tree.query_ball_point(origin_face, radius)
        attempts = 0
        while len(idx) < 8 and attempts < 6:
            radius *= 1.8
            idx = tree.query_ball_point(origin_face, radius)
            attempts += 1
        idx_arr = np.asarray(idx, dtype=np.int64)
        idx_arr = idx_arr[idx_arr != fi]

        hits = []
        for d in dirs:
            ray_origin = origin_face + inward_normal * (cell * 1e-3)
            t = np.inf
            if len(idx_arr) > 0:
                t = _ray_vs_triangles(
                    ray_origin, d, tri_v0[idx_arr], tri_v1[idx_arr], tri_v2[idx_arr]
                )
            if not np.isfinite(t):
                # 近傍候補で見つからない場合のみ全面総当たり(稀)
                t = _ray_vs_triangles(ray_origin, d, tri_v0, tri_v1, tri_v2)
            if np.isfinite(t):
                hits.append(t)

        if hits:
            thickness[fi] = float(np.median(hits))

    return thickness, sample_idx


def _dual_graph(mesh: trimesh.Trimesh) -> csr_matrix:
    n_faces = len(mesh.faces)
    adjacency = mesh.face_adjacency
    if len(adjacency) == 0:
        return csr_matrix((n_faces, n_faces))
    centers = mesh.triangles_center
    a, b = adjacency[:, 0], adjacency[:, 1]
    dist = np.linalg.norm(centers[a] - centers[b], axis=1)
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    data = np.concatenate([dist, dist])
    return csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))


# --------------------------------------------------------------------------
# 手動シード誘導分解 (guidance="manual"、ユーザーが3Dビューア上でクリックした
# 数点から測地距離ベースでパーツを分割する。Graph Cutの代替、追加依存なし)
# --------------------------------------------------------------------------
# 凹エッジ(くびれ)を跨ぐコストの倍率。`_split_subregion`/`_split_subregion_from_seeds`
# と同じ重み設計(base_dist * (1 + w * concave_strength))を踏襲するが、こちらは
# ユーザーが明示的に指定したシード同士の境界を求める用途のため、より強く
# くびれへ吸い付かせたい。3〜10 の範囲で試した結果、5では緩く境界が直線的に
# なるケース(首・肩のような浅いくびれ)があり、10だとくびれ以外の場所でも
# わずかな凹みに過敏に反応して境界がギザついたため、中間の 8.0 を既定値とする。
_MANUAL_SEED_CONCAVITY_WEIGHT = 8.0
# 肉厚遷移(|Δlog(肉厚)|)を跨ぐコストの倍率。Hunyuan3D生成メッシュの背面
# (画像補完由来で滑らか)や耳の付け根のように凹二面角の信号が弱い箇所でも、
# 肉厚コントラスト(耳=薄い/頭=厚い)は残るため、肉厚の急変帯で波面が
# 止まるようにする。対数差なのでモデルスケールに不変。
# 値の根拠(tests/test_pattern_parts.py の耳モック=大球r28+付け根r2.5+小球r6
# をvoxelize→marching_cubesした形状での実測。この工程で付け根の凹二面角は
# ほぼ平滑化され、凹み信号が弱い実メッシュの失敗モードを再現する。なお
# 耳モックの付け根は凹み信号≈0のため、加算形と乗算形の重みは一致し、
# 以下の実測値は加算形でもそのまま有効):
#   理想境界=付け根 z 50〜58 に対し、境界リング位置(z平均)は
#   w=0: 35.8(頭の途中で止まる=トライアルで報告された症状) / w=20: 45.9 /
#   w=40: 51.0(付け根帯に到達) / w=60: 53.2 / w=80: 54.2(飽和)。
#   一方で一様肉厚の球の2分割では w=40 でも境界スプレッド増は約1mmに留まり
#   (0.8→1.8mm)、肉厚推定ノイズによるギザつきは実質ない。ダンベルの
#   非対称シードも w=40 で境界 z≈34(w=0の36.6より理想30に近い)と回帰なし。
#   凹みと肉厚遷移が共存する胴+腕モック(Tポーズ、付け根に溝)では、
#   w=40 の加算形で胴の腕への誤食い込み1.0%(w=0では20%)。境界が付け根の
#   肉厚遷移帯(付け根のわずかに腕側)へ寄るぶん腕側の取りこぼしが約11%
#   (数mm幅の帯)生じるが、型紙用途では付け根直近の縫い目として許容範囲。
#   以上から、付け根到達と安定性のバランスで 40.0 を既定値とする。
_MANUAL_SEED_THICKNESS_WEIGHT = 40.0
_MANUAL_SEED_SMOOTH_ITERATIONS = 3


def _dual_graph_weighted(
    mesh: trimesh.Trimesh,
    concavity_weight: float,
    thickness: Optional[np.ndarray] = None,
    thickness_weight: float = 0.0,
) -> csr_matrix:
    """凹み(くびれ)・肉厚遷移を跨ぐエッジのコストを上げた重み付き面双対グラフ。

    エッジ重み = 面重心間距離
                × (1 + concavity_weight × 凹み度 + thickness_weight × |Δlog(肉厚)|)。
    凹み度は隣接面ペアの二面角から求め、凸(または平坦)なエッジは0、凹エッジ
    ほど1に近づく(`_split_subregion` と同じ考え方)。肉厚項は隣接面ペアの
    局所肉厚の対数差(スケール不変)で、細い部位(耳・腕)と太い部位(頭・胴)の
    境目の肉厚急変帯を高コストにする。マルチソース最短路の波面がこれら重みの
    高い(=通りにくい)エッジを避けて進むため、境界が自然にくびれ・部位の
    付け根へ吸い付く。

    **加算形にしている理由(重要)**: 以前は乗算形
    `× (1+w_c×凹み) × (1+w_t×|Δlog t|)` だったが、展開すると交差項
    `w_c×w_t×凹み×|Δlog t|` が現れる。脇の下・腕の付け根のように
    「強い凹みと肉厚急変が同時に存在する」場所ではこの交差項によりエッジ
    コストが過大(共存エッジで加算形の約3倍)になり、波面が本来の境界
    (脇の溝)を避けて信号の弱い経路(肩の上など)で出会ってしまい、
    胴体と腕の分離が乗算形導入前(凹みのみ)より退行したと実機トライアルで
    報告された。加算形は交差項を持たないため共存部の過大ペナルティが除去され、
    かつ信号が単独の箇所(くびれのみ/肉厚遷移のみ)では乗算形と完全に一致する
    (もう一方の項が0のため耳モック・ダンベルの挙動は不変)。max形
    `× (1+max(w_c×凹み, w_t×|Δlog t|))` も胴+腕合成メッシュで比較したが、
    精度は加算形と同等で、弱い共存信号を切り捨てるぶん腕の取りこぼしが
    やや悪化した(w_t=40で13.7%対10.8%)ため採用しなかった。

    Args:
        mesh: 対象メッシュ。
        concavity_weight: 凹みペナルティの倍率。
        thickness: (F,) 面ごとの局所肉厚(`compute_local_thickness` の出力)。
            None の場合は肉厚項を適用しない。
        thickness_weight: 肉厚遷移ペナルティの倍率(0で無効)。
    """
    n_faces = len(mesh.faces)
    adjacency = mesh.face_adjacency
    if len(adjacency) == 0:
        return csr_matrix((n_faces, n_faces))
    centers = mesh.triangles_center
    a, b = adjacency[:, 0], adjacency[:, 1]
    base_dist = np.linalg.norm(centers[a] - centers[b], axis=1)
    scale = float(np.median(base_dist[base_dist > 0])) if np.any(base_dist > 0) else 1.0
    base_dist = np.maximum(base_dist, scale * 1e-4)

    try:
        angles = mesh.face_adjacency_angles
        convex = mesh.face_adjacency_convex
        concave_strength = np.where(convex, 0.0, angles / np.pi)
    except Exception:
        concave_strength = np.zeros(len(adjacency))

    penalty = concavity_weight * concave_strength

    if thickness is not None and thickness_weight > 0.0:
        log_t = np.log(np.clip(np.asarray(thickness, dtype=np.float64), 1e-6, None))
        dlog = np.abs(log_t[a] - log_t[b])
        penalty = penalty + thickness_weight * dlog

    weights = base_dist * (1.0 + penalty)

    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    data = np.concatenate([weights, weights])
    return csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))


def _point_to_triangle_distance(p: np.ndarray, tri_v0: np.ndarray, tri_v1: np.ndarray, tri_v2: np.ndarray) -> np.ndarray:
    """点1つと三角形集合の最短距離(ブルートフォース、closest point on triangle)。

    `trimesh.proximity.ProximityQuery` はrtree(未導入・追加依存不可)を要求
    するため使えない。ここでは標準的な「点と三角形の最近接点」の解析解
    (Ericson, Real-Time Collision Detection のアルゴリズム相当)を
    numpyでベクトル化して自前実装する。
    """
    ab = tri_v1 - tri_v0
    ac = tri_v2 - tri_v0
    ap = p - tri_v0

    d1 = np.sum(ab * ap, axis=1)
    d2 = np.sum(ac * ap, axis=1)
    closest = tri_v0.copy()
    # 頂点tri_v0領域
    mask_v0 = (d1 <= 0) & (d2 <= 0)

    bp = p - tri_v1
    d3 = np.sum(ab * bp, axis=1)
    d4 = np.sum(ac * bp, axis=1)
    mask_v1 = (d3 >= 0) & (d4 <= d3) & ~mask_v0

    cp = p - tri_v2
    d5 = np.sum(ab * cp, axis=1)
    d6 = np.sum(ac * cp, axis=1)
    mask_v2 = (d6 >= 0) & (d5 <= d6) & ~mask_v0 & ~mask_v1

    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~mask_v0 & ~mask_v1 & ~mask_v2
    v_ab = np.where(np.abs(d1 - d3) > 1e-12, d1 / np.maximum(d1 - d3, 1e-12), 0.0)

    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~mask_v0 & ~mask_v1 & ~mask_v2 & ~mask_ab
    v_ac = np.where(np.abs(d2 - d6) > 1e-12, d2 / np.maximum(d2 - d6, 1e-12), 0.0)

    va = d3 * d6 - d5 * d4
    denom_bc = (d4 - d3) + (d5 - d6)
    mask_bc = (
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        & ~mask_v0 & ~mask_v1 & ~mask_v2 & ~mask_ab & ~mask_ac
    )
    v_bc = np.where(np.abs(denom_bc) > 1e-12, (d4 - d3) / np.maximum(denom_bc, 1e-12), 0.0)

    mask_face = ~(mask_v0 | mask_v1 | mask_v2 | mask_ab | mask_ac | mask_bc)
    denom_face = vc + vb + va
    v_face = np.where(np.abs(denom_face) > 1e-12, vb / np.maximum(denom_face, 1e-12), 1.0 / 3.0)
    w_face = np.where(np.abs(denom_face) > 1e-12, va / np.maximum(denom_face, 1e-12), 1.0 / 3.0)

    closest = tri_v0.copy()
    closest[mask_v1] = tri_v1[mask_v1]
    closest[mask_v2] = tri_v2[mask_v2]
    closest[mask_ab] = tri_v0[mask_ab] + ab[mask_ab] * v_ab[mask_ab, None]
    closest[mask_ac] = tri_v0[mask_ac] + ac[mask_ac] * v_ac[mask_ac, None]
    closest[mask_bc] = tri_v1[mask_bc] + (tri_v2[mask_bc] - tri_v1[mask_bc]) * v_bc[mask_bc, None]
    closest[mask_face] = (
        tri_v0[mask_face] + ab[mask_face] * v_face[mask_face, None] + ac[mask_face] * w_face[mask_face, None]
    )

    return np.linalg.norm(closest - p[None, :], axis=1)


def _snap_points_to_faces(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """各点に最も近い面(三角形)のインデックスを返す(rtree非依存)。

    面重心への`cKDTree`最近傍探索で候補面を絞り込み(半径を段階的に拡大)、
    候補内で正確な点—三角形距離(`_point_to_triangle_distance`)を計算して
    最小のものを採用する(`_sample_local_thickness` と同様の
    「cKDTreeで絞り込み→ブルートフォースで厳密化」というこのモジュール既存の
    パターンを踏襲し、rtree等の追加依存を避ける)。
    """
    centers = mesh.triangles_center
    tree = cKDTree(centers)
    tri = mesh.triangles
    tri_v0, tri_v1, tri_v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    scale = max(float(mesh.scale), 1e-6)

    result = np.zeros(len(points), dtype=np.int64)
    for i, p in enumerate(points):
        k = min(16, len(centers))
        radius = scale * 0.05
        idx = np.array([], dtype=np.int64)
        attempts = 0
        while len(idx) < k and attempts < 8:
            idx = np.asarray(tree.query_ball_point(p, radius), dtype=np.int64)
            radius *= 2.0
            attempts += 1
        if len(idx) == 0:
            # 最終フォールバック: 全面総当たり
            idx = np.arange(len(centers))
        dists = _point_to_triangle_distance(p, tri_v0[idx], tri_v1[idx], tri_v2[idx])
        result[i] = idx[int(np.argmin(dists))]

    return result


# 背面仮想シードのトンネル検出閾値: 採用exitまでの距離が、シード面の局所
# 肉厚(`compute_local_thickness` の値、拡散補間・平滑化済みで安定)の
# この倍数を超える候補は「レイが同一部位の反対側スキンではなく、別部位まで
# 貫通している」とみなして距離プールから除外する(例: 肩の上のシードから
# 真下へのレイが胴を縦断して足元に抜けるケース)。プールが空になった場合は
# トンネルとしてスキップする。
#
# **既知の限界(トンネル検出・所有権チェックの両方をすり抜けるケース)**:
# `compute_local_thickness` は双対グラフ上で拡散補間・平滑化(既定5反復の
# 近傍平均)を行うため、小さいパーツ(帽子)が厚い部位(頭)に隣接している
# と、肉厚値が部位境界を越えて滲み出し、小パーツ側のシード面で肉厚が
# 実寸(帽子の直径≈18mm)より大きく過大評価されることがある(回帰テスト用
# モック実測: 帽子シード面の推定肉厚19.3mm)。この場合、トンネル検出の
# 閾値(肉厚×3倍=実測57.9mm)が、頭を貫通して首元まで抜ける誤ったexit
# (実測距離25〜31mm)まで許容してしまう。
#
# この誤exitを追加ガードで捕捉できないか、以下を検証したがいずれも既存の
# 正当ケースを壊すため不採用とした(詳細な実測比較は完了報告参照):
#   - 距離プールを「最小距離からの比率」でさらに絞り込む: 十字モック
#     (2つの円柱が直交する交差部)のように「正しい対蹠面が遠く(円柱の
#     反対端)、交差部内壁への誤った反射が近い」形状では前提が逆転する。
#   - 所有権チェックの閾値を厳格化: 誤exitの所有権比率(実測1.04)が、
#     十字モックの正当候補(実測1.235)や頭部の正当な対蹠面(実測1.15、
#     実ジョブでは1.27)と値域が重なり、線形分離できない。
#   - 対蹠アライメントスコアと所有権比率の複合スコアリング: 上記の値域
#     重複のため、帽子ケースを分離できる係数では同様に十字モック等を
#     壊す。
# 現状はこの特定の失敗モードを未解決の既知の限界として受け入れ、
# `labels_from_seeds` 側のペア単位均衡・多数決平滑化による部分的な緩和に
# 委ねる(`tests/test_pattern_parts.py`
# `test_propagate_hat_virtual_seed_tunnel_known_limitation` で現状の
# 挙動を固定する回帰テストとして記録)。
_VIRTUAL_SEED_MAX_THICKNESS_RATIO = 3.0
# 対蹠アライメント探索(_opposite_virtual_seed_faces)のコーン設定。
# 近傍平均逆法線を中心に広角(60°)で多めのレイ(20本)を飛ばし、うち
# 「局所肉厚の一定倍以内の距離プール内で対蹠アライメントスコア(レイ方向と
# exit面法線の内積)が最大」のレイを採用する。頭頂寄りのシードのように
# 部位内で偏った位置では、単一方向(逆法線1本)の幾何学的な直近対蹠点が
# 隣接パーツ側(首側)に落ちてしまうため、コーン内で「最も素直に(=対蹠に)
# 抜ける」レイを探索することで対応する(実ジョブでの検証は完了報告参照)。
_VIRTUAL_SEED_ALIGNMENT_RAYS = 20
_VIRTUAL_SEED_ALIGNMENT_CONE_DEG = 60.0
# 自己近傍除外の絶対下限: 「面積の中央値の平方根×この倍率」未満のexit距離は
# 縮退候補(シード面のすぐ隣、法線がたまたま逆向きに揃う)として除外する。
_VIRTUAL_SEED_MIN_EXIT_RATIO = 3.0
# 所有権チェックの許容比: exit面の予備Dijkstra割当(ユーザーシードのみ)で、
# 自グループまでの距離が「他グループまでの最小距離×この係数」以下であること。
# exit面が他パーツの領域の奥深く(自グループから測地的に遠い)にある場合=
# レイが首などを貫通して隣のパーツのスキンに抜けたケースを棄却する。
# 当初はレイの往復チェック(exit面からの逆レイが元シード付近に戻るか)を
# 使っていたが、実メッシュの背面(画像補完由来)は法線が凸凹で正当なケース
# でも戻りレイが大きく逸れる(実ジョブ実測: 正当な頭で誤差比0.49、細い腕で
# 1.4〜4.3)ため、測地距離ベースに置き換えた。
# 閾値の根拠: 頭+胴モックの誤配置(胴シードのレイが首を貫通して頭背面へ)は
# ratio=1.50。対蹠アライメント探索導入後の実ジョブ(犬9パーツ)実測では、
# 正当な頭部の対蹠面がratio=1.27(帽子パーツと背面上部を分け合う配置のため
# 1.0をやや超える)となるケースを確認したため、1.3では正当ケースを誤って
# 弾いてしまう。1.3→1.6に緩和(誤配置ケースの1.50はなお防御する層として
# tunnelガードが先に働くため、1.6への緩和でも実害はない)。
_VIRTUAL_SEED_OWNERSHIP_RATIO = 1.6
# ペア単位均衡のコンパクト判定: 予備割当(ユーザーシードのみ)でそのグループに
# 割り当てられた面積が総面積のこの比率以下なら「コンパクト」= 耳・しっぽの
# ような小パーツで、隣接グループだけが背面仮想シードを得ても背面を奪われる
# リスクが低いとみなす(4グループモック実測: 耳3.5〜4.8% / 頭・胴45%超)。
_VIRTUAL_BALANCE_COMPACT_AREA_FRACTION = 0.06


def _opposite_virtual_seed_faces(
    mesh: trimesh.Trimesh,
    seed_faces: np.ndarray,
    seed_groups: np.ndarray,
    thickness: Optional[np.ndarray] = None,
    skip_reasons_out: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """各ユーザーシード面の「反対側スキン」に同グループの仮想シード面を生成する。

    頭や胴体は丸/円柱系の形状であることが多く、正面のスキンから面の逆法線
    方向(メッシュ内部へ)にレイを飛ばして最初に外へ抜ける交点(first exit)
    は、高確率で同じパーツの背面スキンである。そこに同グループの仮想シードを
    置くことで、正面にしかシードを打っていない場合でも背面の割当が改善する
    (ユーザー提案の「正面から見た後ろのスキンを検出して繋げる」の実装)。

    first exit 判定: レイと全三角形の交差(`_ray_hit_distances`)のうち、
    「面法線がレイ方向と同じ向き(=内→外に抜ける)」のヒットの最近傍を採る。
    ヒットなし(開メッシュ等)はスキップする(例外は出さない)。

    誤配置ガード(スキップ条件。理由は `skip_reasons_out` に記録):
        1. `no_hit`: first exit なし(開メッシュ・縮退法線等)。
        2. `tunnel`: first exit までの距離が、シード面の局所肉厚
           (±25°コーンの数本レイの中央値)の `_VIRTUAL_SEED_MAX_THICKNESS_RATIO`
           倍を超える(丸い部位の反対側なら距離≈局所肉厚のはずで、
           大きく超えるのは別部位まで貫通しているケース。肩→足元など)。
        3. `taken`: exit 面が既にユーザーシード面・先行の仮想シード面
           (グループを問わず安全側でスキップ。同グループなら単に冗長)。

    exit面の所有権チェック(`ownership`、予備Dijkstra割当で自グループから
    測地的に遠すぎる候補の棄却。ダンベルの極から極へ貫通したexitが相手
    シードの真上に落ちるケースもこれで除外される)と、グループ間の均衡判定
    (`balance`、どのグループの候補を採用するか)は本関数では行わず、
    呼び出し側(`labels_from_seeds`)が予備Dijkstra割当に基づいて行う。

    **既知の限界(トンネル検出・所有権チェックの両方をすり抜けるケース)**:
    帽子のような小パーツが厚い部位(頭)に隣接する場合、`compute_local_
    thickness` の拡散補間・平滑化により小パーツ側の局所肉厚が過大評価され
    (回帰テスト用モック実測: 帽子シード面19.3mm、実際の直径は18mm相当)、
    トンネル検出の閾値が緩くなって頭を貫通した誤exitが距離プールに残る
    ことがある。この誤exitの所有権比率(実測1.04)は、十字モックのような
    「正しい対蹠面が測地的に遠い」正当ケース(実測1.235)や頭部の正当な
    対蹠面(実測1.15、実ジョブでは1.27)と値域が重なるため、所有権チェックの
    閾値調整・対蹠アライメントスコアとの複合スコアリング・距離プールの
    クラスタリング等、複数の方式を検証したが、いずれも本ケースの分離に
    必要な閾値では十字モック等の既存正当ケースを退行させてしまい、単独の
    追加ガードでは解決できなかった(検討過程は完了報告参照)。現状は
    `labels_from_seeds` 側の**ペア単位均衡判定**(`balance`)が緩和策として
    働く(帽子が誤って背面上部を独占しても、頭の正面シードからの測地距離が
    優先されない領域は隣接グループとして再評価されうる)。

    Args:
        thickness: (F,) 面ごとの局所肉厚(`compute_local_thickness` の出力、
            拡散補間・平滑化済み)。指定された場合、トンネル検出の基準
            (ガード2)にこれを使う。None の場合はコーンレイのexit距離の
            中央値で代用する(トンネル検出の基準として不安定になりうる。
            丸い部位の対蹠面までの距離と、たまたま近くにヒットする低品質
            候補の距離が大きく異なる実測ケースを確認したため、可能な限り
            `thickness` を渡すことを推奨)。
        skip_reasons_out: 指定された場合、スキップしたシードごとに
            `{"seed_index": int, "group": int, "reason": str}` を追記する。

    Returns:
        (virtual_faces, virtual_groups): 仮想シード候補の面インデックスと
        グループIDの配列(生成なしなら両方とも長さ0)。
    """
    centers = mesh.triangles_center
    normals = mesh.face_normals
    tri = mesh.triangles
    tri_v0, tri_v1, tri_v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    eps = max(float(mesh.scale) * 1e-5, 1e-9)
    rng = np.random.default_rng(0)

    # `taken` 用: 使用済みの面(ユーザーシード面+生成済み仮想シード面)
    taken: set[int] = set(int(f) for f in seed_faces)

    virtual_faces: list[int] = []
    virtual_groups: list[int] = []

    def _skip(i: int, g: int, reason: str) -> None:
        if skip_reasons_out is not None:
            skip_reasons_out.append({"seed_index": int(i), "group": int(g), "reason": reason})

    # 近傍平均法線用のKDTree(シード面の単一法線は実メッシュのローカルな
    # バンプで傾いていることがあり、逆法線レイが隣のパーツまで貫通する原因に
    # なる=実ジョブ実測: 犬の頭シードが鼻先の上向き面にスナップし、レイが
    # 胴を縦断して脚の背面へ。近傍の面積重み平均法線でならすと正しく背面を
    # 向く。半径はメッシュスケールの6%: 実ジョブ実測で3%(4mm)では鼻の
    # 上向きバンプを平均しきれず、6%(8mm)で頭の背面に正しく抜けた)。
    center_tree = cKDTree(centers)
    face_areas_v = mesh.area_faces
    avg_radius = max(float(mesh.scale) * 0.06, 1e-6)

    for i, (sf, sg) in enumerate(zip(seed_faces, seed_groups)):
        sf, sg = int(sf), int(sg)
        nb = np.asarray(center_tree.query_ball_point(centers[sf], avg_radius), dtype=np.int64)
        if len(nb) == 0:
            nb = np.array([sf], dtype=np.int64)
        n_avg = (normals[nb] * face_areas_v[nb][:, None]).sum(axis=0)
        norm_len = np.linalg.norm(n_avg)
        if not np.all(np.isfinite(n_avg)) or norm_len < 1e-9:
            # 平均が縮退(近傍法線が打ち消し合う等)なら単一法線にフォールバック
            n_avg = normals[sf]
            norm_len = np.linalg.norm(n_avg)
            if not np.all(np.isfinite(n_avg)) or norm_len < 1e-9:
                _skip(i, sg, "no_hit")
                continue
        base_dir = -n_avg / norm_len

        # 近傍平均逆法線を中心にした広めのコーンで複数レイを飛ばし、その中で
        # 「距離が局所肉厚(`compute_local_thickness`、拡散補間・平滑化済みで
        # 安定)の一定倍以内の候補群のうち、対蹠アライメントスコア(レイ方向と
        # exit面法線の内積)が最大」のレイを採用する(案3。単純な逆法線1本
        # だと、シードが部位の縁寄り(頭頂近くなど)にある場合、幾何学的な
        # 直近対蹠点が隣接パーツ側(首側)に落ちてしまう=実ジョブ実測: 犬の
        # 頭部シード(頭頂寄り)の反対側が首元z=51.2に着地し、本来背面上部を
        # 担うはずの領域を隣接する帽子パーツの仮想シードに奪われた)。
        # 真に「同じ塊の反対側の面」なら両方とも外向き法線同士がほぼ平行に
        # なる(d≈exit法線、score≈1)。
        #
        # スコアのみで選ぶと、法線がたまたま揃う遠方の誤った候補(実測:
        # 反対側の帽子シードで、正面近くまで戻る鋭角レイがscore=1.00の
        # 高スコアで別パーツの面に当たった)を拾ってしまうため、まず局所
        # 肉厚を基準とした距離プールで絞り込む。距離の絶対下限
        # (`_VIRTUAL_SEED_MIN_EXIT_RATIO`)は、シード直近の自己近傍面
        # (法線がたまたま逆向きに揃う縮退候補、実測でt<1mmのケースを確認)を
        # 除外するために必要。
        origin = centers[sf] + base_dir * eps
        ray_dirs = _cone_directions(base_dir, _VIRTUAL_SEED_ALIGNMENT_RAYS, _VIRTUAL_SEED_ALIGNMENT_CONE_DEG, rng)
        abs_min_t = np.sqrt(float(np.median(face_areas_v))) * _VIRTUAL_SEED_MIN_EXIT_RATIO
        candidates: list[tuple[int, float, float]] = []  # (面, 距離, スコア)
        for d in ray_dirs:
            t_ray = _ray_hit_distances(origin, d, tri_v0, tri_v1, tri_v2)
            exiting = (normals @ d) > 1e-9
            t_ray = np.where(exiting, t_ray, np.inf)
            t_ray[sf] = np.inf
            fj = int(np.argmin(t_ray))
            t_j = float(t_ray[fj])
            if not np.isfinite(t_j) or t_j < abs_min_t:
                continue  # 自己近傍等の縮退候補を除外
            score = float(np.dot(d, normals[fj]))
            candidates.append((fj, t_j, score))
        if not candidates:
            _skip(i, sg, "no_hit")  # ヒットなし(開メッシュ・縮退法線等)
            continue

        # ガード1: トンネル検出を兼ねた距離プール。局所肉厚の一定倍以内の
        # 候補だけをプールし、その中でスコア最大のレイを採用する。プールが
        # 空(=局所肉厚の範囲内に対蹠面が無い)場合はトンネルとしてスキップ
        # する。丸い部位の反対側なら距離≈局所肉厚のはずで、腕先端キャップの
        # ように部位の軸方向を向いたシードのレイは隣のパーツまで貫通するため
        # (プールが空になり)ここで棄却される。
        if thickness is not None:
            local_thickness = max(float(thickness[sf]), abs_min_t)
        else:
            # thickness未指定時のフォールバック: コーン内候補の中央値
            # (`labels_from_seeds` からは常にthicknessが渡されるため、この
            # 分岐は本関数を単体で使う場合のみ通る)。
            local_thickness = max(float(np.median([t for _f, t, _s in candidates])), abs_min_t)
        pool = [c for c in candidates if c[1] <= _VIRTUAL_SEED_MAX_THICKNESS_RATIO * local_thickness]
        if not pool:
            _skip(i, sg, "tunnel")
            continue
        fi, t_exit, _score = max(pool, key=lambda c: c[2])

        # ガード2: 既にシード面として使用済み
        if fi in taken:
            _skip(i, sg, "taken")
            continue

        taken.add(fi)
        virtual_faces.append(fi)
        virtual_groups.append(sg)

    return (
        np.array(virtual_faces, dtype=np.int64),
        np.array(virtual_groups, dtype=np.int64),
    )


# --------------------------------------------------------------------------
# 境界の平面フィット正則化 (planar boundary regularization)
# --------------------------------------------------------------------------
# ペアの境界エッジ数がこれ未満なら平面フィットしない(統計的に不安定)。
_PLANAR_MIN_BOUNDARY_EDGES = 10
# 曖昧帯の判定: Dijkstraの1位/2位グループ距離のマージン(d2-d1)が
# 「パーツスケール(小さい方のパーツ面積の平方根)×この係数」未満の面のみ
# 平面による再割当の対象にする。確信度の高い面(正面の深い溝で明確に
# 分かれた面)は変更しない。値の根拠は頭+胴モック・胴+腕モックでの実測
# (コミット時の完了報告参照)。
_PLANAR_MARGIN_FRACTION = 0.35
# アンカーベースのトリム反復フィットのパラメータ:
# 各反復で「信頼度調整残差 |残差|/trust」の小さい順にこの割合を残して再フィット
# する(trustの高い深い溝の点はアンカーとして残りやすく、染み出してうねった
# 低trust区間が外れ値として排除される)。
_PLANAR_TRIM_ITERATIONS = 4
_PLANAR_INLIER_FRACTION = 0.7
# アンカー初期集合: trustが中央値のこの倍数以上の境界点のみでフィットを
# 開始する(「凹み強度の高い境界点を優先」の実装)。境界のtrustが一様
# (コントラストなし)の場合は該当点が少なくなり全点初期化にフォールバック
# する(一様trust境界で上位N%を強制選別するとノイズ選別になり、任意の弧に
# 傾いた平面がフィットされて誤切断する=ダンベルモックで実測)。
_PLANAR_STRONG_TRUST_RATIO = 3.0
# カバレッジガード: アンカー集合の面内広がり(√λ_mid)が全境界点の広がりの
# この比率未満の場合はスキップする(サドル境界のうち局所的に平面な一部の
# 弧だけがアンカーに残って固有値比ガードを通り抜けるのを防ぐ)。
_PLANAR_MIN_COVERAGE = 0.5
# 再割当の範囲: True=ペアの全面を平面側で切り直すフルカット(ユーザー原案
# 「区切りをそのまま伸ばす」)/ False=曖昧帯(マージン小)限定。
# 染み出し再現モック(腕シードを手首に置き胴が肩へ食い込む配置)での比較で、
# 曖昧帯限定は「確信して間違っている染み出し領域」(マージンが大きく曖昧帯に
# 入らない)を矯正できなかったためフルカットを採用(実測はテスト参照)。
_PLANAR_FULL_CUT = True
# フィット品質ガード: **アンカー(インライア)集合**の重み付き共分散の
# 固有値比 λ_min/λ_mid がこれを超える場合、境界が本質的に非平面(鞍状など)
# としてスキップする。リング状の平面境界は面内の広がり(λ_mid)に対して
# 法線方向の広がり(λ_min)がごく小さいのに対し、サドル境界は3次元的に
# 広がるため比が大きい。スケール不変かつ「溝の中での数mmの自然なうねり」を
# 過剰に棄却しない(RMSの絶対値/パーツスケール比では『正当だがうねる境界
# (比0.067〜0.08)』とサドル(0.09〜0.10)の分離が際どかった)。
# 実測: 頭+胴 0.032 / 胴+腕(全周溝) 0.023 / 染み出し(部分グルーブ、
# 矯正したいケース) 0.246 / 十字サドル(棄却したいケース) 0.583 → 0.4 で分離。
_PLANAR_MAX_EIGEN_RATIO = 0.4
# 再割当規模ガード: 再割当の結果、どちらかのパーツの面数がこの比率未満に
# 縮む場合は異常(平面の向きの誤判定等)としてスキップする。
_PLANAR_MIN_KEEP_FRACTION = 0.5
# 平面フィットの重みの下限(凹み強度が全てゼロの境界でも動くように)。
_PLANAR_CONCAVITY_WEIGHT_EPS = 0.05


def _planar_boundary_regularize(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    group_dist: np.ndarray,
    seed_faces: np.ndarray,
    seed_groups: np.ndarray,
    thickness: Optional[np.ndarray] = None,
    concavity_weight: float = _MANUAL_SEED_CONCAVITY_WEIGHT,
    thickness_weight: float = _MANUAL_SEED_THICKNESS_WEIGHT,
    neighbors: Optional[list[list[int]]] = None,
    report_out: Optional[list] = None,
) -> np.ndarray:
    """隣接パーツペアの境界に平面をフィットし、曖昧帯を平面で再割当する。

    ユーザー提案「腕は前方の脇部分で検出した区切りをそのまま肩へ伸ばす」の
    一般化。正面の深い溝など信頼できる境界区間(凹み強度が高い)に重み付きで
    平面をフィットし、確信度の低い区間(肩の上・背面など、Dijkstraの1位と
    2位のグループ距離が拮抗している曖昧帯)だけを平面のどちら側かで
    再割当する。ぬいぐるみの取付口は平面的な楕円が縫いやすいため、
    用途(型紙)とも整合する。

    手順(ペアごと、境界エッジ数の多い順に逐次適用):
        1. ペア(A,B)の境界エッジ(現ラベル基準)を収集。少なすぎればスキップ。
        2. 境界エッジ中点を重み付きPCA(重心=重み付き平均、法線=重み付き
           共分散の最小固有ベクトル)により平面フィット。重みは
           **Dijkstraのエッジペナルティと同じ量**(w_c×凹み強度+w_t×|Δlog肉厚|)
           +下限ε: 「強い誘導信号(深い溝・肉厚急変)で止まった境界区間ほど
           信頼できる」という、分割アルゴリズム自身の信頼モデルと整合する
           重み付け。凹みのみだと生成メッシュの背面のように凹み信号が皆無の
           形状で信頼区間を識別できないため、肉厚遷移も含める。
        3. ガード(下記)を通過したら、A∪Bの曖昧帯(マージン小)の面を
           面重心の平面側で再割当。確信面は変更しない。
        4. 全ペア処理後、ユーザーシード面を含まない飛び地成分を元ラベルへ
           戻し、軽い多数決平滑化(ユーザーシード面ピン)を行う。

    ガード(ペア単位でスキップ、`report_out` に理由を記録):
        - `few_edges`: 境界エッジ数 < `_PLANAR_MIN_BOUNDARY_EDGES`
        - `few_anchors`: トリム反復後のアンカー数が不足
        - `non_planar`: アンカー集合の重み付き共分散の固有値比
          λ_min/λ_mid > `_PLANAR_MAX_EIGEN_RATIO`(境界が本質的に非平面=
          鞍状など。染み出し区間はトリムで除外済みのため、真に非平面な
          場合のみ発動)
        - `low_coverage`: アンカーの面内広がりが全境界の
          `_PLANAR_MIN_COVERAGE` 未満(サドル境界の局所的に平面な一部の弧
          だけが残った場合の誤切断防止)
        - `degenerate`: A/Bの確信面が平面の同じ側に集まる(平面が境界を
          分離していない)
        - `seed_side`: A・Bのユーザーシード面がロバスト平面の正しい側にない
        - `shrink`: 再割当でどちらかのパーツ面数が
          `_PLANAR_MIN_KEEP_FRACTION` 未満に縮む(異常な大変動)

    再割当はペアの全面を平面側で切り直す**フルカット**(ユーザー原案
    「区切りをそのまま伸ばす」)。曖昧帯限定の再割当では「確信して間違って
    いる染み出し領域」(マージンが大きく曖昧帯に入らない)を矯正できない
    ため(染み出し再現モックでの比較実測に基づく採用判断)。

    Args:
        mesh: 対象メッシュ。
        labels: (F,) 現在の面ラベル(グループID)。
        group_dist: (n_groups, F) Dijkstraのグループ別最短距離
            (`labels_from_seeds` が計算したもの。曖昧帯の判定に使う)。
        seed_faces: ユーザーシード面(仮想シードは含めない)。
        seed_groups: seed_faces に対応するグループID。
        neighbors: 面隣接リスト(省略時は内部で構築)。
        report_out: 指定された場合、ペアごとの適用結果
            `{"parts": [a, b], "applied": bool, "reason": str|None,
              "rms": float|None, "n_reassigned": int}` を追記する。

    Returns:
        (F,) int64 の更新済みラベル。
    """
    labels_before = labels.copy()
    labels = labels.copy()
    n_faces = len(labels)
    if n_faces == 0:
        return labels

    adjacency = mesh.face_adjacency
    if len(adjacency) == 0:
        return labels
    edges_v = mesh.face_adjacency_edges
    centers = mesh.triangles_center
    areas = mesh.area_faces
    vertices = mesh.vertices

    try:
        angles = mesh.face_adjacency_angles
        convex = mesh.face_adjacency_convex
        concave = np.where(convex, 0.0, angles / np.pi)
    except Exception:
        concave = np.zeros(len(adjacency))

    # 平面フィットの信頼度重み = Dijkstraのエッジペナルティと同じ量
    # (w_c×凹み + w_t×|Δlog肉厚|)+下限ε
    edge_trust = concavity_weight * concave
    if thickness is not None and thickness_weight > 0.0:
        log_t = np.log(np.clip(np.asarray(thickness, dtype=np.float64), 1e-6, None))
        dlog = np.abs(log_t[adjacency[:, 0]] - log_t[adjacency[:, 1]])
        edge_trust = edge_trust + thickness_weight * dlog
    edge_trust = edge_trust + _PLANAR_CONCAVITY_WEIGHT_EPS

    # 曖昧帯判定用マージン: 1位と2位のグループ距離の差(infは確信扱い。
    # 両方infの面はinf-inf=nanになるためerrstateで抑制してinfへ置換する)
    if group_dist.shape[0] < 2:
        return labels
    two_smallest = np.partition(group_dist, 1, axis=0)[:2]
    with np.errstate(invalid="ignore"):
        margin = two_smallest[1] - two_smallest[0]
    margin = np.where(np.isfinite(margin), margin, np.inf)

    # ペア列挙(初期ラベル基準、境界エッジ数の多い順)
    la = labels[adjacency[:, 0]]
    lb = labels[adjacency[:, 1]]
    diff0 = la != lb
    pair_counts: dict[tuple[int, int], int] = {}
    for i in np.where(diff0)[0]:
        key = (int(min(la[i], lb[i])), int(max(la[i], lb[i])))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    pair_order = sorted(pair_counts.keys(), key=lambda k: -pair_counts[k])

    def _report(
        a: int,
        b: int,
        applied: bool,
        reason: Optional[str],
        rms: Optional[float],
        n_re: int,
        anchor_rms: Optional[float] = None,
        n_anchors: int = 0,
    ):
        if report_out is not None:
            entry = {
                "parts": [int(a), int(b)],
                "applied": bool(applied),
                "n_reassigned": int(n_re),
            }
            if reason is not None:
                entry["reason"] = reason
            if rms is not None:
                entry["rms"] = round(float(rms), 3)
            if anchor_rms is not None:
                entry["anchor_rms"] = round(float(anchor_rms), 3)
                entry["n_anchors"] = int(n_anchors)
            report_out.append(entry)

    user_seed_faces = np.asarray(seed_faces, dtype=np.int64)
    user_seed_groups = np.asarray(seed_groups, dtype=np.int64)

    for a, b in pair_order:
        # 現ラベル基準でこのペアの境界エッジを再収集(先行ペアの適用で変わりうる)
        la_c = labels[adjacency[:, 0]]
        lb_c = labels[adjacency[:, 1]]
        pair_mask = ((la_c == a) & (lb_c == b)) | ((la_c == b) & (lb_c == a))
        idx = np.where(pair_mask)[0]
        if len(idx) < _PLANAR_MIN_BOUNDARY_EDGES:
            _report(a, b, False, "few_edges", None, 0, None, 0)
            continue

        # 境界エッジ中点と信頼度重み(Dijkstraエッジペナルティ+ε)
        mids = vertices[edges_v[idx]].mean(axis=1)
        w = edge_trust[idx]

        # --- アンカーベースのトリム反復フィット(ロバスト化) ------------------
        # 染み出してうねった境界区間(信頼度が低い=誘導信号の弱い面上を通る)
        # を排除し、信頼できる区間(深い溝・肉厚急変=trust大)を優先して
        # 平面をフィットする。初期集合は「trustが中央値の一定倍以上」の
        # 高信頼点(実際にコントラストがある場合のみ。一様trustの境界で
        # 上位N%を強制選別するとノイズ選別になり、任意の弧に傾いた平面が
        # フィットされて誤切断する=ダンベルモックで実測)。
        strong = w >= _PLANAR_STRONG_TRUST_RATIO * float(np.median(w))
        if int(strong.sum()) >= _PLANAR_MIN_BOUNDARY_EDGES:
            inlier = strong
        else:
            inlier = np.ones(len(idx), dtype=bool)
        centroid = mids.mean(axis=0)
        normal = np.array([0.0, 0.0, 1.0])
        for _ in range(_PLANAR_TRIM_ITERATIONS):
            wi = w[inlier]
            mi = mids[inlier]
            w_sum = float(np.sum(wi))
            centroid = (mi * wi[:, None]).sum(axis=0) / w_sum
            d_in = mi - centroid
            cov = (d_in * wi[:, None]).T @ d_in / w_sum
            _eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]  # 最小固有値の固有ベクトル=平面法線
            # アンカー集合内で残差の大きい点をトリム(集合外は復帰させない)
            res = np.abs((mids - centroid) @ normal)
            thresh = float(np.quantile(res[inlier], _PLANAR_INLIER_FRACTION))
            new_inlier = inlier & (res <= thresh)
            if int(new_inlier.sum()) < _PLANAR_MIN_BOUNDARY_EDGES:
                break
            if np.array_equal(new_inlier, inlier):
                break
            inlier = new_inlier

        # 最終アンカー集合で共分散を再計算し、法線・固有値比・RMSを確定する
        n_anchors = int(inlier.sum())
        w_anchor = w[inlier]
        m_anchor = mids[inlier]
        wa_sum = float(np.sum(w_anchor))
        centroid = (m_anchor * w_anchor[:, None]).sum(axis=0) / wa_sum
        d_anchor3 = m_anchor - centroid
        cov_a = (d_anchor3 * w_anchor[:, None]).T @ d_anchor3 / wa_sum
        eigvals_a, eigvecs_a = np.linalg.eigh(cov_a)
        normal = eigvecs_a[:, 0]
        eigen_ratio = float(eigvals_a[0] / max(eigvals_a[1], 1e-12))
        d_anchor = d_anchor3 @ normal
        anchor_rms = float(np.sqrt(np.sum(w_anchor * d_anchor**2) / max(wa_sum, 1e-12)))
        d_all = (mids - centroid) @ normal
        rms = float(np.sqrt(np.sum(w * d_all**2) / max(float(np.sum(w)), 1e-12)))

        if n_anchors < _PLANAR_MIN_BOUNDARY_EDGES:
            _report(a, b, False, "few_anchors", rms, 0, anchor_rms, n_anchors)
            continue
        # 品質ガードはアンカー(インライア)集合の固有値比で評価する
        # (全境界点RMSでの評価は「治したい染み出しがあるほど棄却される」
        # 自己矛盾になるため。`_PLANAR_MAX_EIGEN_RATIO` のコメント参照)。
        if eigen_ratio > _PLANAR_MAX_EIGEN_RATIO:
            _report(a, b, False, "non_planar", rms, 0, anchor_rms, n_anchors)
            continue
        # カバレッジガード: アンカーが境界全体のうち局所的な一部の弧に
        # 集中している場合(サドル境界のうち平面な一部だけが残った等)は、
        # その平面で全体を切ると誤切断になるためスキップする。
        d_all3 = mids - mids.mean(axis=0)
        cov_all = d_all3.T @ d_all3 / len(mids)
        eig_all = np.linalg.eigvalsh(cov_all)
        d_anc3u = m_anchor - m_anchor.mean(axis=0)
        cov_anc = d_anc3u.T @ d_anc3u / len(m_anchor)
        eig_anc = np.linalg.eigvalsh(cov_anc)
        coverage = float(np.sqrt(max(eig_anc[1], 0.0) / max(eig_all[1], 1e-12)))
        if coverage < _PLANAR_MIN_COVERAGE:
            _report(a, b, False, "low_coverage", rms, 0, anchor_rms, n_anchors)
            continue

        # 面重心の平面に対する符号付き距離と、A/Bの側の決定(確信面の中央値)
        area_a = float(np.sum(areas[labels == a]))
        area_b = float(np.sum(areas[labels == b]))
        part_scale = float(np.sqrt(max(min(area_a, area_b), 1e-12)))
        sd = (centers - centroid) @ normal
        margin_thresh = _PLANAR_MARGIN_FRACTION * part_scale
        mask_a = labels == a
        mask_b = labels == b
        conf_a = mask_a & (margin >= margin_thresh)
        conf_b = mask_b & (margin >= margin_thresh)
        # 確信面が無い場合は全面で代用(パーツ全体が曖昧な小パーツ)
        side_a = float(np.median(sd[conf_a])) if np.any(conf_a) else float(np.median(sd[mask_a]))
        side_b = float(np.median(sd[conf_b])) if np.any(conf_b) else float(np.median(sd[mask_b]))
        if side_a == 0.0 or side_b == 0.0 or np.sign(side_a) == np.sign(side_b):
            _report(a, b, False, "degenerate", rms, 0, anchor_rms, n_anchors)
            continue

        # シード保護: A・Bのユーザーシード面が(ロバスト)平面の正しい側にあること
        seed_ok = True
        for sf, sg in zip(user_seed_faces, user_seed_groups):
            if int(sg) == a and np.sign(sd[int(sf)]) != np.sign(side_a):
                seed_ok = False
                break
            if int(sg) == b and np.sign(sd[int(sf)]) != np.sign(side_b):
                seed_ok = False
                break
        if not seed_ok:
            _report(a, b, False, "seed_side", rms, 0, anchor_rms, n_anchors)
            continue

        # 再割当: 既定はペアの全面を平面側で切り直すフルカット
        # (`_PLANAR_FULL_CUT` 参照)。誤切断はシード保護・規模ガード・
        # 後段のシード連結性回復で防ぐ。
        if _PLANAR_FULL_CUT:
            pair_faces = np.where(mask_a | mask_b)[0]
        else:
            pair_faces = np.where((mask_a | mask_b) & (margin < margin_thresh))[0]
            if len(pair_faces) == 0:
                _report(a, b, False, "no_ambiguous", rms, 0, anchor_rms, n_anchors)
                continue
        candidate = labels.copy()
        to_a = np.sign(sd[pair_faces]) == np.sign(side_a)
        candidate[pair_faces[to_a]] = a
        candidate[pair_faces[~to_a]] = b

        # 再割当規模ガード: どちらかのパーツが極端に縮んだら異常としてスキップ
        n_a_old, n_b_old = int(mask_a.sum()), int(mask_b.sum())
        n_a_new = int(np.sum(candidate == a))
        n_b_new = int(np.sum(candidate == b))
        if (
            n_a_new < _PLANAR_MIN_KEEP_FRACTION * n_a_old
            or n_b_new < _PLANAR_MIN_KEEP_FRACTION * n_b_old
        ):
            _report(a, b, False, "shrink", rms, 0, anchor_rms, n_anchors)
            continue

        n_reassigned = int(np.sum(candidate != labels))
        labels = candidate
        _report(a, b, True, None, rms, n_reassigned, anchor_rms, n_anchors)

    # 1ペアも適用されなかった場合は完全に元のラベルを返す
    # (後処理の平滑化も行わない=ガード発動時の結果不変を保証する)。
    if np.array_equal(labels, labels_before):
        return labels_before

    # --- 後処理: シード連結性の回復 + 軽い多数決平滑化 -----------------------
    if neighbors is None:
        neighbors = _face_neighbors(mesh)

    # 各グループについて、ユーザーシード面を含まない飛び地成分は
    # 平面化前のラベルへ戻す(平面が別の部位をまたいで切ってしまった場合の保険)。
    seed_faces_of_group: dict[int, set[int]] = {}
    for sf, sg in zip(user_seed_faces, user_seed_groups):
        seed_faces_of_group.setdefault(int(sg), set()).add(int(sf))
    for g in np.unique(labels):
        g = int(g)
        face_idx = np.where(labels == g)[0]
        comps = _connected_subcomponents(face_idx, neighbors)
        if len(comps) <= 1:
            continue
        anchor_faces = seed_faces_of_group.get(g, set())
        for comp in comps:
            if anchor_faces & set(comp):
                continue
            # シードを含まない成分: 最大成分は許容(シードなしグループの保険)、
            # それ以外は平面化前のラベルへ戻す
            if not anchor_faces and len(comp) == max(len(c) for c in comps):
                continue
            for f in comp:
                labels[f] = labels_before[f]

    # 軽い多数決平滑化(ユーザーシード面ピン、2回)
    seed_face_set = set(int(f) for f in user_seed_faces)
    for _ in range(2):
        new_labels = labels.copy()
        for f in range(n_faces):
            if f in seed_face_set or not neighbors[f]:
                continue
            nbr_labels = labels[neighbors[f]]
            values, counts = np.unique(nbr_labels, return_counts=True)
            if counts.max() > len(neighbors[f]) / 2:
                new_labels[f] = int(values[np.argmax(counts)])
        labels = new_labels

    return labels


def labels_from_seeds(
    mesh: trimesh.Trimesh,
    seed_points: np.ndarray,
    seed_groups: Optional[np.ndarray] = None,
    concavity_weight: float = _MANUAL_SEED_CONCAVITY_WEIGHT,
    thickness_weight: float = _MANUAL_SEED_THICKNESS_WEIGHT,
    smooth_iterations: int = _MANUAL_SEED_SMOOTH_ITERATIONS,
    thickness: Optional[np.ndarray] = None,
    propagate_opposite: bool = True,
    virtual_seeds_out: Optional[list] = None,
    virtual_skips_out: Optional[list] = None,
    planar_boundaries: bool = True,
    planar_fit_out: Optional[list] = None,
) -> np.ndarray:
    """ユーザー指定のシード点(3D座標)から測地距離ベースでパーツを分割する。

    Graph Cut相当の効果を追加依存なしで得るため、以下の手順を取る:
        1. 各シード点を最近傍の面(三角形)へスナップする。
        2. `propagate_opposite=True` の場合、各シード面の反対側スキン
           (逆法線レイの first exit)に同グループの**仮想シード**を生成する
           (`_opposite_virtual_seed_faces`。誤配置ガード付き)。正面にしか
           シードを打っていない場合の背面の割当を改善する。
        3. 凹み・肉厚遷移誘導の重み付き面双対グラフ(`_dual_graph_weighted`)
           上で、シード面群(ユーザー+仮想)からのマルチソース重み付き
           Dijkstraを行い、各面を「最も近いシードグループ」に割り当てる
           (scipy.sparse.csgraph.dijkstra、`min_only=False` で全シードからの
           距離行列を求め、グループ内で min → グループ間で argmin する。
           シード数は仮想込みでも高々100程度、面数は数千〜数万のオーダーの
           ため、(N_seeds, F) 距離行列を保持しても問題ない)。
        4. メッシュが非連結成分を含む等で双対グラフ上到達不能な面が残る場合、
           シード面へのユークリッド距離による最近傍フォールバックで埋める。
        5. 境界のギザつきを抑えるため、**ユーザーシード面のみ**ラベルを固定した
           上で隣接面の多数決による平滑化を数イテレーション行う(仮想シードは
           ピン固定しない: 誤って別パーツ領域に落ちた場合に周囲の多数決で
           自然に矯正されるように。ユーザーの明示シードとの信頼度差)。

    Args:
        mesh: 対象メッシュ(前処理済み推奨。`prepare_mesh`参照)。ビューアで
            表示中のモデルと同じローカル座標系(mm)であること。
        seed_points: (N, 3) float配列。N=2〜48を想定(呼び出し側でバリデーション
            する想定だが、本関数自体は制約を課さない)。
        seed_groups: (N,) int配列(任意)。各シードが属するパーツ(名前グループ)
            の0始まり連番ID。同一グループの複数シードはマルチソース(そのグループ
            の全シード面からの最短距離)として扱われる。例: 「胴体」を正面と
            背面に1点ずつ打つと、両点から波面が広がり背面の割当が改善する。
            None の場合は1シード=1グループ(従来動作)。
        concavity_weight: 凹エッジを跨ぐコストの倍率(`_dual_graph_weighted`
            参照)。既定値の根拠は `_MANUAL_SEED_CONCAVITY_WEIGHT` のコメント参照。
        thickness_weight: 肉厚遷移を跨ぐコストの倍率(`_dual_graph_weighted`
            参照)。0で無効。既定値の根拠は `_MANUAL_SEED_THICKNESS_WEIGHT` の
            コメント参照。0より大きい場合、`compute_local_thickness` による
            肉厚推定(数千〜数万面で数秒程度)が追加で走る。
        smooth_iterations: 境界平滑化(多数決)の反復回数。
        thickness: (F,) 面ごとの局所肉厚(任意)。呼び出し側で計算済みの場合に
            渡すと再計算を省ける。None かつ `thickness_weight > 0` の場合は
            内部で `compute_local_thickness` を計算する。
        propagate_opposite: True の場合、各ユーザーシードの反対側スキンに
            同グループの仮想シードを自動生成する(既定True)。
        virtual_seeds_out: 指定された場合、採用された仮想シードの情報
            `{"face": int, "x": float, "y": float, "z": float, "group": int}`
            (座標は面重心)を呼び出し側のリストへ破壊的に追記する
            (デバッグ・`pattern_parts.json` への記録用)。
        virtual_skips_out: 指定された場合、仮想シード候補のスキップ情報
            `{"seed_index": int, "group": int, "reason": str}` を追記する
            (reason: no_hit/tunnel/taken/ownership/balance。
            balance はペア単位均衡判定による棄却で seed_index=-1)。
        planar_boundaries: True(既定)の場合、多数決平滑化の後に隣接パーツ
            ペアの境界へ平面をフィットし、曖昧帯(Dijkstraの1位/2位グループ
            距離が拮抗する面)を平面のどちら側かで再割当する
            (`_planar_boundary_regularize` 参照。背面・肩の上など誘導信号の
            弱い区間の境界を、正面の深い溝で決まった平面に沿わせる)。
        planar_fit_out: 指定された場合、ペアごとの平面フィット適用結果を
            追記する(`_planar_boundary_regularize` の `report_out`)。

    Returns:
        (F,) int64 配列。各面が属するパーツ(グループ)のID(0始まり、
        `seed_groups` の値に対応。`seed_groups=None` なら `seed_points` の
        行インデックスに対応)。

    Raises:
        ValueError: グループ数(パーツ数)が2未満、`seed_groups` の長さが
            `seed_points` と不一致、または**異なるグループ**の2つ以上の
            シードが同一の面にスナップした場合(パーツが縮退するため。
            同一グループ内の重複スナップは単に冗長なだけなので許容する)。
    """
    seed_points = np.asarray(seed_points, dtype=np.float64)
    if seed_points.ndim != 2 or seed_points.shape[1] != 3:
        raise ValueError("seed_pointsは(N, 3)のndarrayである必要があります。")
    n_seeds = len(seed_points)

    if seed_groups is None:
        seed_groups = np.arange(n_seeds, dtype=np.int64)
    else:
        seed_groups = np.asarray(seed_groups, dtype=np.int64)
        if seed_groups.shape != (n_seeds,):
            raise ValueError("seed_groupsはseed_pointsと同じ長さの1次元配列である必要があります。")

    unique_groups = np.unique(seed_groups)
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError("パーツ(シードの名前グループ)は2つ以上である必要があります。")
    # グループIDは0始まり連番を想定(呼び出し側で正規化する)。念のため
    # 非連番でも動くよう、ここで連番へ写像する。
    group_remap = {int(g): i for i, g in enumerate(unique_groups)}
    seed_groups = np.array([group_remap[int(g)] for g in seed_groups], dtype=np.int64)

    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros((0,), dtype=np.int64)

    # --- 1. シード→最近傍面スナップ -----------------------------------------
    seed_faces = _snap_points_to_faces(mesh, seed_points)

    # 異なるグループのシードが同一面にスナップした場合のみエラーとする
    # (同一グループの重複はマルチソースとして単に冗長なだけなので許容)。
    face_to_group: dict[int, int] = {}
    for sf, sg in zip(seed_faces, seed_groups):
        sf_int, sg_int = int(sf), int(sg)
        if sf_int in face_to_group and face_to_group[sf_int] != sg_int:
            raise ValueError(
                "異なるパーツ名のシードが同一の面にスナップしました。シード点をもっと離してください。"
            )
        face_to_group[sf_int] = sg_int

    # --- 2. 反対側スキンへの仮想シード候補生成(背面の割当改善) --------------
    # 仮想シードのトンネル検出には安定した局所肉厚推定が要るため、
    # (thickness_weight=0で本来不要な場合でも)ここで計算しておく。
    if thickness is None and (thickness_weight > 0.0 or propagate_opposite):
        thickness = compute_local_thickness(mesh, seed=0)

    virt_faces = np.zeros(0, dtype=np.int64)
    virt_groups = np.zeros(0, dtype=np.int64)
    if propagate_opposite:
        virt_faces, virt_groups = _opposite_virtual_seed_faces(
            mesh, seed_faces, seed_groups, thickness=thickness, skip_reasons_out=virtual_skips_out
        )

    # --- 3. 重み付きマルチソースDijkstra(ユーザー+仮想候補を一括計算) ------
    graph = _dual_graph_weighted(
        mesh, concavity_weight, thickness=thickness, thickness_weight=thickness_weight
    )
    n_user = len(seed_faces)
    all_faces = np.concatenate([seed_faces, virt_faces]) if len(virt_faces) else seed_faces
    all_groups = np.concatenate([seed_groups, virt_groups]) if len(virt_faces) else seed_groups
    dist_matrix = dijkstra(graph, indices=all_faces, directed=False, min_only=False)

    # --- 3b. ペア単位の均衡判定(仮想シード候補の採用可否) -------------------
    # 予備割当(ユーザーシードのみ、dist_matrixの先頭n_user行を再利用)で
    # パーツ隣接関係を求め、グループGの候補は「Gと隣接する各グループHが
    # 候補を持つ、またはHがコンパクト(Hの全面が自グループシードから近い
    # =背面を奪われるリスクが低い小パーツ)」の場合のみ採用する。
    # 片側の大パーツだけ背面が強化されると隣接境界が偏って悪化するため
    # (頭+胴の実測: 13.7→23.4mm)、その形だけをペア単位で棄却する
    # (旧実装の「1グループでも候補ゼロなら全棄却」は多パーツ構成では
    # 事実上常時OFFになるため廃止)。
    if propagate_opposite and len(virt_faces) > 0:
        user_dist = np.full((n_groups, n_faces), np.inf)
        for g in range(n_groups):
            rows = np.where(seed_groups == g)[0]
            user_dist[g] = np.min(dist_matrix[:n_user][rows], axis=0)
        prelim_reachable = np.isfinite(user_dist).any(axis=0)
        prelim = np.zeros(n_faces, dtype=np.int64)
        if np.any(prelim_reachable):
            prelim[prelim_reachable] = np.argmin(user_dist[:, prelim_reachable], axis=0)
        if not np.all(prelim_reachable):
            centers_p = mesh.triangles_center
            seed_centers_p = centers_p[seed_faces]
            un_idx = np.where(~prelim_reachable)[0]
            d_p = np.linalg.norm(
                centers_p[un_idx][:, None, :] - seed_centers_p[None, :, :], axis=2
            )
            prelim[un_idx] = seed_groups[np.argmin(d_p, axis=1)]

        # 所有権チェック: exit面が自グループから測地的に遠すぎる
        # (=レイが首などを貫通して隣のパーツのスキンに抜けた)候補を棄却。
        # 「自グループまでの距離 ≤ 他グループまでの最小距離×許容比」を要求する。
        cand_idx = np.arange(len(virt_faces))
        own_ok = np.ones(len(virt_faces), dtype=bool)
        for k, (vf, vg) in enumerate(zip(virt_faces, virt_groups)):
            vf_i, vg_i = int(vf), int(vg)
            d_own_g = float(user_dist[vg_i, vf_i])
            other_ds = [user_dist[h, vf_i] for h in range(n_groups) if h != vg_i]
            d_other_min = float(np.min(other_ds)) if other_ds else np.inf
            if not np.isfinite(d_own_g):
                own_ok[k] = False
            elif np.isfinite(d_other_min) and d_own_g > _VIRTUAL_SEED_OWNERSHIP_RATIO * d_other_min:
                own_ok[k] = False
            if not own_ok[k] and virtual_skips_out is not None:
                virtual_skips_out.append(
                    {"seed_index": -1, "group": vg_i, "reason": "ownership"}
                )
        cand_idx = cand_idx[own_ok]
        virt_faces = virt_faces[own_ok]
        virt_groups = virt_groups[own_ok]

        # パーツ隣接グラフ(予備割当の境界エッジから)
        adjacency_m = mesh.face_adjacency
        pa = prelim[adjacency_m[:, 0]]
        pb = prelim[adjacency_m[:, 1]]
        neighbor_groups: dict[int, set[int]] = {g: set() for g in range(n_groups)}
        for k in np.where(pa != pb)[0]:
            neighbor_groups[int(pa[k])].add(int(pb[k]))
            neighbor_groups[int(pb[k])].add(int(pa[k]))

        # コンパクト判定: 予備割当の面積比が小さい=耳・しっぽのような小パーツ
        areas_all = mesh.area_faces
        total_area = float(np.sum(areas_all)) or 1.0
        compact = np.zeros(n_groups, dtype=bool)
        for g in range(n_groups):
            frac = float(np.sum(areas_all[prelim == g])) / total_area
            compact[g] = frac <= _VIRTUAL_BALANCE_COMPACT_AREA_FRACTION

        groups_with_candidates = set(int(g) for g in virt_groups)
        adopted_groups: set[int] = set()
        for g in sorted(groups_with_candidates):
            ok = True
            for h in neighbor_groups.get(g, set()):
                if h in groups_with_candidates or compact[h]:
                    continue
                ok = False
                break
            if ok:
                adopted_groups.add(g)
            elif virtual_skips_out is not None:
                virtual_skips_out.append(
                    {"seed_index": -1, "group": int(g), "reason": "balance"}
                )

        keep_local = np.array(
            [i for i, vg in enumerate(virt_groups) if int(vg) in adopted_groups],
            dtype=np.int64,
        )
        # dist_matrix行の参照は「元の候補インデックス」で行う(所有権チェックで
        # 候補が間引かれているため)
        keep = cand_idx[keep_local] if len(keep_local) else np.zeros(0, dtype=np.int64)
        virt_faces = virt_faces[keep_local] if len(keep_local) else np.zeros(0, dtype=np.int64)
        virt_groups = virt_groups[keep_local] if len(keep_local) else np.zeros(0, dtype=np.int64)

        # 採用された仮想シードのみ報告・使用する
        if virtual_seeds_out is not None and len(virt_faces) > 0:
            centers_v = mesh.triangles_center
            for vf, vg in zip(virt_faces, virt_groups):
                c = centers_v[int(vf)]
                virtual_seeds_out.append(
                    {
                        "face": int(vf),
                        "x": float(c[0]),
                        "y": float(c[1]),
                        "z": float(c[2]),
                        "group": int(vg),
                    }
                )

        # 採用行のみで dist_matrix / all_* を組み直す
        used_rows = np.concatenate([np.arange(n_user), n_user + keep]) if len(keep) else np.arange(n_user)
        dist_matrix = dist_matrix[used_rows]
        all_faces = np.concatenate([seed_faces, virt_faces]) if len(virt_faces) else seed_faces
        all_groups = np.concatenate([seed_groups, virt_groups]) if len(virt_faces) else seed_groups

    # --- 3c. グループ距離への縮約 --------------------------------------------
    # dist_matrix: (n_all_seeds, n_faces)。グループ内で最小を取って
    # (n_groups, n_faces) に縮約する(同一グループの複数シード+採用済み
    # 仮想シードをマルチソースとして扱う)。全シードから到達不能(inf)な面は
    # 後段のフォールバックで埋める。
    group_dist = np.full((n_groups, n_faces), np.inf)
    for g in range(n_groups):
        rows = np.where(all_groups == g)[0]
        group_dist[g] = np.min(dist_matrix[rows], axis=0)

    reachable = np.isfinite(group_dist).any(axis=0)
    labels = np.zeros(n_faces, dtype=np.int64)
    if np.any(reachable):
        labels[reachable] = np.argmin(group_dist[:, reachable], axis=0)

    # --- 4. 到達不能面のフォールバック(シード面へのユークリッド距離) --------
    if not np.all(reachable):
        centers = mesh.triangles_center
        seed_centers = centers[all_faces]
        unreachable_idx = np.where(~reachable)[0]
        d = np.linalg.norm(
            centers[unreachable_idx][:, None, :] - seed_centers[None, :, :], axis=2
        )
        labels[unreachable_idx] = all_groups[np.argmin(d, axis=1)]

    # ユーザーシード面自身は必ずそのグループのラベルにしておく(Dijkstraの
    # 自己距離0で通常はそうなるはずだが、フォールバック分岐や同点タイの際の
    # 保険)。仮想シード面はピンしない(信頼度が低いため)。
    labels[seed_faces] = seed_groups

    # --- 5. 境界の多数決平滑化(ユーザーシード面のみ固定、仮想シードは
    #        誤配置の可能性があるため固定せず多数決に委ねる) -------------------
    neighbors: Optional[list[list[int]]] = None
    if smooth_iterations > 0:
        neighbors = _face_neighbors(mesh)
        seed_face_set = set(int(f) for f in seed_faces)
        for _ in range(smooth_iterations):
            new_labels = labels.copy()
            for f in range(n_faces):
                if f in seed_face_set or not neighbors[f]:
                    continue
                nbr_labels = labels[neighbors[f]]
                values, counts = np.unique(nbr_labels, return_counts=True)
                majority = int(values[np.argmax(counts)])
                if counts.max() > len(neighbors[f]) / 2:
                    new_labels[f] = majority
            labels = new_labels

    # --- 6. 境界の平面フィット正則化(曖昧帯を平面で再割当) ------------------
    if planar_boundaries:
        labels = _planar_boundary_regularize(
            mesh,
            labels,
            group_dist,
            seed_faces,
            seed_groups,
            thickness=thickness,
            concavity_weight=concavity_weight,
            thickness_weight=thickness_weight,
            neighbors=neighbors,
            report_out=planar_fit_out,
        )

    return labels


def _diffuse_thickness(
    n_faces: int, thickness: np.ndarray, sample_idx: np.ndarray, graph: csr_matrix
) -> np.ndarray:
    """サンプルされた面から、双対グラフ上の最短路で最も近いサンプルの値を
    未サンプル面へ伝播させる(疎な観測値の補間)。"""
    valid_samples = sample_idx[np.isfinite(thickness[sample_idx])]
    if len(valid_samples) == 0:
        return np.ones(n_faces)

    _, _, sources = dijkstra(
        graph,
        indices=valid_samples,
        directed=False,
        return_predecessors=True,
        min_only=True,
    )
    fallback = float(np.nanmedian(thickness[valid_samples]))
    filled = np.full(n_faces, fallback)
    reachable = sources >= 0
    filled[reachable] = thickness[sources[reachable]]
    return filled


def _face_neighbors(mesh: trimesh.Trimesh) -> list[list[int]]:
    n_faces = len(mesh.faces)
    neighbors: list[list[int]] = [[] for _ in range(n_faces)]
    for a, b in mesh.face_adjacency:
        neighbors[a].append(int(b))
        neighbors[b].append(int(a))
    return neighbors


def _smooth_over_dual_graph(values: np.ndarray, neighbors: list[list[int]], n_iters: int = 5) -> np.ndarray:
    """双対グラフ上での近傍平均平滑化(ノイズの多い肉厚推定を滑らかにし、
    k-meansクラスタリング前にパーツ境界のギザつきを抑える)。"""
    values = values.copy()
    for _ in range(n_iters):
        new_values = values.copy()
        for f, nbrs in enumerate(neighbors):
            if not nbrs:
                continue
            new_values[f] = 0.5 * values[f] + 0.5 * float(np.mean(values[nbrs]))
        values = new_values
    return values


def compute_local_thickness(
    mesh: trimesh.Trimesh,
    n_samples: int = _DEFAULT_N_SAMPLES,
    rays_per_sample: int = _RAYS_PER_SAMPLE,
    cone_deg: float = _CONE_DEG,
    seed: int = 0,
    smooth_iterations: int = 5,
) -> np.ndarray:
    """全面の局所肉厚(mm)を返す(サンプル+拡散補間+平滑化済み)。

    Args:
        mesh: 対象メッシュ。
        n_samples: 直接レイキャストする面数の上限(それ以上は拡散補間)。
        rays_per_sample: サンプル1面あたりのレイ数(コーン内)。
        cone_deg: 内向き法線からのコーン半頂角(度)。
        seed: 乱数シード。
        smooth_iterations: 双対グラフ上の平滑化反復回数。

    Returns:
        (F,) float64 配列。各面の局所肉厚(mm相当)。
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros((0,), dtype=np.float64)

    thickness, sample_idx = _sample_local_thickness(mesh, n_samples, rays_per_sample, cone_deg, seed)
    graph = _dual_graph(mesh)
    filled = _diffuse_thickness(n_faces, thickness, sample_idx, graph)
    neighbors = _face_neighbors(mesh)
    smoothed = _smooth_over_dual_graph(filled, neighbors, n_iters=smooth_iterations)
    return smoothed


# --------------------------------------------------------------------------
# 連結成分・極小パーツ吸収 (segment.pyと同様の考え方)
# --------------------------------------------------------------------------
def _split_into_components(labels: np.ndarray, neighbors: list[list[int]]) -> list[tuple[int, list[int]]]:
    n_faces = len(labels)
    visited = np.zeros(n_faces, dtype=bool)
    components: list[tuple[int, list[int]]] = []
    for start in range(n_faces):
        if visited[start]:
            continue
        label = labels[start]
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for nb in neighbors[f]:
                if not visited[nb] and labels[nb] == label:
                    visited[nb] = True
                    stack.append(nb)
        components.append((int(label), comp))
    return components


def _absorb_tiny_parts(
    labels: np.ndarray,
    neighbors: list[list[int]],
    face_areas: np.ndarray,
    face_centers: np.ndarray,
    min_area_fraction: float = _MIN_AREA_FRACTION,
    max_iters: int = 2000,
    protected_labels: Optional[set[int]] = None,
) -> np.ndarray:
    """極小パーツを隣接パーツへ吸収する。

    通常は境界を共有する隣接パーツ(最も共有面数が多いもの)へ吸収するが、
    連結成分抽出の副作用で稀に生じる完全孤立断片(縮退三角形等、双対グラフ上
    どの他パーツとも隣接しない断片)は、重心が最も近いパーツへ吸収する
    フォールバックを用いる。

    Args:
        protected_labels: 指定された場合、これらのラベルは面積が閾値未満でも
            「極小パーツ」とみなさず吸収対象から除外する(手動シード誘導
            `guidance="manual"` で、ユーザーが明示的にクリックした部位が
            面数の少なさだけを理由に消滅しないようにするため)。
    """
    labels = labels.copy()
    total_area = float(np.sum(face_areas)) or 1.0
    protected_labels = protected_labels or set()

    for _ in range(max_iters):
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            break
        areas = {int(lbl): float(np.sum(face_areas[labels == lbl])) for lbl in unique_labels}
        tiny = [
            lbl for lbl, a in areas.items()
            if a / total_area < min_area_fraction and lbl not in protected_labels
        ]
        if not tiny:
            break
        smallest = min(tiny, key=lambda lbl: areas[lbl])
        faces_of_label = np.where(labels == smallest)[0]

        neighbor_label_counts: dict[int, int] = {}
        for f in faces_of_label:
            for nb in neighbors[f]:
                nb_label = int(labels[nb])
                if nb_label != smallest:
                    neighbor_label_counts[nb_label] = neighbor_label_counts.get(nb_label, 0) + 1

        if neighbor_label_counts:
            new_label = max(neighbor_label_counts.items(), key=lambda kv: kv[1])[0]
        else:
            other_mask = labels != smallest
            if not np.any(other_mask):
                break
            other_tree = cKDTree(face_centers[other_mask])
            other_labels = labels[other_mask]
            frag_center = face_centers[faces_of_label].mean(axis=0)
            _, nn_idx = other_tree.query(frag_center, k=1)
            new_label = int(other_labels[nn_idx])

        labels[faces_of_label] = new_label

    return labels


def _enforce_part_connectivity(
    labels: np.ndarray,
    neighbors: list[list[int]],
    face_centers: np.ndarray,
    face_areas: Optional[np.ndarray] = None,
    max_iters: int = 8,
) -> np.ndarray:
    """各パーツが単一連結成分になるよう、飛び地(非最大連結成分)を
    再割当する。

    `_absorb_tiny_parts` は「パーツ全体」の面積が閾値未満かどうかしか
    見ないため、十分大きなパーツの内部に生じた微小な孤立断片
    (簡略化による縮退三角形ペア等、双対グラフ上どの他パーツとも
    隣接しない断片)は見逃されうる。ここではパーツ単位で連結成分を数え、
    最大成分のみを残し、それ以外は境界を共有する隣接パーツ(なければ
    重心最近傍のパーツ)へ再割当する。

    面積ゼロの縮退三角形が(メッシュ全体から見ても)完全に孤立した
    浮遊断片としてメッシュに残っている場合、どのパーツへ再割当しても
    「連結」にはならない(そもそもメッシュの他部分と非隣接なため)。
    このような実質無害な断片で無限に再試行し続けないよう、
    `face_areas` が与えられた場合は面積ゼロに近い断片は1回だけ
    近傍パーツへ寄せて以降は無視する。
    """
    labels = labels.copy()
    negligible_area = 1e-9
    for _ in range(max_iters):
        changed = False
        for label in np.unique(labels):
            face_idx = np.where(labels == label)[0]
            comps = _connected_subcomponents(face_idx, neighbors)
            if len(comps) <= 1:
                continue
            comps_sorted = sorted(comps, key=len, reverse=True)
            for stray in comps_sorted[1:]:
                if face_areas is not None and float(np.sum(face_areas[stray])) <= negligible_area:
                    # 面積ゼロの浮遊断片: 見た目・体積計算に実害がないため、
                    # 隣接パーツがあれば寄せるが、以降の反復では追跡しない
                    # (無限ループ防止。連結性判定はpart_statsで面積加重するため
                    # このまま残っても「連結」と判定される)。
                    neighbor_label_counts_once: dict[int, int] = {}
                    for f in stray:
                        for nb in neighbors[f]:
                            nb_label = int(labels[nb])
                            if nb_label != label:
                                neighbor_label_counts_once[nb_label] = (
                                    neighbor_label_counts_once.get(nb_label, 0) + 1
                                )
                    if neighbor_label_counts_once:
                        new_label = max(neighbor_label_counts_once.items(), key=lambda kv: kv[1])[0]
                        for f in stray:
                            labels[f] = new_label
                    continue

                neighbor_label_counts: dict[int, int] = {}
                for f in stray:
                    for nb in neighbors[f]:
                        nb_label = int(labels[nb])
                        if nb_label != label:
                            neighbor_label_counts[nb_label] = neighbor_label_counts.get(nb_label, 0) + 1
                if neighbor_label_counts:
                    new_label = max(neighbor_label_counts.items(), key=lambda kv: kv[1])[0]
                else:
                    other_mask = labels != label
                    if not np.any(other_mask):
                        continue
                    other_tree = cKDTree(face_centers[other_mask])
                    other_labels = labels[other_mask]
                    frag_center = face_centers[stray].mean(axis=0)
                    _, nn_idx = other_tree.query(frag_center, k=1)
                    new_label = int(other_labels[nn_idx])
                for f in stray:
                    labels[f] = new_label
                changed = True
        if not changed:
            break
    return labels


def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    mapping = {int(old): new for new, old in enumerate(unique_labels)}
    return np.array([mapping[int(lbl)] for lbl in labels], dtype=np.int64)


# --------------------------------------------------------------------------
# くびれ(凹二面角)誘導による部分領域の2分割 (segment.pyのVoronoi手法を再利用)
# --------------------------------------------------------------------------
def _split_subregion(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    k: int,
    seed: int = 0,
) -> dict[int, int]:
    """面集合を凹二面角誘導のfarthest-point + 多始点最短路でk分割する。

    Returns:
        {face_index(mesh基準): local_label(0..k-1)}
    """
    face_indices = np.asarray(face_indices, dtype=np.int64)
    n = len(face_indices)
    if k <= 1 or n <= max(k, 3):
        return {int(f): 0 for f in face_indices}

    local_id = {int(f): i for i, f in enumerate(face_indices)}
    centers = mesh.triangles_center[face_indices]

    adjacency = mesh.face_adjacency
    try:
        angles = mesh.face_adjacency_angles
        convex = mesh.face_adjacency_convex
        concave_strength = np.where(convex, 0.0, angles / np.pi)
    except Exception:
        concave_strength = np.zeros(len(adjacency))

    face_set = set(int(f) for f in face_indices)
    mask_a = np.array([int(x) in face_set for x in adjacency[:, 0]])
    mask_b = np.array([int(x) in face_set for x in adjacency[:, 1]])
    both = mask_a & mask_b
    sub_adj = adjacency[both]
    sub_concave = concave_strength[both]

    if len(sub_adj) == 0:
        return {int(f): i % k for i, f in enumerate(face_indices)}

    a_local = np.array([local_id[int(x)] for x in sub_adj[:, 0]])
    b_local = np.array([local_id[int(x)] for x in sub_adj[:, 1]])
    base_dist = np.linalg.norm(centers[a_local] - centers[b_local], axis=1)
    scale = float(np.median(base_dist[base_dist > 0])) if np.any(base_dist > 0) else 1.0
    base_dist = np.maximum(base_dist, scale * 1e-4)
    # 凹エッジ(くびれ)を通るコストを上げ、Voronoi境界がそこに乗りやすくする
    # (segment.pyのdocstringに詳述した設計判断と同じ理由)。
    weights = base_dist * (1.0 + 5.0 * sub_concave)

    rows = np.concatenate([a_local, b_local])
    cols = np.concatenate([b_local, a_local])
    data = np.concatenate([weights, weights])
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))

    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, n))]
    min_dist = None
    for _ in range(k - 1):
        d = dijkstra(graph, indices=seeds[-1], directed=False)
        d = np.where(np.isfinite(d), d, 0.0)
        min_dist = d if min_dist is None else np.minimum(min_dist, d)
        candidate = min_dist.copy()
        candidate[seeds] = -1.0
        next_seed = int(np.argmax(candidate))
        if candidate[next_seed] <= 0:
            remaining = [i for i in range(n) if i not in seeds]
            if not remaining:
                break
            next_seed = int(rng.choice(remaining))
        seeds.append(next_seed)

    dist_matrix = dijkstra(graph, indices=seeds, directed=False)
    dist_matrix = np.where(np.isfinite(dist_matrix), dist_matrix, np.inf)
    local_labels = np.argmin(dist_matrix, axis=0)

    return {int(f): int(local_labels[local_id[int(f)]]) for f in face_indices}


def _split_subregion_from_seeds(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    seed_faces: list[int],
) -> dict[int, int]:
    """`_split_subregion` と同じ凹エッジ誘導Voronoi分割だが、farthest-point
    サンプリングの代わりに `seed_faces`(mesh基準の面インデックス)を
    そのままシードとして使う版。画像ラベルのチャンク重心最近傍面を
    シードにすることで、画像領域境界に沿った分割を誘導する。

    Returns:
        {face_index(mesh基準): local_label(0..len(seed_faces)-1)}
    """
    face_indices = np.asarray(face_indices, dtype=np.int64)
    n = len(face_indices)
    k = len(seed_faces)
    if k <= 1 or n <= max(k, 3):
        return {int(f): 0 for f in face_indices}

    local_id = {int(f): i for i, f in enumerate(face_indices)}
    centers = mesh.triangles_center[face_indices]

    adjacency = mesh.face_adjacency
    try:
        angles = mesh.face_adjacency_angles
        convex = mesh.face_adjacency_convex
        concave_strength = np.where(convex, 0.0, angles / np.pi)
    except Exception:
        concave_strength = np.zeros(len(adjacency))

    face_set = set(int(f) for f in face_indices)
    mask_a = np.array([int(x) in face_set for x in adjacency[:, 0]])
    mask_b = np.array([int(x) in face_set for x in adjacency[:, 1]])
    both = mask_a & mask_b
    sub_adj = adjacency[both]
    sub_concave = concave_strength[both]

    if len(sub_adj) == 0:
        return {int(f): i % k for i, f in enumerate(face_indices)}

    a_local = np.array([local_id[int(x)] for x in sub_adj[:, 0]])
    b_local = np.array([local_id[int(x)] for x in sub_adj[:, 1]])
    base_dist = np.linalg.norm(centers[a_local] - centers[b_local], axis=1)
    scale = float(np.median(base_dist[base_dist > 0])) if np.any(base_dist > 0) else 1.0
    base_dist = np.maximum(base_dist, scale * 1e-4)
    weights = base_dist * (1.0 + 5.0 * sub_concave)

    rows = np.concatenate([a_local, b_local])
    cols = np.concatenate([b_local, a_local])
    data = np.concatenate([weights, weights])
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))

    seeds_local = []
    for sf in seed_faces:
        sf = int(sf)
        if sf in local_id:
            seeds_local.append(local_id[sf])
    seeds_local = sorted(set(seeds_local))
    if len(seeds_local) < 2:
        return {int(f): 0 for f in face_indices}

    dist_matrix = dijkstra(graph, indices=seeds_local, directed=False)
    dist_matrix = np.where(np.isfinite(dist_matrix), dist_matrix, np.inf)
    local_labels = np.argmin(dist_matrix, axis=0)

    return {int(f): int(local_labels[local_id[int(f)]]) for f in face_indices}


def _image_guided_subdivide(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    face_areas: np.ndarray,
    image_labels: np.ndarray,
    seed: int = 0,
    significant_chunk_fraction: float = _IMG_SIGNIFICANT_CHUNK_FRACTION,
    source_image_label_of: Optional[dict[int, int]] = None,
) -> np.ndarray:
    """大パーツ(全面積比 `_IMG_LARGE_PART_AREA_FRACTION` 超)について、画像の
    2D領域ラベルがそのパーツを2つ以上の有意なチャンクに分割する場合のみ、
    そのチャンク境界に沿ってサブ分割する。

    Args:
        mesh: 対象メッシュ。
        labels: 現在の面ラベル(ジオメトリベースの分解結果)。
        face_areas: 面ごとの面積。
        image_labels: `project_labels_to_faces` の出力(面ごとの2D領域ラベル、
            -1は未取得)。
        seed: 乱数シード(未使用箇所もあるが将来の拡張のため保持)。
        significant_chunk_fraction: サブ分割を採用する条件となる、各チャンクの
            パーツ面積に対する比率の閾値。色領域誘導では `_IMG_SIGNIFICANT_CHUNK_FRACTION`
            (0.12)、LLMのbbox誘導(bboxが重なり合うため小パーツが相対的に
            小面積になりやすい)では `_IMG_SIGNIFICANT_CHUNK_FRACTION_LLM` (0.08)
            を渡す想定(呼び出し側で使い分ける)。
        source_image_label_of: 指定された場合、出力パーツラベルをキーに
            「そのパーツの由来となった `image_labels` の値」を書き込む
            (呼び出し側の辞書を破壊的に更新する、任意)。LLMパーツ名を
            サブ分割後のパーツラベルに対応付けるために使う。

    Returns:
        更新後のラベル(連番でなくてよい。呼び出し側で `_relabel_contiguous` する)。
    """
    labels = labels.copy()
    total_area = float(np.sum(face_areas)) or 1.0
    neighbors = _face_neighbors(mesh)
    next_id = int(labels.max()) + 1 if len(labels) else 0

    for part_label in list(np.unique(labels)):
        part_face_idx = np.where(labels == part_label)[0]
        part_area = float(np.sum(face_areas[part_face_idx]))
        if part_area / total_area <= _IMG_LARGE_PART_AREA_FRACTION:
            continue

        part_image_labels = image_labels[part_face_idx]
        valid = part_image_labels >= 0
        if not np.any(valid):
            continue

        # パーツ内で、画像ラベルごとの連結成分(双対グラフ上)を求め、
        # 各々の面積がパーツ面積の`_IMG_SIGNIFICANT_CHUNK_FRACTION`以上の
        # ものだけを「有意なチャンク」とする。
        local_img_labels = np.full(len(part_face_idx), -1, dtype=np.int64)
        local_img_labels[valid] = part_image_labels[valid]
        local_neighbors: list[list[int]] = [[] for _ in range(len(part_face_idx))]
        global_to_local = {int(g): i for i, g in enumerate(part_face_idx)}
        part_face_set = set(int(f) for f in part_face_idx)
        for f in part_face_idx:
            for nb in neighbors[f]:
                if nb in part_face_set:
                    local_neighbors[global_to_local[int(f)]].append(global_to_local[int(nb)])

        visited = np.zeros(len(part_face_idx), dtype=bool)
        chunks: list[tuple[int, list[int]]] = []  # (image_label, local face indices)
        for start in range(len(part_face_idx)):
            if visited[start] or local_img_labels[start] < 0:
                continue
            lbl = local_img_labels[start]
            stack = [start]
            visited[start] = True
            comp = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in local_neighbors[cur]:
                    if not visited[nb] and local_img_labels[nb] == lbl:
                        visited[nb] = True
                        stack.append(nb)
            chunks.append((int(lbl), comp))

        significant = [
            (lbl, comp)
            for lbl, comp in chunks
            if float(np.sum(face_areas[part_face_idx[comp]])) / part_area
            >= significant_chunk_fraction
        ]
        if len(significant) < 2:
            continue

        # 各有意チャンクの面積重心に最も近い面をシードにして、
        # 凹エッジ誘導Voronoi分割でパーツをサブ分割する。
        centers = mesh.triangles_center
        seed_faces = []
        seed_image_labels = []
        for lbl, comp in significant:
            comp_global = part_face_idx[comp]
            centroid = centers[comp_global].mean(axis=0)
            dists = np.linalg.norm(centers[comp_global] - centroid, axis=1)
            seed_faces.append(int(comp_global[int(np.argmin(dists))]))
            seed_image_labels.append(int(lbl))

        sub_labels = _split_subregion_from_seeds(mesh, part_face_idx, seed_faces)
        distinct = set(sub_labels.values())
        if len(distinct) < 2:
            continue

        # サブ分割の各チャンクの面積がいずれも有意水準を満たすことを確認する
        # (境界スナップにより一部が痩せ細って無意味な分割になるのを防ぐ)。
        sub_areas: dict[int, float] = {}
        for f, local_lbl in sub_labels.items():
            sub_areas[local_lbl] = sub_areas.get(local_lbl, 0.0) + float(face_areas[f])
        if any(a / part_area < significant_chunk_fraction for a in sub_areas.values()):
            continue

        # local_lbl(0..len(significant)-1、_split_subregion_from_seedsの
        # seed_faces順)ごとの由来image_labelを記録できるようにする。
        local_to_image_label = {i: seed_image_labels[i] for i in range(len(significant))}

        label_map = {0: int(part_label)}
        for local_lbl in sorted(distinct):
            if local_lbl == 0:
                continue
            label_map[local_lbl] = next_id
            next_id += 1

        if source_image_label_of is not None:
            for local_lbl, out_label in label_map.items():
                src = local_to_image_label.get(local_lbl)
                if src is not None:
                    source_image_label_of[out_label] = src

        for f, local_lbl in sub_labels.items():
            labels[f] = label_map[local_lbl]

    return labels


def _split_largest_until_hint(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    face_areas: np.ndarray,
    n_parts_hint: int,
    seed: int = 0,
    min_faces_to_split: int = 20,
) -> np.ndarray:
    labels = labels.copy()
    next_id = int(labels.max()) + 1 if len(labels) else 0
    max_attempts = max(0, n_parts_hint) * 4 + 4  # 分割不能な停滞を避ける安全弁

    for _ in range(max_attempts):
        if len(np.unique(labels)) >= n_parts_hint:
            break
        areas = {int(lbl): float(np.sum(face_areas[labels == lbl])) for lbl in np.unique(labels)}
        largest = max(areas.items(), key=lambda kv: kv[1])[0]
        face_idx = np.where(labels == largest)[0]
        if len(face_idx) < min_faces_to_split:
            break
        sub_labels = _split_subregion(mesh, face_idx, 2, seed=seed)
        if len(set(sub_labels.values())) < 2:
            break
        for f, local_lbl in sub_labels.items():
            labels[f] = largest if local_lbl == 0 else next_id
        next_id += 1

    return labels


def _merge_smallest_until_hint(
    labels: np.ndarray,
    neighbors: list[list[int]],
    face_areas: np.ndarray,
    n_parts_hint: int,
    max_iters: int = 2000,
) -> np.ndarray:
    labels = labels.copy()
    for _ in range(max_iters):
        unique_labels = np.unique(labels)
        if len(unique_labels) <= max(1, n_parts_hint):
            break
        areas = {int(lbl): float(np.sum(face_areas[labels == lbl])) for lbl in unique_labels}
        smallest = min(areas.items(), key=lambda kv: kv[1])[0]
        faces_of_label = np.where(labels == smallest)[0]
        neighbor_label_counts: dict[int, int] = {}
        for f in faces_of_label:
            for nb in neighbors[f]:
                nb_label = int(labels[nb])
                if nb_label != smallest:
                    neighbor_label_counts[nb_label] = neighbor_label_counts.get(nb_label, 0) + 1
        if not neighbor_label_counts:
            break
        new_label = max(neighbor_label_counts.items(), key=lambda kv: kv[1])[0]
        labels[faces_of_label] = new_label
    return labels


def _seed_name_groups(
    n_seeds: int, seed_names: Optional[list[Optional[str]]]
) -> tuple[np.ndarray, list[Optional[str]]]:
    """シード名リストからグループID配列とグループ名リストを作る。

    同じ(空でない)名前のシードは同一グループ(=同一パーツ)に統合される。
    名前が空/Noneのシードは1シード=1グループとして扱う(呼び出し側の
    アダプタが自動名 part_N を振る場合、各自動名はユニークなので
    結果的に同じ挙動になる)。グループIDは初出順の0始まり連番で、
    そのまま `decompose_parts` が返すパーツIDになる。

    Returns:
        (seed_groups, group_names):
            seed_groups: (n_seeds,) int64。各シードのグループID。
            group_names: グループIDでインデックスされる名前リスト
                (名前なしグループは None)。
    """
    seed_groups = np.zeros(n_seeds, dtype=np.int64)
    group_names: list[Optional[str]] = []
    name_to_group: dict[str, int] = {}
    for i in range(n_seeds):
        name = None
        if seed_names is not None and i < len(seed_names) and seed_names[i]:
            name = seed_names[i]
        if name is not None and name in name_to_group:
            seed_groups[i] = name_to_group[name]
        else:
            g = len(group_names)
            group_names.append(name)
            if name is not None:
                name_to_group[name] = g
            seed_groups[i] = g
    return seed_groups, group_names


def _decompose_parts_from_seeds(
    mesh: trimesh.Trimesh,
    seed_points: np.ndarray,
    seed_names: Optional[list[str]],
    propagate_opposite: bool = True,
    planar_boundaries: bool = True,
    manual_info_out: Optional[dict] = None,
) -> tuple[np.ndarray, list[dict]]:
    """`decompose_parts` の手動シード誘導経路(`guidance="manual"`)。

    同じ名前のシードは同一パーツ(名前グループ)に統合され、マルチソースの
    最短距離で扱われる(例: 「胴体」を正面と背面に1点ずつ→背面の割当が改善)。
    `propagate_opposite=True` の場合はさらに、各シードの反対側スキンに
    同グループの仮想シードを自動生成する(`labels_from_seeds` 参照)。
    `labels_from_seeds` の結果に対し、既存の後処理(連結性強制・極小パーツ
    吸収)を適用する。ただし、ユーザーが明示的にクリックした部位は面数が
    少なくても消滅させないよう、シード由来のラベルは `_absorb_tiny_parts`
    の吸収対象から保護する(`protected_labels`)。

    Args:
        propagate_opposite: 反対側スキンへの仮想シード自動生成の有効/無効。
        planar_boundaries: 境界の平面フィット正則化の有効/無効
            (`_planar_boundary_regularize` 参照)。
        manual_info_out: 指定された場合、`{"virtual_seeds": [{"x","y","z",
            "name"}...], "planar_fit": [...]}` を破壊的に書き込む
            (`pattern_parts.json` への記録用。名前は仮想シードの由来グループの
            部位名。planar_fit はペアごとの平面フィット適用結果)。

    Returns:
        (labels, stats): labels はパーツ(名前グループ)単位の0始まり連番。
        stats の各要素にはグループ名が `name` として付く(名前なしは None)。
    """
    seed_points = np.asarray(seed_points, dtype=np.float64)
    n_seeds = len(seed_points)

    seed_groups, group_names = _seed_name_groups(n_seeds, seed_names)
    n_groups = len(group_names)

    # 肉厚は labels_from_seeds(エッジ重みの肉厚項)と part_stats
    # (mean_thickness_mm)の両方で使うため、1回だけ計算して共有する。
    thickness = compute_local_thickness(mesh, seed=0)
    virtual_seeds_raw: list[dict] = []
    virtual_skips_raw: list[dict] = []
    planar_fit_raw: list[dict] = []
    labels = labels_from_seeds(
        mesh,
        seed_points,
        seed_groups=seed_groups,
        thickness=thickness,
        propagate_opposite=propagate_opposite,
        virtual_seeds_out=virtual_seeds_raw,
        virtual_skips_out=virtual_skips_raw,
        planar_boundaries=planar_boundaries,
        planar_fit_out=planar_fit_raw,
    )

    if manual_info_out is not None:
        manual_info_out["virtual_seeds"] = [
            {
                "x": vs["x"],
                "y": vs["y"],
                "z": vs["z"],
                "name": group_names[vs["group"]] if 0 <= vs["group"] < n_groups else None,
            }
            for vs in virtual_seeds_raw
        ]
        # 仮想シードを得られなかったグループごとに代表的な棄却理由を1つ記録する
        # (トライアルの診断効率のため)。優先順位: balance(均衡棄却)>
        # 各シードの棄却理由の最頻値。
        groups_with_virtual = {int(vs["group"]) for vs in virtual_seeds_raw}
        skips_summary: list[dict] = []
        for g in range(n_groups):
            if g in groups_with_virtual:
                continue
            reasons = [s["reason"] for s in virtual_skips_raw if int(s["group"]) == g]
            if not reasons:
                continue
            if "balance" in reasons:
                reason = "balance"
            else:
                values, counts = np.unique(reasons, return_counts=True)
                reason = str(values[np.argmax(counts)])
            skips_summary.append(
                {"name": group_names[g] if 0 <= g < n_groups else None, "reason": reason}
            )
        manual_info_out["virtual_seed_skips"] = skips_summary
        manual_info_out["planar_fit"] = planar_fit_raw

    neighbors = _face_neighbors(mesh)
    face_areas = mesh.area_faces
    face_centers = mesh.triangles_center

    # シード由来の全グループラベル(0..n_groups-1)を吸収対象から保護する。
    protected = set(range(n_groups))
    labels = _enforce_part_connectivity(labels, neighbors, face_centers, face_areas)
    labels = _absorb_tiny_parts(
        labels, neighbors, face_areas, face_centers, protected_labels=protected
    )
    labels = _enforce_part_connectivity(labels, neighbors, face_centers, face_areas)

    # 手動シードはラベル値=グループIDのまま維持したい(名前対応付けを
    # 単純にするため)。_enforce_part_connectivity は値そのものは変えない
    # (飛び地を既存ラベル値へ再割当するだけ)ので、ここでは_relabel_contiguousは
    # 呼ばず、グループIDがそのままパーツIDになる前提で名前を対応付ける。
    stats = part_stats(mesh, labels, thickness=thickness)

    for s in stats:
        name = None
        if 0 <= s["part_id"] < n_groups:
            name = group_names[s["part_id"]]
        s["name"] = name

    return labels, stats


# --------------------------------------------------------------------------
# 公開API: パーツ自動分解
# --------------------------------------------------------------------------
def decompose_parts(
    mesh: trimesh.Trimesh,
    n_parts_hint: int = 0,
    seed: int = 0,
    image_rgba: Optional[np.ndarray] = None,
    image_labels: Optional[np.ndarray] = None,
    significant_chunk_fraction: float = _IMG_SIGNIFICANT_CHUNK_FRACTION,
    part_names: Optional[dict[int, str]] = None,
    seed_points: Optional[np.ndarray] = None,
    seed_names: Optional[list[str]] = None,
    seed_propagate_opposite: bool = True,
    seed_planar_boundaries: bool = True,
    manual_info_out: Optional[dict] = None,
) -> tuple[np.ndarray, list[dict]]:
    """メッシュを部位単位のパーツへ自動分解する。

    Args:
        mesh: 対象メッシュ(前処理済み推奨。`prepare_mesh`参照)。
        n_parts_hint: 目標パーツ数のヒント(0=自動、2〜10を想定)。
            0の場合は局所肉厚クラスタリングの結果をそのまま用いる。
            正の値が指定された場合、現在のパーツ数がヒントに満たなければ
            最大パーツをくびれ誘導で分割し、超えていれば最小パーツから
            隣接統合してヒントに近づける。`seed_points` 指定時は無視される
            (手動シードの点数がそのままパーツ数になる)。
        seed: 乱数シード(サンプリング・farthest-point-seeding等)。
        image_rgba: (H, W, 4) uint8 ndarray(背景除去済み入力画像、任意)。
            指定された場合、`extract_image_regions` で2D色領域を抽出し、
            ジオメトリ分解結果の大パーツ(全面積30%超)を画像領域境界に
            沿ってサブ分割する(くびれの浅い頭・胴等の分離補助)。
            `image_labels` が同時に指定された場合はそちらを優先し、
            `extract_image_regions` の再計算はスキップする。
            `seed_points` 指定時は無視される。
        image_labels: `extract_image_regions` または `labels_from_bboxes` が
            返す (H, W) int64 ラベル画像を事前計算済みの場合に渡す(任意、
            呼び出し側でのキャッシュ用)。`labels_from_bboxes` の出力
            (LLMパーツ検出のbbox由来)を渡す場合、`part_names` も併せて
            渡すとサブ分割後のパーツに部位名を対応付けられる。
            `seed_points` 指定時は無視される。
        significant_chunk_fraction: 画像誘導サブ分割の有意チャンク閾値
            (`_image_guided_subdivide` 参照)。色領域誘導は既定の0.12を、
            LLMのbbox誘導(bbox同士が重なるため小パーツが相対的に小面積に
            なりやすい)は呼び出し側で0.08程度に緩めて渡す想定。
        part_names: `image_labels` の値(0始まりのラベル番号)をキーに、
            対応する部位名(文字列)を渡す辞書(任意、LLMパーツ検出向け)。
            指定された場合、画像誘導サブ分割で追加されたパーツ、および
            画像ラベルが優勢な既存パーツの `parts_meta` に `name` フィールド
            が追加される(対応する名前が無い場合は `None`)。
        seed_points: (N, 3) float配列(任意)。指定された場合、他の誘導
            (肉厚クラスタリング・`n_parts_hint`分割/統合・画像/LLM誘導)は
            すべてスキップし、`labels_from_seeds` によるユーザー指定シード
            誘導分解(`guidance="manual"` 相当)を行う。最優先の誘導方式。
        seed_names: `seed_points` と同じ長さ・順序の部位名リスト(任意)。
            **同じ(空でない)名前のシードは同一パーツに統合され**、その
            パーツの全シード面からのマルチソース最短距離で面が割り当てられる
            (例: 「胴体」を正面と背面に1点ずつ打つと背面の分離が改善する)。
            パーツID(=labelsの値)は名前の初出順の0始まり連番。名前が
            空文字列/Noneのシードは1シード=1パーツとして扱われ、そのパーツの
            `name` は `None` になる。
        seed_propagate_opposite: True(既定)の場合、各シードの反対側スキン
            (逆法線レイの first exit)に同グループの仮想シードを自動生成し、
            正面のみのシードでも背面の割当を改善する(`labels_from_seeds` の
            `propagate_opposite` 参照)。`seed_points` 指定時のみ有効。
        seed_planar_boundaries: True(既定)の場合、隣接パーツペアの境界へ
            平面をフィットし、曖昧帯(背面・肩の上など)を平面で再割当する
            (`_planar_boundary_regularize` 参照)。`seed_points` 指定時のみ有効。
        manual_info_out: 指定された場合、手動シード経路の付加情報
            (`virtual_seeds`: 生成された仮想シードの座標・部位名リスト、
            `planar_fit`: ペアごとの平面フィット適用結果)を
            破壊的に書き込む(任意)。`seed_points` 指定時のみ有効。

    Returns:
        (labels, parts_meta):
            labels: (F,) int64 配列。各面が属するパーツID(0始まり連番)。
            parts_meta: `part_stats`と同じ形式の辞書のリスト。`part_names`が
                指定された場合は各要素に `name` フィールドが追加される。
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros((0,), dtype=np.int64), []

    if seed_points is not None:
        return _decompose_parts_from_seeds(
            mesh,
            seed_points,
            seed_names,
            propagate_opposite=seed_propagate_opposite,
            planar_boundaries=seed_planar_boundaries,
            manual_info_out=manual_info_out,
        )

    if n_faces < 8:
        # 極小メッシュはパーツ分解の意味がないため単一パーツとして扱う
        labels = np.zeros(n_faces, dtype=np.int64)
        stats = part_stats(mesh, labels)
        if part_names is not None:
            for s in stats:
                s["name"] = None
        return labels, stats

    thickness = compute_local_thickness(mesh, seed=seed)

    log_thickness = np.log(np.clip(thickness, 1e-6, None))
    cluster_labels = np.zeros(n_faces, dtype=np.int64)
    try:
        centroids, candidate_labels = kmeans2(log_thickness.reshape(-1, 1), 2, seed=seed, minit="++")
        centroid_ratio = float(np.exp(np.max(centroids)) / max(np.exp(np.min(centroids)), 1e-9))
        # 太/細クラスタの重心比が閾値未満なら実質一様な肉厚とみなし、
        # 意味のある分離候補がないため単一クラスタのまま扱う。
        if centroid_ratio >= _THICKNESS_BIMODAL_RATIO_THRESHOLD:
            cluster_labels = candidate_labels
    except Exception:
        pass

    neighbors = _face_neighbors(mesh)
    components = _split_into_components(cluster_labels, neighbors)

    labels = np.zeros(n_faces, dtype=np.int64)
    for new_id, (_old_label, comp) in enumerate(components):
        for f in comp:
            labels[f] = new_id

    face_areas = mesh.area_faces
    face_centers = mesh.triangles_center

    labels = _absorb_tiny_parts(labels, neighbors, face_areas, face_centers)
    labels = _enforce_part_connectivity(labels, neighbors, face_centers, face_areas)
    labels = _absorb_tiny_parts(labels, neighbors, face_areas, face_centers)
    labels = _relabel_contiguous(labels)

    # --- 画像誘導サブ分割(image_rgba/image_labelsが与えられた場合のみ) -----
    if image_labels is None and image_rgba is not None:
        region_label_img, n_regions = extract_image_regions(image_rgba, seed=seed)
        image_labels = region_label_img if n_regions > 0 else None

    source_image_label_of: dict[int, int] = {}
    face_image_labels_for_naming: Optional[np.ndarray] = None
    if image_labels is not None:
        face_image_labels = project_labels_to_faces(mesh, image_labels)
        face_image_labels_for_naming = face_image_labels
        labels = _image_guided_subdivide(
            mesh,
            labels,
            face_areas,
            face_image_labels,
            seed=seed,
            significant_chunk_fraction=significant_chunk_fraction,
            source_image_label_of=source_image_label_of if part_names is not None else None,
        )
        labels = _enforce_part_connectivity(labels, neighbors, face_centers, face_areas)
        labels = _absorb_tiny_parts(labels, neighbors, face_areas, face_centers)
        labels = _relabel_contiguous(labels)

    # 名前解決用: この時点(画像誘導サブ分割直後)のラベルごとの由来
    # image_labelを、面単位のマップとして保持する(以降の hint 分割/統合・
    # 連結性強制でラベル番号が変わってもname対応を追跡できるようにするため、
    # ラベル番号ではなく面配列で持つ)。
    face_source_image_label = np.full(n_faces, -1, dtype=np.int64)
    if part_names is not None and face_image_labels_for_naming is not None:
        for lbl in np.unique(labels):
            lbl_int = int(lbl)
            face_idx_of_label = np.where(labels == lbl_int)[0]
            # そのラベルがサブ分割由来(source_image_label_ofに記録済み)なら
            # それを優先し、なければ多数決で画像ラベルを決める。
            if lbl_int in source_image_label_of:
                face_source_image_label[face_idx_of_label] = source_image_label_of[lbl_int]
            else:
                part_img_labels = face_image_labels_for_naming[face_idx_of_label]
                valid = part_img_labels[part_img_labels >= 0]
                if len(valid) > 0:
                    values, counts = np.unique(valid, return_counts=True)
                    face_source_image_label[face_idx_of_label] = int(values[np.argmax(counts)])

    if n_parts_hint and n_parts_hint > 0:
        current_n = len(np.unique(labels))
        if current_n < n_parts_hint:
            labels = _split_largest_until_hint(mesh, labels, face_areas, n_parts_hint, seed=seed)
        elif current_n > n_parts_hint:
            labels = _merge_smallest_until_hint(labels, neighbors, face_areas, n_parts_hint)
        labels = _relabel_contiguous(labels)

    labels = _enforce_part_connectivity(labels, neighbors, face_centers, face_areas)
    labels = _relabel_contiguous(labels)

    stats = part_stats(mesh, labels)

    if part_names is not None:
        for s in stats:
            part_id = s["part_id"]
            face_idx_of_part = np.where(labels == part_id)[0]
            src_labels = face_source_image_label[face_idx_of_part]
            valid = src_labels[src_labels >= 0]
            name = None
            if len(valid) > 0:
                values, counts = np.unique(valid, return_counts=True)
                # そのパーツの過半数の面が同一image_label由来の場合のみ名前を
                # 採用する(統合等で複数の由来が混ざったパーツには名前を
                # 付けない=不確かな名前より「名前なし」の方が安全なため)。
                majority_label = int(values[np.argmax(counts)])
                if counts.max() / len(valid) >= 0.5:
                    name = part_names.get(majority_label)
            s["name"] = name

    return labels, stats


# --------------------------------------------------------------------------
# パーツ統計
# --------------------------------------------------------------------------
def part_stats(
    mesh: trimesh.Trimesh, labels: np.ndarray, thickness: Optional[np.ndarray] = None
) -> list[dict]:
    """パーツごとの面数・面積・体積(キャップ後)・肉厚平均を返す。

    Args:
        thickness: (F,) 面ごとの局所肉厚(任意)。呼び出し側で
            `compute_local_thickness` を計算済みの場合に渡すと再計算を省ける
            (手動シード誘導経路が肉厚項のために計算済みのものを流用する)。
            None の場合は内部で遅延計算する(従来動作)。

    Returns:
        パーツIDでソートされた辞書のリスト:
        `{"part_id", "n_faces", "area_mm2", "volume_mm3", "mean_thickness_mm",
          "connected", "watertight_after_cap"}`
    """
    if len(labels) == 0:
        return []

    face_areas = mesh.area_faces
    neighbors = _face_neighbors(mesh)

    stats = []
    for part_id in sorted(int(x) for x in np.unique(labels)):
        face_idx = np.where(labels == part_id)[0]
        area = float(np.sum(face_areas[face_idx]))

        comps = _connected_subcomponents(face_idx, neighbors)
        if len(comps) <= 1:
            is_connected = True
        else:
            # 縮退三角形(面積ゼロ)由来の孤立断片は無視し、実質的な面積の
            # 99.5%以上が単一連結成分に収まっていれば「連結」とみなす
            # (メッシュ簡略化が稀に生成する面積ゼロの浮遊三角形は、パーツ
            # 分解のロジックでは修復不能かつ実害がないため)。
            comp_areas = sorted(
                (float(np.sum(face_areas[comp])) for comp in comps), reverse=True
            )
            is_connected = (comp_areas[0] / area) >= 0.995 if area > 0 else True

        volume_mm3 = 0.0
        watertight_after_cap = False
        try:
            closed_mesh, _cap_info = cap_part(mesh, face_idx)
            volume_mm3 = float(abs(closed_mesh.volume))
            watertight_after_cap = bool(closed_mesh.is_watertight)
        except Exception:
            pass

        if thickness is None:
            thickness = compute_local_thickness(mesh, seed=0)
        mean_thickness = float(np.mean(thickness[face_idx])) if len(face_idx) else 0.0

        stats.append(
            {
                "part_id": part_id,
                "n_faces": int(len(face_idx)),
                "area_mm2": area,
                "volume_mm3": volume_mm3,
                "mean_thickness_mm": mean_thickness,
                "connected": bool(is_connected),
                "watertight_after_cap": watertight_after_cap,
            }
        )
    return stats


def _connected_subcomponents(face_idx: np.ndarray, neighbors: list[list[int]]) -> list[list[int]]:
    face_set = set(int(f) for f in face_idx)
    visited: set[int] = set()
    comps: list[list[int]] = []
    for f in face_idx:
        f = int(f)
        if f in visited:
            continue
        stack = [f]
        visited.add(f)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in neighbors[cur]:
                if nb in face_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


# --------------------------------------------------------------------------
# 公開API: パーツのキャップ(切断面を閉じる)
# --------------------------------------------------------------------------
def _boundary_loops(sub: trimesh.Trimesh) -> list[list[int]]:
    """境界ループを頂点インデックス(sub基準)の順序付きリストのリストで返す。"""
    try:
        outline = sub.outline(process=False)
        if outline is not None and len(outline.entities) > 0:
            loops = []
            for entity in outline.entities:
                pts = list(entity.points)
                if len(pts) >= 2 and pts[0] == pts[-1]:
                    pts = pts[:-1]
                if len(pts) >= 3:
                    loops.append([int(p) for p in pts])
            return loops
    except Exception:
        pass
    return _boundary_loops_manual(sub)


def _boundary_loops_manual(sub: trimesh.Trimesh) -> list[list[int]]:
    """trimeshのoutlineが使えない場合の手動境界ループ復元
    (境界エッジ=1面にしか属さないエッジをたどってループ化)。"""
    if len(sub.faces) == 0:
        return []
    edges = sub.edges_sorted
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    if len(boundary_edges) == 0:
        return []

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in adjacency:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current = start
        prev: Optional[int] = None
        while True:
            candidates = [n for n in adjacency[current] if n != prev]
            nxt = None
            for n in candidates:
                if n == start and len(loop) > 2:
                    nxt = start
                    break
                if n not in visited:
                    nxt = n
                    break
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev = current
            current = nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def cap_part(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> tuple[trimesh.Trimesh, dict]:
    """パーツを部分メッシュとして切り出し、開いた境界ループを蓋で閉じる。

    蓋は各境界ループの頂点重心を新規頂点(ファンの要)としたファン三角形
    分割で作る(非平面な境界ループでも破綻しない: 重心は3D空間上の単純な
    平均点であり、平面性を仮定しない)。境界ループの巡回順序は
    `trimesh.outline`(またはフォールバックの手動境界エッジ追跡)が返す
    メッシュ本来の巻き方向に従うため、生成される蓋面の法線は追加の巻き方向
    調整なしに外向きになる(`fix_normals()`で最終防御もかける)。

    Args:
        mesh: 元メッシュ。
        face_indices: このパーツに属する面インデックス(mesh.faces基準)。

    Returns:
        (closed_mesh, cap_info):
            closed_mesh: 蓋で閉じた新しい `trimesh.Trimesh`。
            cap_info: `{"n_boundary_loops", "cap_face_mask" (closed_meshの
                面数分のbool配列、Trueが蓋=取付口の面), "is_watertight"}`。
    """
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if len(face_indices) == 0:
        empty = trimesh.Trimesh()
        return empty, {"n_boundary_loops": 0, "cap_face_mask": np.zeros(0, dtype=bool), "is_watertight": False}

    sub = mesh.submesh([face_indices], append=True, repair=False)
    if sub is None or len(sub.faces) == 0:
        empty = trimesh.Trimesh()
        return empty, {"n_boundary_loops": 0, "cap_face_mask": np.zeros(0, dtype=bool), "is_watertight": False}

    # 縮退三角形(面積ゼロ)はメッシュ簡略化等の副産物として稀に残り、
    # 非多様体エッジ(同一エッジを3面以上が共有)を生んでキャップ後の
    # watertight判定を偽陰性にする。キャップ前に除去しておく。
    degenerate = sub.area_faces < 1e-9
    if np.any(degenerate) and not np.all(degenerate):
        sub = sub.submesh([np.where(~degenerate)[0]], append=True, repair=False)
        if sub is None or len(sub.faces) == 0:
            empty = trimesh.Trimesh()
            return empty, {"n_boundary_loops": 0, "cap_face_mask": np.zeros(0, dtype=bool), "is_watertight": False}

    loops = _boundary_loops(sub)

    all_vertices = sub.vertices.copy()
    all_faces = sub.faces.copy()
    n_original_faces = len(all_faces)

    for loop in loops:
        if len(loop) < 3:
            continue
        loop_pts = sub.vertices[loop]
        centroid = loop_pts.mean(axis=0)
        new_vertex_idx = len(all_vertices)
        all_vertices = np.vstack([all_vertices, centroid.reshape(1, 3)])
        n = len(loop)
        new_faces = np.array(
            [[loop[i], loop[(i + 1) % n], new_vertex_idx] for i in range(n)], dtype=np.int64
        )
        all_faces = np.vstack([all_faces, new_faces])

    closed = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)

    cap_face_mask = np.zeros(len(all_faces), dtype=bool)
    cap_face_mask[n_original_faces:] = True

    try:
        closed.fix_normals()
    except Exception:
        pass

    cap_info = {
        "n_boundary_loops": len(loops),
        "cap_face_mask": cap_face_mask,
        "is_watertight": bool(closed.is_watertight),
    }
    return closed, cap_info


# --------------------------------------------------------------------------
# 取付口(穴)を取り囲んだパネルの円盤位相修復
# --------------------------------------------------------------------------
def _find_split_target_loop_faces(
    sub: trimesh.Trimesh,
    labels: np.ndarray,
    lbl: int,
    neighbors: list[list[int]],
) -> Optional[set[int]]:
    """ラベル`lbl`のパネルが円盤位相でない(境界ループ2本以上)場合、
    分割の基準にする境界ループに接する面(パネルローカルインデックス)の
    集合を返す。円盤位相なら None。

    基準ループの選択:
        1. シーム面を1つも含まないループ(=パネル内部に取り囲まれた穴、
           取付口など)があればそのうち最小のもの。
        2. なければ(円筒状パネル: 両端がシームに接する場合など)
           最小のループ。
    どちらの場合も「そのループ上の互いに遠い2面をシードに2分割」すると
    ループが2つのサブパネルに分かれ、円盤位相に近づく(穴なら穴が外周に
    開通し、円筒なら縦割りで半殻2枚=円盤になる)。
    """
    face_idx = np.where(labels == lbl)[0]
    if len(face_idx) == 0:
        return None
    panel_sub = sub.submesh([face_idx], append=True, repair=False)
    if panel_sub is None or len(panel_sub.faces) == 0:
        return None
    loops = _boundary_loops(panel_sub)
    if len(loops) <= 1:
        return None

    global_to_local = {int(g): i for i, g in enumerate(face_idx)}
    seam_faces_set = {
        global_to_local[int(f)]
        for f in face_idx
        if any(labels[nb] != lbl for nb in neighbors[f])
    }

    loop_id_of_vertex: dict[int, int] = {}
    for li, lp in enumerate(loops):
        for v in lp:
            loop_id_of_vertex[int(v)] = li
    faces_touching: list[set[int]] = [set() for _ in loops]
    for fi_local, tri in enumerate(panel_sub.faces):
        for v in tri:
            li = loop_id_of_vertex.get(int(v))
            if li is not None:
                faces_touching[li].add(fi_local)

    # シーム面を1つも含まないループ = パネル内部に取り囲まれた穴(優先)
    enclosed = [li for li in range(len(loops)) if not (faces_touching[li] & seam_faces_set)]
    candidates = enclosed if enclosed else list(range(len(loops)))
    target_li = min(candidates, key=lambda li: len(loops[li]))
    return faces_touching[target_li] or None


def _split_non_disk_panels(
    sub: trimesh.Trimesh, labels: np.ndarray, max_rounds: int = 10
) -> np.ndarray:
    """円盤位相でないパネル(境界ループ2本以上)を「基準ループを跨ぐ
    2シードVoronoi分割」で2分割して修復する。

    シードは基準ループ(`_find_split_target_loop_faces` 参照)に接する面から
    2つ、パネル内双対グラフ上で互いに最も遠いペアを選ぶ。多始点BFSで各面を
    近い方のシードに割り当てると基準ループが2つのサブパネルに分かれる:
    - 取り囲まれた穴(取付口)の場合: 穴が両サブパネルの外周の一部になる。
    - 円筒状パネル(両端がシーム)の場合: 縦割りで半殻2枚=円盤になる。
    対象パネル以外は一切変更しないため、修復が他パネルへ波及しない
    (パネル総数は1回の修復につき1増える)。
    """
    labels = labels.copy()
    neighbors = _face_neighbors(sub)
    next_id = int(labels.max()) + 1 if len(labels) else 0

    for _ in range(max_rounds):
        fixed_any = False
        for lbl in list(np.unique(labels)):
            hole_faces = _find_split_target_loop_faces(sub, labels, int(lbl), neighbors)
            if not hole_faces:
                continue

            face_idx = np.where(labels == lbl)[0]
            if len(face_idx) < 8:
                continue

            # パネル内ローカル双対グラフ
            panel_sub = sub.submesh([face_idx], append=True, repair=False)
            local_neighbors: list[list[int]] = [[] for _ in range(len(face_idx))]
            for a, b in panel_sub.face_adjacency:
                local_neighbors[a].append(int(b))
                local_neighbors[b].append(int(a))

            def _bfs_dist(sources: set[int]) -> np.ndarray:
                dist = np.full(len(face_idx), -1, dtype=np.int64)
                queue = list(sources)
                for s in sources:
                    dist[s] = 0
                head = 0
                while head < len(queue):
                    cur = queue[head]
                    head += 1
                    for nb in local_neighbors[cur]:
                        if dist[nb] < 0:
                            dist[nb] = dist[cur] + 1
                            queue.append(nb)
                return dist

            # シード1: 穴に接する任意の面。シード2: 穴に接する面のうち
            # シード1から(パネル内で)最も遠い面 → 2シードの境界が穴を跨ぐ。
            s1 = min(hole_faces)
            dist_from_s1 = _bfs_dist({s1})
            hole_list = sorted(hole_faces)
            s2 = max(hole_list, key=lambda f: dist_from_s1[f] if dist_from_s1[f] >= 0 else -1)
            if s2 == s1 or dist_from_s1[s2] <= 0:
                continue

            dist1 = dist_from_s1
            dist2 = _bfs_dist({s2})
            reach1 = dist1 >= 0
            reach2 = dist2 >= 0
            d1 = np.where(reach1, dist1, np.iinfo(np.int64).max)
            d2 = np.where(reach2, dist2, np.iinfo(np.int64).max)
            assign_to_s2 = d2 < d1

            if not np.any(assign_to_s2) or np.all(assign_to_s2):
                continue

            labels[face_idx[assign_to_s2]] = next_id
            next_id += 1
            fixed_any = True

        if not fixed_any:
            break

    # 分割の副作用(飛び地)を隣接ラベルへ寄せて連結性を回復する
    labels = _enforce_part_connectivity(labels, neighbors, sub.triangles_center, sub.area_faces)
    return labels


# --------------------------------------------------------------------------
# 公開API: パーツ単位のパネル分割 (2段階構成の2段目、segment.pyを再利用)
# --------------------------------------------------------------------------
def segment_part_panels(
    closed_mesh: trimesh.Trimesh,
    cap_face_mask: np.ndarray,
    max_panels: int = 4,
    vertex_colors: Optional[np.ndarray] = None,
    use_colors: bool = True,
    seed: int = 0,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """蓋済みパーツメッシュを、蓋(取付口)を除いた表面についてパネル分割する。

    蓋の面はパネルに含めない(=蓋の縁がそのままパネル境界の一部になり、
    SVG出力で「取付口」として扱える)。取付口(=表面に空いた穴)が複数ある
    パーツでは、パネルが穴を完全に取り囲んで円盤位相を失いやすいため、
    パネル数を増やしながら全パネルが円盤位相になるまでリトライする
    (上限 `max_panels`。それでも解消しない場合は最良の分割を返し、
    円盤位相でないパネルは下流の `flatten_panel` が failed として報告する)。

    Args:
        closed_mesh: `cap_part` が返した蓋済みメッシュ。
        cap_face_mask: `cap_part` の cap_info["cap_face_mask"]。
        max_panels: このパーツのパネル数上限(2〜6を想定)。
        vertex_colors: closed_mesh の頂点数と同数の頂点カラー(0-255、任意)。
        use_colors: 色境界誘導を有効にするか。
        seed: 乱数シード。

    Returns:
        (sub_mesh, panel_labels, rim_coords):
            sub_mesh: 蓋を除いた部分メッシュ(パネル分割・平坦化の対象)。
            panel_labels: (F_sub,) int64。sub_meshの面ごとのパネルID。
            rim_coords: (R,3) float64。取付口の縁(蓋と表面の共有頂点)の
                3D座標。パネル境界点との照合で取付口区間の判定に使う。
    """
    cap_face_mask = np.asarray(cap_face_mask, dtype=bool)
    n_openings = 0

    cap_face_idx = np.where(cap_face_mask)[0]
    non_cap_face_idx = np.where(~cap_face_mask)[0]

    if len(non_cap_face_idx) == 0:
        empty = trimesh.Trimesh()
        return empty, np.zeros((0,), dtype=np.int64), np.zeros((0, 3))

    # 取付口の縁 = 蓋の面と表面の面が共有する頂点
    if len(cap_face_idx) > 0:
        cap_verts = set(closed_mesh.faces[cap_face_idx].ravel().tolist())
        non_cap_verts = set(closed_mesh.faces[non_cap_face_idx].ravel().tolist())
        rim_vert_idx = sorted(cap_verts & non_cap_verts)
        rim_coords = np.asarray(closed_mesh.vertices[rim_vert_idx], dtype=np.float64)
    else:
        rim_coords = np.zeros((0, 3))

    sub = closed_mesh.submesh([non_cap_face_idx], append=True, repair=False)
    if sub is None or len(sub.faces) == 0:
        empty = trimesh.Trimesh()
        return empty, np.zeros((0,), dtype=np.int64), rim_coords

    # 頂点カラーの引き継ぎ(closed_mesh頂点 → sub頂点は座標一致のため最近傍)
    sub_colors = None
    if use_colors and vertex_colors is not None and len(vertex_colors) == len(closed_mesh.vertices):
        try:
            tree = cKDTree(closed_mesh.vertices)
            _, nn_idx = tree.query(sub.vertices, k=1)
            sub_colors = np.asarray(vertex_colors)[nn_idx]
        except Exception:
            sub_colors = None

    n_openings = len(_boundary_loops(sub))

    # パネル数の自動決定(SPEC.md: パーツ表面積とくびれの複雑さから2〜4目安、
    # 単純な腕・脚は2枚)。取付口が多いほど円盤位相の確保に多くのパネルが
    # 要るため、取付口数に応じて増やす。
    max_panels = max(2, int(max_panels))
    n_panels = min(max_panels, max(2, 2 + max(0, n_openings - 1)))

    # segment.py を遅延import(循環importをモジュール読み込み時に発生させない)
    from .segment import panel_stats as _panel_stats
    from .segment import segment_panels as _segment_panels

    best_labels: Optional[np.ndarray] = None
    best_n_disk = -1
    attempt_seed = seed
    while True:
        labels = _segment_panels(
            sub, n_panels=n_panels, vertex_colors=sub_colors, use_colors=use_colors, seed=attempt_seed
        )
        # 円盤位相でないパネル(穴の取り囲み・円筒状)を2分割で修復する
        # (この修復はパネル数を増やしうるため、実パネル数はmax_panelsを
        # 超えることがある。円盤位相=平坦化可能性を優先する)
        labels = _split_non_disk_panels(sub, labels)
        stats = _panel_stats(sub, labels)
        n_disk = sum(1 for s in stats if s["disk_topology"])
        if n_disk > best_n_disk:
            best_n_disk = n_disk
            best_labels = labels
        if n_disk == len(stats):
            break
        if n_panels >= max_panels:
            break
        n_panels += 1
        attempt_seed += 1

    assert best_labels is not None
    return sub, best_labels, rim_coords
