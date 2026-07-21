"""パーツ自動分解 `server/pattern/parts.py` の単体テスト (SPEC.md §3.12「2段階構成 (4c)」)。

合成「雪だるま」形状(球2つ+円柱の腕2本・脚2本)で検証する。
trimesh.boolean.union はエンジン(manifold3d等)非導入環境で使えないため、
重ねた複合メッシュを voxelize → fill → marching_cubes で1つの
watertightメッシュに結合する(タスク指示のフォールバック方式)。

純粋モジュールの方針(DEVELOPMENT_POLICY.md §3.5)に従い、TestClientを
使わない純粋関数テストとする。
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from server.pattern import (
    cap_part,
    compute_local_thickness,
    decompose_parts,
    extract_image_regions,
    flatten_panel,
    labels_from_bboxes,
    labels_from_seeds,
    segment_part_panels,
)
from server.pattern.parts import part_stats, project_labels_to_faces


# --------------------------------------------------------------------------
# 合成フィクスチャ
# --------------------------------------------------------------------------
def make_snowman_mesh(pitch: float = 1.4, target_faces: int = 15000) -> trimesh.Trimesh:
    """雪だるま: 胴(球r28)+頭(球r16、首のくびれあり)+腕2本+脚2本。"""
    body = trimesh.creation.icosphere(subdivisions=3, radius=28.0)
    body.apply_translation([0, 0, 28])
    head = trimesh.creation.icosphere(subdivisions=3, radius=16.0)
    head.apply_translation([0, 0, 66])

    def limb(radius, length, translation, rotation_axis, rotation_angle):
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
        if rotation_angle:
            rot = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
            cyl.apply_transform(rot)
        cyl.apply_translation(translation)
        return cyl

    arm_l = limb(6.0, 40.0, [-36, 0, 38], [0, 1, 0], np.radians(90))
    arm_r = limb(6.0, 40.0, [36, 0, 38], [0, 1, 0], np.radians(90))
    leg_l = limb(8.0, 35.0, [-14, 0, -14], [1, 0, 0], 0)
    leg_r = limb(8.0, 35.0, [14, 0, -14], [1, 0, 0], 0)

    concat = trimesh.util.concatenate([body, head, arm_l, arm_r, leg_l, leg_r])
    vox = concat.voxelized(pitch=pitch).fill()
    remeshed = vox.marching_cubes
    # marching_cubes はボクセルインデックス座標系で返るため、元のワールド
    # 座標系(mm)へ戻す(テストのプローブ座標をワールド座標で書けるように)。
    remeshed.apply_transform(vox.transform)
    assert remeshed.is_watertight
    if len(remeshed.faces) > target_faces:
        remeshed = remeshed.simplify_quadric_decimation(face_count=target_faces)
    return remeshed


@pytest.fixture(scope="module")
def snowman():
    return make_snowman_mesh()


@pytest.fixture(scope="module")
def snowman_decomposition(snowman):
    """自動分解の結果(重いので module 内で1回だけ実行して使い回す)。"""
    labels, stats = decompose_parts(snowman, n_parts_hint=0, seed=0)
    return snowman, labels, stats


# --------------------------------------------------------------------------
# 局所肉厚 (SDF風レイキャスト)
# --------------------------------------------------------------------------
def test_local_thickness_sphere_matches_diameter():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    thickness = compute_local_thickness(sphere, seed=0)
    assert thickness.shape == (len(sphere.faces),)
    # 球の内向きレイの対面距離は直径(=20)近傍になる(コーン内レイのため
    # やや短めも許容)
    assert 15.0 < float(np.median(thickness)) <= 20.5


def test_local_thickness_distinguishes_thin_and_thick(snowman):
    thickness = compute_local_thickness(snowman, seed=0)
    centers = snowman.triangles_center
    # 腕(x=±36付近の細い円柱、直径12)は胴(直径56)よりはっきり薄い。
    # 胴のプローブは腹の表面(y≈-28, z≈28)付近の面を選ぶ。
    arm_faces = np.where(np.abs(centers[:, 0]) > 30)[0]
    body_faces = np.where(
        (np.abs(centers[:, 0]) < 10)
        & (centers[:, 1] < -20)
        & (np.abs(centers[:, 2] - 28) < 10)
    )[0]
    assert len(arm_faces) > 0 and len(body_faces) > 0
    assert np.median(thickness[arm_faces]) < 0.6 * np.median(thickness[body_faces])


# --------------------------------------------------------------------------
# 自動分解(雪だるま)
# --------------------------------------------------------------------------
def test_snowman_auto_decomposes_into_4_to_6_parts(snowman_decomposition):
    _mesh, labels, stats = snowman_decomposition
    n_parts = len(np.unique(labels))
    assert 4 <= n_parts <= 6
    assert len(stats) == n_parts
    assert sorted(s["part_id"] for s in stats) == list(range(n_parts))


def test_snowman_arms_separate_from_body(snowman_decomposition):
    mesh, labels, _stats = snowman_decomposition
    centers = mesh.triangles_center

    def label_nearest(point):
        idx = int(np.argmin(np.linalg.norm(centers - np.asarray(point), axis=1)))
        return int(labels[idx])

    body_label = label_nearest([0.0, -28.0, 28.0])   # 胴の腹側表面
    arm_l_label = label_nearest([-53.0, 0.0, 38.0])  # 左腕の先端
    arm_r_label = label_nearest([53.0, 0.0, 38.0])   # 右腕の先端

    assert arm_l_label != body_label
    assert arm_r_label != body_label
    assert arm_l_label != arm_r_label  # 左右の腕は別パーツ


def test_snowman_parts_connected_and_watertight_after_cap(snowman_decomposition):
    _mesh, _labels, stats = snowman_decomposition
    for s in stats:
        assert s["connected"], f"part {s['part_id']} is not connected"
        assert s["watertight_after_cap"], f"part {s['part_id']} cap failed"
        assert s["n_faces"] > 0
        assert s["area_mm2"] > 0
        assert s["volume_mm3"] > 0
        assert s["mean_thickness_mm"] > 0


def test_snowman_no_tiny_parts(snowman_decomposition):
    _mesh, _labels, stats = snowman_decomposition
    total_area = sum(s["area_mm2"] for s in stats)
    for s in stats:
        assert s["area_mm2"] / total_area >= 0.02


def test_n_parts_hint_is_respected(snowman):
    for hint in (2, 6):
        labels, stats = decompose_parts(snowman, n_parts_hint=hint, seed=0)
        assert len(np.unique(labels)) == hint
        assert len(stats) == hint


def test_uniform_sphere_stays_single_part():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    labels, stats = decompose_parts(sphere, n_parts_hint=0, seed=0)
    assert len(np.unique(labels)) == 1
    assert stats[0]["watertight_after_cap"]


def test_labels_cover_all_faces(snowman_decomposition):
    mesh, labels, _stats = snowman_decomposition
    assert labels.shape == (len(mesh.faces),)
    assert labels.min() >= 0


# --------------------------------------------------------------------------
# cap_part(切断面の蓋)
# --------------------------------------------------------------------------
def test_cap_part_closes_open_hemisphere():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    upper_faces = np.where(sphere.triangles_center[:, 2] > 0)[0]

    closed, cap_info = cap_part(sphere, upper_faces)

    assert cap_info["n_boundary_loops"] == 1
    assert cap_info["is_watertight"]
    assert closed.is_watertight
    # 取付口フラグ面が存在し、蓋以外の面は元の面数と一致する
    cap_mask = cap_info["cap_face_mask"]
    assert cap_mask.sum() > 0
    assert (~cap_mask).sum() == len(upper_faces)
    # 半球+蓋の体積は解析値 (2/3)πr³ に近い
    expected = (2.0 / 3.0) * np.pi * 1000.0
    assert abs(closed.volume - expected) / expected < 0.1


def test_cap_part_closed_input_needs_no_cap():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    all_faces = np.arange(len(sphere.faces))
    closed, cap_info = cap_part(sphere, all_faces)
    assert cap_info["n_boundary_loops"] == 0
    assert cap_info["cap_face_mask"].sum() == 0
    assert closed.is_watertight


def test_cap_part_cylinder_segment_two_loops():
    cyl = trimesh.creation.cylinder(radius=5.0, height=40.0, sections=32)
    mid_faces = np.where(np.abs(cyl.triangles_center[:, 2]) < 10.0)[0]
    closed, cap_info = cap_part(cyl, mid_faces)
    assert cap_info["n_boundary_loops"] == 2
    assert closed.is_watertight


# --------------------------------------------------------------------------
# パーツ単位のパネル分割 (2段階構成の2段目)
# --------------------------------------------------------------------------
def test_segment_part_panels_limb_two_disk_panels(snowman_decomposition):
    mesh, labels, stats = snowman_decomposition
    # 最小面積のパーツ=腕か脚(単純な円柱状)を対象にする
    limb = min(stats, key=lambda s: s["area_mm2"])
    face_idx = np.where(labels == limb["part_id"])[0]

    closed, cap_info = cap_part(mesh, face_idx)
    sub, panel_labels, rim_coords = segment_part_panels(
        closed, cap_info["cap_face_mask"], max_panels=4, use_colors=False, seed=0
    )

    panel_ids = np.unique(panel_labels)
    # 単純な腕・脚は2枚目安(SPEC.md)。円盤位相修復で若干増えることは許容。
    assert 2 <= len(panel_ids) <= 4
    assert len(rim_coords) > 0  # 取付口の縁がある

    n_ok = 0
    for pid in panel_ids:
        result = flatten_panel(sub, np.where(panel_labels == pid)[0], n_arap_iterations=5)
        if not result.get("flatten_failed"):
            n_ok += 1
    assert n_ok == len(panel_ids)  # 全パネルが平坦化可能(=円盤位相)


def test_segment_part_panels_attachment_mask_on_rim(snowman_decomposition):
    """パネル境界のうち取付口(切断面のリム)に一致する区間が検出できる。"""
    from scipy.spatial import cKDTree

    mesh, labels, stats = snowman_decomposition
    limb = min(stats, key=lambda s: s["area_mm2"])
    face_idx = np.where(labels == limb["part_id"])[0]

    closed, cap_info = cap_part(mesh, face_idx)
    sub, panel_labels, rim_coords = segment_part_panels(
        closed, cap_info["cap_face_mask"], max_panels=4, use_colors=False, seed=0
    )
    rim_tree = cKDTree(rim_coords)

    n_panels_with_attachment = 0
    for pid in np.unique(panel_labels):
        result = flatten_panel(sub, np.where(panel_labels == pid)[0], n_arap_iterations=5)
        if result.get("flatten_failed"):
            continue
        loop3d = result["vertices_3d"][result["boundary_loop_indices"]]
        dists, _ = rim_tree.query(loop3d, k=1)
        if np.any(dists <= 0.01):
            n_panels_with_attachment += 1
    # 取付口はどこかのパネル境界に必ず現れる
    assert n_panels_with_attachment >= 1


def test_part_stats_shape(snowman_decomposition):
    mesh, labels, _ = snowman_decomposition
    stats = part_stats(mesh, labels)
    required = {
        "part_id",
        "n_faces",
        "area_mm2",
        "volume_mm3",
        "mean_thickness_mm",
        "connected",
        "watertight_after_cap",
    }
    for s in stats:
        assert required <= set(s.keys())


# --------------------------------------------------------------------------
# パーツグループ化SVG(取付口ラベル)
# --------------------------------------------------------------------------
def test_pattern_svg_with_parts_and_attachment_openings(snowman_decomposition):
    """雪だるま全体で2段階フローを通し、SVGに部位グループ・凡例・
    取付口ラベルが含まれることを検証する。"""
    import xml.etree.ElementTree as ET

    from scipy.spatial import cKDTree

    from server.pattern import build_pattern_svg
    from server.pattern.preview import PALETTE_HEX

    mesh, labels, stats = snowman_decomposition
    panels_2d = []
    global_id = 0
    for s in stats:
        part_id = s["part_id"]
        face_idx = np.where(labels == part_id)[0]
        closed, cap_info = cap_part(mesh, face_idx)
        sub, panel_labels, rim_coords = segment_part_panels(
            closed, cap_info["cap_face_mask"], max_panels=4, use_colors=False, seed=0
        )
        rim_tree = cKDTree(rim_coords) if len(rim_coords) else None
        for panel_no, pid in enumerate(np.unique(panel_labels), start=1):
            result = flatten_panel(sub, np.where(panel_labels == pid)[0], n_arap_iterations=5)
            if result.get("flatten_failed"):
                continue
            if rim_tree is not None:
                loop3d = result["vertices_3d"][result["boundary_loop_indices"]]
                dists, _ = rim_tree.query(loop3d, k=1)
                result["attachment_mask"] = dists <= 0.01
            result["panel_id"] = global_id
            result["panel_no"] = panel_no
            result["part_id"] = part_id
            result["part_label"] = f"部位{part_id + 1}"
            result["part_color_hex"] = PALETTE_HEX[part_id % len(PALETTE_HEX)]
            panels_2d.append(result)
            global_id += 1

    svg_text = build_pattern_svg(panels_2d, seam_allowance_mm=7.0, model_name="snowman")

    root = ET.fromstring(svg_text)  # 整形式XMLであること
    assert root.tag.endswith("svg")
    assert root.attrib["width"].endswith("mm")

    assert "部位1-P1" in svg_text          # 「部位N-PM」形式のパネルラベル
    assert 'data-part-id="' in svg_text    # パーツごとのグループ化
    assert "パーツ対応表" in svg_text       # ビューア色との対応凡例
    assert "取付口" in svg_text            # 取付口の太線+ラベル
    assert 'class="attachment-opening"' in svg_text


# --------------------------------------------------------------------------
# 画像誘導分解 (くびれの浅い頭・胴等をパーツ分けするための補助)
# --------------------------------------------------------------------------
def make_waistless_capsule(radius: float = 15.0, height: float = 80.0) -> trimesh.Trimesh:
    """くびれのない縦長カプセル(頭と胴の統合を模した合成形状)。

    半球キャップ(半径15)+円柱胴(z=15..95相当)。円柱区間は面積が
    高さに対して線形に近い(半球キャップ部は極に近づくほど圧縮される)ため、
    テストの色境界は円柱区間内(高さ比0.4〜0.6)に置くことで、色境界の
    高さ位置と面積比の対応を素直に検証できるようにする。
    """
    capsule = trimesh.creation.capsule(radius=radius, height=height, count=[32, 32])
    capsule.apply_translation([0.0, 0.0, -capsule.bounds[0][2]])  # z>=0に接地
    return capsule


def make_two_region_vertical_image(
    height_fraction: float = 0.45, h: int = 256, w: int = 128
) -> np.ndarray:
    """上`height_fraction`がクリーム色、残りが紺色の縦2領域画像(RGBA、全域不透明)。

    colorproc.py と同一規約(v=0が画像上端、v=1が下端)に従い、上側
    (v小)がクリーム色になるようにする(メッシュのZ大きい側=頭頂に対応)。
    """
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    split_row = int(h * height_fraction)
    img[:split_row, :, :3] = [255, 253, 208]  # クリーム
    img[split_row:, :, :3] = [0, 0, 128]      # 紺
    return img


def test_extract_image_regions_four_color_blocks():
    h, w = 40, 40
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:20, :20] = [255, 0, 0, 255]
    img[:20, 20:] = [0, 255, 0, 255]
    img[20:, :20] = [0, 0, 255, 255]
    img[20:, 20:] = [255, 255, 0, 255]

    label_img, n_regions = extract_image_regions(img, max_regions=8)
    assert n_regions == 4
    assert label_img.shape == (h, w)
    assert set(np.unique(label_img).tolist()) == {0, 1, 2, 3}


def test_extract_image_regions_transparent_pixels_are_minus_one():
    h, w = 20, 20
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :10, :3] = [200, 50, 50]
    img[:, :10, 3] = 255
    # 右半分は透明(背景)のまま

    label_img, n_regions = extract_image_regions(img, max_regions=8)
    assert n_regions >= 1
    assert np.all(label_img[:, 10:] == -1)
    assert np.all(label_img[:, :10] >= 0)


def test_extract_image_regions_merges_tiny_regions():
    """全画素の1%未満の孤立した小領域は大領域へマージされ、領域数に
    数えられないことを検証する。"""
    h, w = 100, 100
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    img[:, :, :3] = [0, 0, 128]  # 大部分は紺
    # 1x1の孤立ノイズ画素(全体の0.01%未満)を1点だけ別色にする
    img[5, 5, :3] = [255, 0, 0]

    label_img, n_regions = extract_image_regions(img, max_regions=8)
    # 孤立ノイズは大領域(紺)へ吸収され、領域数は1のまま
    assert n_regions == 1


def test_project_labels_to_faces_matches_color_regions():
    """カプセルへ上下2色画像を投影すると、上部の面が上側ラベル、
    下部の面が下側ラベルになる(色境界誘導の前提となる投影の正しさ)。"""
    capsule = make_waistless_capsule()
    img = make_two_region_vertical_image(height_fraction=0.45)
    label_img, n_regions = extract_image_regions(img, max_regions=8)
    assert n_regions == 2

    face_labels = project_labels_to_faces(capsule, label_img)
    assert face_labels.shape == (len(capsule.faces),)
    assert set(np.unique(face_labels).tolist()) == {0, 1}

    centers = capsule.triangles_center
    top_label = face_labels[np.argmax(centers[:, 2])]
    bottom_label = face_labels[np.argmin(centers[:, 2])]
    assert top_label != bottom_label


def test_waistless_capsule_image_guided_splits_into_two_parts():
    """くびれのないカプセル(ジオメトリだけでは1パーツのまま)が、
    画像誘導ON時のみ色境界に沿って2パーツへ分割されることを検証する。
    境界のZ位置は画像の色境界(高さ比0.45相当のZ)に対して
    メッシュ高さの±15%以内に収まることを確認する。
    """
    capsule = make_waistless_capsule()
    img = make_two_region_vertical_image(height_fraction=0.45)

    # OFF: 画像を渡さない場合は従来通り単一パーツのまま(回帰なし)
    labels_off, stats_off = decompose_parts(capsule, n_parts_hint=0, seed=0)
    assert len(stats_off) == 1

    # ON: 画像誘導で2パーツに分割される
    labels_on, stats_on = decompose_parts(capsule, n_parts_hint=0, seed=0, image_rgba=img)
    assert len(stats_on) == 2

    centers = capsule.triangles_center
    zmin, zmax = capsule.bounds[0][2], capsule.bounds[1][2]
    height = zmax - zmin

    part_ids = sorted(np.unique(labels_on).tolist())
    mean_z = {
        pid: float(np.mean(centers[labels_on == pid][:, 2])) for pid in part_ids
    }
    upper_part = max(mean_z, key=lambda pid: mean_z[pid])
    lower_part = min(mean_z, key=lambda pid: mean_z[pid])
    assert upper_part != lower_part

    upper_z = centers[labels_on == upper_part][:, 2]
    lower_z = centers[labels_on == lower_part][:, 2]
    boundary_z = (upper_z.min() + lower_z.max()) / 2.0

    # v=height_fraction(0.45)の色境界 -> z_norm = 1 - 0.45 = 0.55
    expected_boundary_z = zmin + 0.55 * height
    assert abs(boundary_z - expected_boundary_z) <= 0.15 * height

    # 各パーツが単一連結成分・キャップ後watertightであること(既存の
    # 後処理が画像誘導サブ分割後も正しく適用されていることの確認)。
    for s in stats_on:
        assert s["connected"]
        assert s["watertight_after_cap"]


def test_decompose_parts_image_rgba_none_matches_legacy_behavior(snowman):
    """image_rgba=None(既定)の場合、従来のジオメトリのみの分解と
    完全に同じ結果になる(回帰なし)。"""
    labels_legacy, stats_legacy = decompose_parts(snowman, n_parts_hint=0, seed=0)
    labels_explicit_none, stats_explicit_none = decompose_parts(
        snowman, n_parts_hint=0, seed=0, image_rgba=None
    )
    assert np.array_equal(labels_legacy, labels_explicit_none)
    assert len(stats_legacy) == len(stats_explicit_none)


# --------------------------------------------------------------------------
# labels_from_bboxes (LLMパーツ検出のbbox -> 2Dラベル画像、純粋関数)
# --------------------------------------------------------------------------
def test_labels_from_bboxes_smaller_box_wins_over_larger_overlapping_box():
    """胴体(大きいbbox)と帽子(胴体に包含される小さいbbox)が重なる場合、
    小さい方(帽子)が優先されて上書きされることを検証する。"""
    h, w = 100, 100
    alpha = np.full((h, w), 255, dtype=np.uint8)

    # 胴体: 画像の大部分を占める大きいbbox
    body_bbox = (0.1, 0.1, 0.9, 0.9)
    # 帽子: 胴体bboxに完全に包含される小さいbbox(画面上部)
    hat_bbox = (0.3, 0.15, 0.7, 0.35)

    label_img = labels_from_bboxes([body_bbox, hat_bbox], alpha)

    assert label_img.shape == (h, w)
    # 帽子bbox中心の画素は帽子ラベル(index 1)であること(胴体に飲まれない)
    hat_cy, hat_cx = int(0.25 * h), int(0.5 * w)
    assert label_img[hat_cy, hat_cx] == 1
    # 胴体のみの領域(帽子bboxの外、胴体bboxの内)は胴体ラベル(index 0)
    body_cy, body_cx = int(0.7 * h), int(0.5 * w)
    assert label_img[body_cy, body_cx] == 0


def test_labels_from_bboxes_alpha_zero_outside_stays_background():
    h, w = 50, 50
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[:, :25] = 255  # 左半分のみ不透明

    bbox = (0.0, 0.0, 1.0, 1.0)  # 画像全体を覆うbbox
    label_img = labels_from_bboxes([bbox], alpha)

    assert np.all(label_img[:, 25:] == -1)  # アルファ0側は常に背景
    assert np.all(label_img[:, :25] == 0)   # 不透明側はラベル0


def test_labels_from_bboxes_empty_bboxes_returns_all_background():
    h, w = 20, 20
    alpha = np.full((h, w), 255, dtype=np.uint8)
    label_img = labels_from_bboxes([], alpha)
    assert np.all(label_img == -1)


def test_labels_from_bboxes_order_independent_of_input_order():
    """描画順は面積降順で内部的に決まるため、入力リストの順序を変えても
    (インデックス自体は変わるが)小さいbboxが勝つ結果は変わらない。"""
    h, w = 100, 100
    alpha = np.full((h, w), 255, dtype=np.uint8)
    body_bbox = (0.1, 0.1, 0.9, 0.9)
    hat_bbox = (0.3, 0.15, 0.7, 0.35)

    label_img_a = labels_from_bboxes([body_bbox, hat_bbox], alpha)
    label_img_b = labels_from_bboxes([hat_bbox, body_bbox], alpha)

    hat_cy, hat_cx = int(0.25 * h), int(0.5 * w)
    # 入力順序に関わらず、その画素は「帽子bboxのインデックス」になっている
    assert label_img_a[hat_cy, hat_cx] == 1  # hat_bboxはindex1
    assert label_img_b[hat_cy, hat_cx] == 0  # hat_bboxはindex0(順序を入れ替えたため)


# --------------------------------------------------------------------------
# decompose_parts: LLM由来の image_labels + part_names でパーツ名が付く
# --------------------------------------------------------------------------
def test_decompose_parts_with_part_names_assigns_names_to_llm_guided_split():
    """`image_labels`(bbox由来)+`part_names`を渡すと、画像誘導サブ分割で
    生じたパーツに対応する部位名が `name` フィールドとして付与されることを
    検証する(くびれのないカプセルを2パーツに分割するケースを流用)。"""
    capsule = make_waistless_capsule()
    img = make_two_region_vertical_image(height_fraction=0.45)
    alpha = img[:, :, 3]

    # 上(頭)・下(胴体)の2bboxをLLM検出結果に見立てて用意する
    head_bbox = (0.0, 0.0, 1.0, 0.45)
    body_bbox = (0.0, 0.45, 1.0, 1.0)
    image_labels = labels_from_bboxes([head_bbox, body_bbox], alpha)
    part_names = {0: "頭", 1: "胴体"}

    labels, stats = decompose_parts(
        capsule,
        n_parts_hint=0,
        seed=0,
        image_labels=image_labels,
        part_names=part_names,
        significant_chunk_fraction=0.08,
    )

    assert len(stats) == 2
    names = {s["name"] for s in stats}
    assert names == {"頭", "胴体"}


def test_decompose_parts_without_part_names_has_no_name_field_forced():
    """`part_names` を渡さない従来経路では `name` フィールドが追加されない
    (回帰なし)。"""
    capsule = make_waistless_capsule()
    labels, stats = decompose_parts(capsule, n_parts_hint=0, seed=0)
    for s in stats:
        assert "name" not in s


# --------------------------------------------------------------------------
# 手動シード誘導分解 (labels_from_seeds / decompose_parts(seed_points=...))
# --------------------------------------------------------------------------
def make_dumbbell_mesh(theta_n: int = 32, n_z: int = 41) -> trimesh.Trimesh:
    """くびれのあるダンベル形状(2ローブ+くびれ)を回転体として直接構築する。

    `trimesh.boolean.union` はエンジン非導入環境で使えず、voxelize→
    marching_cubes によるフォールバック(他のフィクスチャで使用)はこの
    ピッチでは接合部の凹二面角が平滑化されて消えてしまい、くびれ誘導の
    検証に使えない(marching_cubes出力で実測確認済み)。そのためここでは
    半径プロファイル `r(z)` を明示的に与えて頂点を直接生成し、回転体として
    メッシュ化する(booleanなし、凹二面角がくっきり残る)。

    半径プロファイル: 2つのガウス山(z=0, z=60中心、ピーク半径20)の
    max()に、くびれ最小半径6.0のフロアを掛けたもの。z=30付近で
    急激に狭まるくびれができる。
    """
    z_ctrl = np.linspace(0.0, 60.0, n_z)

    def profile_r(z: np.ndarray) -> np.ndarray:
        r1 = 20.0 * np.exp(-(((z - 0.0) / 22.0) ** 2))
        r2 = 20.0 * np.exp(-(((z - 60.0) / 22.0) ** 2))
        neck = 6.0
        return np.maximum(neck, np.maximum(r1, r2))

    angles = np.linspace(0.0, 2.0 * np.pi, theta_n, endpoint=False)
    verts = []
    for z in z_ctrl:
        r = profile_r(np.array([z]))[0]
        for a in angles:
            verts.append([r * np.cos(a), r * np.sin(a), z])
    verts = np.asarray(verts)

    faces = []
    for i in range(n_z - 1):
        for j in range(theta_n):
            jn = (j + 1) % theta_n
            a = i * theta_n + j
            b = i * theta_n + jn
            c = (i + 1) * theta_n + j
            d = (i + 1) * theta_n + jn
            faces.append([a, b, d])
            faces.append([a, d, c])

    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)


@pytest.fixture(scope="module")
def dumbbell():
    return make_dumbbell_mesh()


def test_labels_from_seeds_boundary_near_waist(dumbbell):
    """2ローブの中心をシードにすると、境界(ラベル切り替わり面)がくびれ
    (z≈30、半径最小)付近に来ることを検証する。"""
    seeds = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 60.0]])
    labels = labels_from_seeds(dumbbell, seeds)

    assert set(np.unique(labels).tolist()) == {0, 1}

    centers = dumbbell.triangles_center
    adjacency = dumbbell.face_adjacency
    diff = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
    assert np.any(diff)
    boundary_z = centers[adjacency[diff][:, 0], 2]
    # くびれの中心はz=30。ダンベル全高60に対し±20%以内に境界が来ること。
    assert abs(float(np.mean(boundary_z)) - 30.0) <= 0.2 * 60.0


def test_labels_from_seeds_both_labels_present_on_each_lobe(dumbbell):
    seeds = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 60.0]])
    labels = labels_from_seeds(dumbbell, seeds)
    centers = dumbbell.triangles_center

    lobe0 = labels[centers[:, 2] < 15.0]
    lobe1 = labels[centers[:, 2] > 45.0]
    # 各ローブの大部分は対応するシードのラベルで占められる
    assert np.mean(lobe0 == 0) > 0.9
    assert np.mean(lobe1 == 1) > 0.9


def test_labels_from_seeds_concavity_pulls_boundary_toward_waist(dumbbell):
    """シードをz方向に非対称配置しても、凹み誘導(concavity_weight)を強めると
    境界がくびれ(z=30)へより近づくことを検証する(重み設計の妥当性確認)。
    凹み項の寄与を単独で測るため、肉厚項・背面仮想シード伝播・境界平面化を
    明示的に無効化する(デフォルトの肉厚項だけでも境界がほぼくびれに来て
    しまい、また背面仮想シード伝播は今回2シードとも十分近いためownership/
    balanceガードで棄却されるはずだが、念のため明示的にOFFにして凹み項の
    寄与だけを切り分ける)。"""
    seeds = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 60.0]])
    centers = dumbbell.triangles_center
    adjacency = dumbbell.face_adjacency

    def boundary_mean_z(weight: float) -> float:
        labels = labels_from_seeds(
            dumbbell,
            seeds,
            concavity_weight=weight,
            thickness_weight=0.0,
            propagate_opposite=False,
            planar_boundaries=False,
        )
        diff = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
        return float(np.mean(centers[adjacency[diff][:, 0], 2]))

    z_no_concavity = boundary_mean_z(0.0)
    z_default = boundary_mean_z(8.0)
    # 既定の凹み重みは、重みなしよりくびれ(z=30)に近い境界を作る
    assert abs(z_default - 30.0) < abs(z_no_concavity - 30.0)


def test_labels_from_seeds_duplicate_seed_on_same_face_raises():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    seeds = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
    with pytest.raises(ValueError):
        labels_from_seeds(sphere, seeds)


def test_labels_from_seeds_requires_at_least_two_points():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    with pytest.raises(ValueError):
        labels_from_seeds(sphere, np.array([[0.0, 0.0, 10.0]]))


def test_labels_from_seeds_disconnected_component_fallback():
    """双対グラフ上で到達不能な非連結成分(離れた別の塊)は、シード面への
    ユークリッド距離による最近傍フォールバックでラベルが埋まることを検証する。"""
    sphere_a = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    sphere_b = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    sphere_b.apply_translation([100.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([sphere_a, sphere_b])

    seeds = np.array([[10.0, 0.0, 0.0], [110.0, 0.0, 0.0]])
    labels = labels_from_seeds(mesh, seeds)

    centers = mesh.triangles_center
    labels_a = labels[centers[:, 0] < 50.0]
    labels_b = labels[centers[:, 0] >= 50.0]
    assert set(np.unique(labels_a).tolist()) == {0}
    assert set(np.unique(labels_b).tolist()) == {1}


def test_decompose_parts_with_seed_points_assigns_seed_names(dumbbell):
    seeds = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 60.0]])
    labels, stats = decompose_parts(dumbbell, seed_points=seeds, seed_names=["頭", "胴体"])

    assert len(stats) == 2
    names = {s["part_id"]: s["name"] for s in stats}
    assert names == {0: "頭", 1: "胴体"}
    assert labels.shape == (len(dumbbell.faces),)
    for s in stats:
        assert s["connected"]
        assert s["watertight_after_cap"]
        assert s["n_faces"] > 0


def test_decompose_parts_with_seed_points_without_names_has_none_name(dumbbell):
    seeds = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 60.0]])
    labels, stats = decompose_parts(dumbbell, seed_points=seeds)
    for s in stats:
        assert s["name"] is None


def test_decompose_parts_with_seed_points_ignores_n_parts_hint(dumbbell):
    """`seed_points` 指定時は `n_parts_hint` を無視し、パーツ数はシード数に
    従うことを検証する。"""
    seeds = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 60.0]])
    labels, stats = decompose_parts(dumbbell, n_parts_hint=6, seed_points=seeds)
    assert len(stats) == 2


def test_decompose_parts_with_many_seed_points_protects_small_parts():
    """小さいパーツ(耳のような突起)をシードで明示指定した場合、
    `_absorb_tiny_parts` に吸収されず残ることを検証する。"""
    body = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    ear = trimesh.creation.icosphere(subdivisions=2, radius=4.0)
    ear.apply_translation([0.0, 0.0, 32.0])
    mesh = trimesh.util.concatenate([body, ear])

    seeds = np.array(
        [
            [0.0, 0.0, -25.0],   # 胴体の反対側
            [0.0, 0.0, 34.0],    # 耳(小パーツ、全体の1%未満の面積)
        ]
    )
    labels, stats = decompose_parts(mesh, seed_points=seeds, seed_names=["胴体", "耳"])

    assert len(stats) == 2
    total_area = sum(s["area_mm2"] for s in stats)
    ear_stat = next(s for s in stats if s["name"] == "耳")
    # 耳は全体の数%程度の小面積のはずだが、保護によりパーツとして残っている
    assert ear_stat["area_mm2"] > 0
    assert ear_stat["area_mm2"] / total_area < 0.1


# --------------------------------------------------------------------------
# 手動シード誘導: 同名シードのマージ(マルチソース)
# --------------------------------------------------------------------------
def test_labels_from_seeds_same_group_merges_multi_source(dumbbell):
    """同一グループの複数シード(正面+背面を模して同ローブの対角2点)が
    1つのラベルに統合され、ラベル数=グループ数になることを検証する。"""
    # 下ローブに2点(x正側とx負側=正面と背面のつもり)、上ローブに1点
    seeds = np.array(
        [
            [15.0, 0.0, 5.0],
            [-15.0, 0.0, 5.0],
            [0.0, 0.0, 60.0],
        ]
    )
    groups = np.array([0, 0, 1])
    labels = labels_from_seeds(dumbbell, seeds, seed_groups=groups)

    assert set(np.unique(labels).tolist()) == {0, 1}

    centers = dumbbell.triangles_center
    lobe0 = labels[centers[:, 2] < 15.0]
    lobe1 = labels[centers[:, 2] > 45.0]
    assert np.mean(lobe0 == 0) > 0.9
    assert np.mean(lobe1 == 1) > 0.9


def test_labels_from_seeds_duplicate_face_same_group_allowed():
    """同一グループのシードが同じ面にスナップしても(単に冗長なだけなので)
    エラーにならないことを検証する。"""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    seeds = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],   # 同一面スナップ(同グループ)
            [0.0, 0.0, -10.0],
        ]
    )
    groups = np.array([0, 0, 1])
    labels = labels_from_seeds(sphere, seeds, seed_groups=groups)
    assert set(np.unique(labels).tolist()) == {0, 1}


def test_labels_from_seeds_duplicate_face_different_group_raises():
    """異なるグループのシードが同じ面にスナップした場合はValueError。"""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    seeds = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
    groups = np.array([0, 1])
    with pytest.raises(ValueError):
        labels_from_seeds(sphere, seeds, seed_groups=groups)


def test_labels_from_seeds_single_group_raises():
    """グループ数(パーツ数)が1のときはValueError(分割にならないため)。"""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    seeds = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]])
    groups = np.array([0, 0])
    with pytest.raises(ValueError):
        labels_from_seeds(sphere, seeds, seed_groups=groups)


def test_decompose_parts_same_name_seeds_merge_into_one_part(dumbbell):
    """同名シード(「胴体」×2点)が1パーツに統合され、パーツ数=ユニーク名数
    になり、名前がグループに正しく対応することを検証する。"""
    seeds = np.array(
        [
            [15.0, 0.0, 5.0],    # 胴体(正面のつもり)
            [-15.0, 0.0, 5.0],   # 胴体(背面のつもり)
            [0.0, 0.0, 60.0],    # 頭
        ]
    )
    labels, stats = decompose_parts(
        dumbbell, seed_points=seeds, seed_names=["胴体", "胴体", "頭"]
    )

    assert len(stats) == 2
    names = {s["part_id"]: s["name"] for s in stats}
    # グループIDは名前の初出順: 胴体=0, 頭=1
    assert names == {0: "胴体", 1: "頭"}
    assert set(np.unique(labels).tolist()) == {0, 1}


# --------------------------------------------------------------------------
# 手動シード誘導: 肉厚項(耳モック=凹み信号の弱い付け根での境界誘導)
# --------------------------------------------------------------------------
def make_ear_mock_mesh(pitch: float = 0.8, target_faces: int = 10000) -> trimesh.Trimesh:
    """大球(頭、r28)+細い付け根(r2.5)+小球(耳、r6)の合成メッシュ。

    voxelize→marching_cubes で結合・等方三角形化する。この過程で付け根の
    凹二面角はほぼ平滑化されて消える(z=40〜60帯の凹み強度平均が全体と
    ほぼ同水準になることを実測確認済み)ため、「Hunyuan3D生成メッシュの
    背面(画像補完由来で滑らか)では凹み誘導が効かない」というトライアルで
    報告された失敗モードを再現する。一方、肉厚コントラスト
    (頭≈56mm / 付け根≈5mm / 耳≈12mm)は保存されるため、肉厚項
    (thickness_weight)の効果検証に使える。
    """
    head = trimesh.creation.icosphere(subdivisions=4, radius=28.0)
    head.apply_translation([0.0, 0.0, 28.0])
    ear = trimesh.creation.icosphere(subdivisions=3, radius=6.0)
    ear.apply_translation([0.0, 0.0, 62.0])
    neck = trimesh.creation.cylinder(radius=2.5, height=10.0, sections=16)
    neck.apply_translation([0.0, 0.0, 54.0])
    concat = trimesh.util.concatenate([head, ear, neck])
    vox = concat.voxelized(pitch=pitch).fill()
    mesh = vox.marching_cubes
    mesh.apply_transform(vox.transform)
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    return mesh


@pytest.fixture(scope="module")
def ear_mock():
    return make_ear_mock_mesh()


def _boundary_z_values(mesh: trimesh.Trimesh, labels: np.ndarray) -> np.ndarray:
    adjacency = mesh.face_adjacency
    diff = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
    assert np.any(diff)
    return mesh.triangles_center[adjacency[diff][:, 0], 2]


def test_ear_mock_thickness_term_pulls_boundary_to_neck(ear_mock):
    """凹み信号の弱い耳モックで、デフォルトの肉厚項により境界が付け根
    (z≈50〜58)近傍に来ることを検証する。肉厚項なしでは境界が頭の途中
    (z≈36)で止まる(=トライアルで報告された耳の分離不良)。"""
    seeds = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 67.5]])  # 頭の底・耳の先端

    labels_default = labels_from_seeds(ear_mock, seeds)
    bz_default = float(np.mean(_boundary_z_values(ear_mock, labels_default)))

    labels_no_thick = labels_from_seeds(ear_mock, seeds, thickness_weight=0.0)
    bz_no_thick = float(np.mean(_boundary_z_values(ear_mock, labels_no_thick)))

    neck_center = 54.0  # 付け根(円柱)の中心z
    # デフォルト(肉厚項あり)は付け根帯(z 50〜58)の近傍に境界が来る
    assert 46.0 <= bz_default <= 60.0
    # 肉厚項なしより明確に付け根に近い
    assert abs(bz_default - neck_center) < abs(bz_no_thick - neck_center)


def test_ear_mock_ear_part_is_small_fraction(ear_mock):
    """耳パーツの面積が全体の少数派(頭より小さい)になることを検証する
    (境界が頭の中腹まで下がって耳が過大にならない)。"""
    seeds = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 67.5]])
    labels, stats = decompose_parts(
        ear_mock, seed_points=seeds, seed_names=["頭", "耳"]
    )
    assert len(stats) == 2
    ear_stat = next(s for s in stats if s["name"] == "耳")
    head_stat = next(s for s in stats if s["name"] == "頭")
    assert ear_stat["area_mm2"] < 0.5 * head_stat["area_mm2"]


# --------------------------------------------------------------------------
# 手動シード誘導: エッジ重み加算形(胴+腕モック=凹みと肉厚遷移が共存する付け根)
# --------------------------------------------------------------------------
_ARM_ORIGIN = np.array([25.0, 0.0, 40.0])  # 腕の付け根(胴表面のx最大点)
_ARM_DIR = np.array([1.0, 0.0, 0.0])       # 腕の軸方向(真横+X、Tポーズ)


def make_torso_arm_mesh(pitch: float = 0.7, target_faces: int = 15000) -> trimesh.Trimesh:
    """Tポーズの胴+腕モック: 楕円体の胴(25×15×40)+真横に伸びる円柱の腕(r7)。

    腕の付け根に環状の溝(半径7〜10.5、腕軸座標1〜4.5)をボクセルレベルで彫り、
    実際の生成メッシュの脇(腕と胴の間の深い溝)を再現する。付け根には
    **強い凹み(溝)と肉厚急変(腕14mm vs 胴30〜50mm)が同時に存在**し、
    エッジ重みの乗算形で交差項が過大になっていた失敗モード(実機トライアルで
    報告された胴体/腕分離の退行)を検証するためのフィクスチャ。
    voxelize→marching_cubes→prepare_mesh(実パイプラインと同じ平滑化・簡略化)
    を通す。
    """
    torso = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    torso.apply_scale([25.0, 15.0, 40.0])
    torso.apply_translation([0.0, 0.0, 40.0])
    arm = trimesh.creation.cylinder(radius=7.0, height=40.0, sections=24)
    rot = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
    arm.apply_transform(rot)
    arm.apply_translation([38.0, 0.0, 40.0])
    concat = trimesh.util.concatenate([torso, arm])
    vox = concat.voxelized(pitch=pitch).fill()

    # ボクセル中心のワールド座標を計算し、腕の付け根の環状帯を彫る
    matrix = vox.matrix.copy()
    idx = np.argwhere(matrix)
    homog = np.column_stack([idx, np.ones(len(idx))])
    world = (vox.transform @ homog.T).T[:, :3]
    rel = world - _ARM_ORIGIN
    t = rel @ _ARM_DIR
    radial = np.linalg.norm(rel - np.outer(t, _ARM_DIR), axis=1)
    carve = (radial > 7.0) & (radial < 10.5) & (t > 1.0) & (t < 4.5)
    matrix[idx[carve, 0], idx[carve, 1], idx[carve, 2]] = False
    vox = trimesh.voxel.VoxelGrid(matrix, transform=vox.transform)

    mesh = vox.marching_cubes
    mesh.apply_transform(vox.transform)

    from server.pattern import prepare_mesh

    return prepare_mesh(mesh, target_faces=target_faces, smooth_iterations=10)


@pytest.fixture(scope="module")
def torso_arm():
    return make_torso_arm_mesh()


def _torso_arm_over_under(mesh: trimesh.Trimesh, labels: np.ndarray) -> tuple[float, float]:
    """幾何学的な真値(腕軸から半径11mm以内かつ付け根より先=真の腕)に対する
    誤分類率を返す: (胴の腕への誤食い込み率, 腕の取りこぼし率)。"""
    c = mesh.triangles_center
    rel = c - _ARM_ORIGIN
    t = rel @ _ARM_DIR
    radial = np.linalg.norm(rel - np.outer(t, _ARM_DIR), axis=1)
    gt_arm = (radial < 11.0) & (t > 0.0)
    pred_arm = labels == 1
    over = float(np.sum(pred_arm & ~gt_arm) / max(np.sum(~gt_arm), 1))
    under = float(np.sum(~pred_arm & gt_arm) / max(np.sum(gt_arm), 1))
    return over, under


def test_torso_arm_boundary_near_junction(torso_arm):
    """凹みと肉厚遷移が共存する腕の付け根で、加算形エッジ重み(既定)の境界が
    付け根近傍に来ることを検証する。乗算形では交差項により共存部のコストが
    過大になり境界が乱れる退行が実機で報告された(本テストは採用した加算形の
    回帰テスト)。"""
    seeds = np.array([[0.0, -15.0, 40.0], [58.0, 0.0, 40.0]])  # 胴正面中央・腕先端
    labels = labels_from_seeds(torso_arm, seeds)

    over, under = _torso_arm_over_under(torso_arm, labels)
    # 胴の腕への誤食い込みがほぼゼロ(実測1.0%。凹みのみでは15〜21%)
    assert over < 0.08
    # 境界リングが付け根近傍(腕軸座標で±数mm)にあること
    # (実測: mean=-0.4。溝は腕軸1〜4.5、離散化により±10mm程度は揺れうる)
    adjacency = torso_arm.face_adjacency
    diff = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
    assert np.any(diff)
    b_ax = (torso_arm.triangles_center[adjacency[diff][:, 0]] - _ARM_ORIGIN) @ _ARM_DIR
    assert -10.0 <= float(np.mean(b_ax)) <= 12.0


def test_torso_arm_thickness_term_reduces_overreach(torso_arm):
    """肉厚項なし(凹みのみ)では胴の腕への誤食い込みが大きく、既定
    (加算形+肉厚項)で明確に減ることを検証する(肉厚項の有効性確認)。"""
    seeds = np.array([[0.0, -15.0, 40.0], [58.0, 0.0, 40.0]])

    labels_default = labels_from_seeds(torso_arm, seeds)
    over_default, _ = _torso_arm_over_under(torso_arm, labels_default)

    labels_no_thick = labels_from_seeds(torso_arm, seeds, thickness_weight=0.0)
    over_no_thick, _ = _torso_arm_over_under(torso_arm, labels_no_thick)

    assert over_default < over_no_thick


# --------------------------------------------------------------------------
# 手動シード誘導: 背面への仮想シード自動伝播 (propagate_opposite)
# --------------------------------------------------------------------------
def test_opposite_virtual_seed_sphere_antipode():
    """球の正面1点から、ほぼ対蹠点に同グループの仮想シードが生成される。"""
    from server.pattern.parts import _opposite_virtual_seed_faces, _snap_points_to_faces

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    seed_faces = _snap_points_to_faces(sphere, np.array([[10.0, 0.0, 0.0]]))
    virt_faces, virt_groups = _opposite_virtual_seed_faces(
        sphere, seed_faces, np.array([0])
    )

    assert len(virt_faces) == 1
    assert virt_groups[0] == 0
    virt_center = sphere.triangles_center[virt_faces[0]]
    # 対蹠点 (-10, 0, 0) の近傍(面重心の離散化を考慮して±2mm)
    assert np.linalg.norm(virt_center - np.array([-10.0, 0.0, 0.0])) < 2.0


def test_opposite_virtual_seed_cylinder_opposite_side():
    """円柱の側面1点から、反対側の側面に仮想シードが生成される。"""
    from server.pattern.parts import _opposite_virtual_seed_faces, _snap_points_to_faces

    cyl = trimesh.creation.cylinder(radius=10.0, height=80.0, sections=32)
    seed_faces = _snap_points_to_faces(cyl, np.array([[10.0, 0.0, 0.0]]))
    virt_faces, _ = _opposite_virtual_seed_faces(cyl, seed_faces, np.array([0]))

    assert len(virt_faces) == 1
    seed_center = cyl.triangles_center[seed_faces[0]]
    virt_center = cyl.triangles_center[virt_faces[0]]
    # x座標が反対側(側面の逆法線=径方向なのでxはほぼ-10)、zはレイの通る
    # 高さ(=シード面重心の高さ)近傍
    assert virt_center[0] < -8.0
    assert abs(virt_center[2] - seed_center[2]) < 5.0


def test_opposite_virtual_seed_open_mesh_skips_without_error():
    """開メッシュ(平面)ではレイが何もヒットせず、例外を出さずにスキップされる。"""
    from server.pattern.parts import _opposite_virtual_seed_faces, _snap_points_to_faces

    plane_v = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=np.float64)
    plane_f = np.array([[0, 1, 2], [0, 2, 3]])
    plane = trimesh.Trimesh(vertices=plane_v, faces=plane_f, process=False)
    seed_faces = _snap_points_to_faces(plane, np.array([[5.0, 5.0, 0.0]]))
    virt_faces, virt_groups = _opposite_virtual_seed_faces(plane, seed_faces, np.array([0]))

    assert len(virt_faces) == 0
    assert len(virt_groups) == 0


def make_head_torso_mesh(pitch: float = 1.0, target_faces: int = 12000) -> trimesh.Trimesh:
    """頭+胴モック(雪だるま型): 胴球r24(中心z=24)+頭球r14(中心z=52)を
    voxelize→marching_cubes で結合した閉メッシュ。首のくびれは z≈46
    (半径最小、実測)。背面仮想シード伝播の検証用。"""
    torso = trimesh.creation.icosphere(subdivisions=4, radius=24.0)
    torso.apply_translation([0.0, 0.0, 24.0])
    head = trimesh.creation.icosphere(subdivisions=4, radius=14.0)
    head.apply_translation([0.0, 0.0, 52.0])
    vox = trimesh.util.concatenate([torso, head]).voxelized(pitch=pitch).fill()
    mesh = vox.marching_cubes
    mesh.apply_transform(vox.transform)
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    return mesh


@pytest.fixture(scope="module")
def head_torso():
    return make_head_torso_mesh()


_HEAD_TORSO_NECK_Z = 46.0  # くびれ(半径最小)の実測z


def _back_boundary_deviation(mesh: trimesh.Trimesh, labels: np.ndarray) -> float:
    """背面(y>3)の境界エッジの、首くびれzからの平均ズレ(mm)。"""
    adjacency = mesh.face_adjacency
    diff = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
    pts = mesh.triangles_center[adjacency[diff][:, 0]]
    back = pts[:, 1] > 3.0
    assert np.any(back)
    return float(np.abs(pts[back][:, 2] - _HEAD_TORSO_NECK_Z).mean())


def test_propagate_improves_back_boundary(head_torso):
    """正面のみのシード(胴正面中央+頭正面)では背面の境界が首から大きく
    ズレる(トライアルで報告された症状)が、背面仮想シードの自動伝播
    (propagate_opposite=True、既定)で首近傍まで改善することを検証する。"""
    seeds = np.array([[0.0, -24.0, 24.0], [0.0, -14.0, 52.0]])  # 胴正面・頭正面

    labels_off = labels_from_seeds(head_torso, seeds, propagate_opposite=False)
    dev_off = _back_boundary_deviation(head_torso, labels_off)

    virtual_out: list = []
    labels_on = labels_from_seeds(
        head_torso, seeds, propagate_opposite=True, virtual_seeds_out=virtual_out
    )
    dev_on = _back_boundary_deviation(head_torso, labels_on)

    # 仮想シードが両グループに生成される(胴背面と頭背面)
    assert len(virtual_out) == 2
    assert {v["group"] for v in virtual_out} == {0, 1}
    # 胴の仮想シードは背面(y>0)の胴の高さに、頭の仮想シードは背面の頭の高さにある
    torso_virt = next(v for v in virtual_out if v["group"] == 0)
    head_virt = next(v for v in virtual_out if v["group"] == 1)
    assert torso_virt["y"] > 15.0 and abs(torso_virt["z"] - 24.0) < 10.0
    assert head_virt["y"] > 8.0 and abs(head_virt["z"] - 52.0) < 8.0

    # 背面境界が首くびれに明確に近づく(実測: OFF≈26.5mm → ON≈6.4mm)
    assert dev_on < dev_off
    assert dev_on < 12.0


def test_propagate_lower_torso_seed_fixed_by_averaged_normal(head_torso):
    """胴シードを下部(単一面法線が上向きに傾く位置)に置いても、近傍平均
    法線により逆法線レイが正しく胴の背面へ抜け、両グループに正しい仮想
    シードが立って背面境界が大幅に改善することを検証する(旧実装では
    単一法線のレイが首を貫通して頭背面に誤配置され、ガードで全棄却=改善
    ゼロだった配置)。"""
    seeds = np.array([[0.0, -24.0, 16.0], [0.0, -14.0, 52.0]])  # 胴は下部(傾き法線)

    virtual_out: list = []
    labels_on = labels_from_seeds(
        head_torso, seeds, propagate_opposite=True, virtual_seeds_out=virtual_out
    )
    labels_off = labels_from_seeds(head_torso, seeds, propagate_opposite=False)

    # 両グループの仮想シードが背面(y>0)の正しい高さに生成される
    assert {v["group"] for v in virtual_out} == {0, 1}
    for v in virtual_out:
        assert v["y"] > 8.0
    # 背面境界が大幅に改善(実測: OFF≈18.1mm → ON≈0.9mm)
    dev_on = _back_boundary_deviation(head_torso, labels_on)
    dev_off = _back_boundary_deviation(head_torso, labels_off)
    assert dev_on < dev_off


def test_propagate_pairwise_balance_guard(head_torso):
    """一方の大パーツだけ仮想シード候補を持つ場合の均衡棄却を検証する:
    頭頂点(法線がほぼ真上向き)にシードを置くと、対蹠アライメント探索でも
    首を貫通するレイしか見つからず所有権チェックで棄却され、残った胴の
    候補も「隣接する頭が候補を持たず非コンパクト(面積比6%超)」のため
    ペア単位均衡(`balance`)で棄却され、propagate ON でも OFF と同一の
    結果になることを検証する(片側の大パーツだけの背面強化=境界悪化の防止)。"""
    centers = head_torso.triangles_center
    top_idx = int(np.argmax(centers[:, 2]))  # 頭頂点(法線がほぼ真上向き)
    seeds = np.array([[0.0, -24.0, 24.0], centers[top_idx]])  # 胴正面・頭頂点

    virtual_out: list = []
    skips_out: list = []
    labels_on = labels_from_seeds(
        head_torso,
        seeds,
        propagate_opposite=True,
        virtual_seeds_out=virtual_out,
        virtual_skips_out=skips_out,
    )
    labels_off = labels_from_seeds(head_torso, seeds, propagate_opposite=False)

    assert len(virtual_out) == 0
    assert np.array_equal(labels_on, labels_off)
    reasons = {s["group"]: s["reason"] for s in skips_out}
    assert reasons.get(1) == "ownership"  # 頭頂点: 首を貫通するレイの棄却
    assert reasons.get(0) == "balance"    # 胴: 均衡棄却


def test_propagate_pairwise_balance_multi_part():
    """4グループモック(胴+頭+耳×2、正面のみシード)で、少なくとも胴・頭の
    仮想シードが生き残ることを検証する(旧実装の「1グループでも候補ゼロなら
    全棄却」は多パーツ構成では事実上常時OFFだった=実ジョブ9パーツで
    n_virtual=0)。耳はコンパクト(面積比6%以下)なので頭の採用をブロック
    しない。耳自身が採用されるか棄却されるかは対蹠アライメント探索の乱数
    コーンに依存するため厳密には問わない(いずれのガードで弾かれても
    reasonは正当な値になることのみ検証する)。"""
    torso = trimesh.creation.icosphere(subdivisions=4, radius=24.0)
    torso.apply_translation([0.0, 0.0, 24.0])
    head = trimesh.creation.icosphere(subdivisions=4, radius=14.0)
    head.apply_translation([0.0, 0.0, 52.0])
    ear_l = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    ear_l.apply_translation([-12.0, 0.0, 62.0])
    ear_r = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    ear_r.apply_translation([12.0, 0.0, 62.0])
    vox = trimesh.util.concatenate([torso, head, ear_l, ear_r]).voxelized(pitch=1.0).fill()
    mesh = vox.marching_cubes
    mesh.apply_transform(vox.transform)
    if len(mesh.faces) > 14000:
        mesh = mesh.simplify_quadric_decimation(face_count=14000)

    seeds = np.array(
        [
            [0.0, -24.0, 24.0],   # 胴 正面
            [0.0, -14.0, 48.0],   # 頭 正面
            [-12.0, 0.0, 67.0],   # 左耳 先端
            [12.0, 0.0, 67.0],    # 右耳 先端
        ]
    )
    virtual_out: list = []
    skips_out: list = []
    labels_from_seeds(
        mesh,
        seeds,
        seed_groups=np.arange(4),
        virtual_seeds_out=virtual_out,
        virtual_skips_out=skips_out,
    )

    adopted = {v["group"] for v in virtual_out}
    assert {0, 1} <= adopted  # 胴・頭は少なくとも採用される
    # 不採用のグループがあれば、正当な理由(トンネル検出/所有権チェック/
    # ペア単位均衡)で棄却されていること
    skip_reasons = {s["group"]: s["reason"] for s in skips_out}
    for g in {0, 1, 2, 3} - adopted:
        assert skip_reasons.get(g) in ("tunnel", "ownership", "balance")


def make_head_torso_hat_mesh(
    pitch: float = 1.0, target_faces: int = 15000, tilt_deg: float = 0.0, noise_amp: float = 0.0, seed: int = 0
) -> trimesh.Trimesh:
    """頭+胴+帽子モック(実ジョブ「犬モデル」の頭部誤配置バグの再現用)。

    胴球r24(中心z=24)+頭球r14(中心z=52、`tilt_deg`度だけX軸回りに傾け
    可能)+帽子球r9(頭頂z=66付近)。`noise_amp>0` で全頂点に法線方向の
    ノイズ(ぬいぐるみの毛のバンプに相当)を加えられる。

    実ジョブでは、正面上部(頭頂寄り)に打った頭シードの仮想シードが
    首側(z低)に誤着地し、背面上部が隣接する帽子パーツの仮想シードに
    奪われる症状が確認された。本モックはその再現用。
    """
    torso = trimesh.creation.icosphere(subdivisions=4, radius=24.0)
    torso.apply_translation([0.0, 0.0, 24.0])

    head = trimesh.creation.icosphere(subdivisions=4, radius=14.0)
    if tilt_deg != 0.0:
        rot = trimesh.transformations.rotation_matrix(np.radians(tilt_deg), [1, 0, 0])
        head.apply_transform(rot)
    head.apply_translation([0.0, 0.0, 52.0])

    hat = trimesh.creation.icosphere(subdivisions=3, radius=9.0)
    hat.apply_translation([0.0, 0.0, 66.0])

    if noise_amp > 0.0:
        rng = np.random.default_rng(seed)
        for m in (torso, head, hat):
            vertex_normals = m.vertex_normals
            noise = rng.normal(0.0, noise_amp, size=len(m.vertices))
            m.vertices = m.vertices + vertex_normals * noise[:, None]

    concat = trimesh.util.concatenate([torso, head, hat])
    vox = concat.voxelized(pitch=pitch).fill()
    mesh = vox.marching_cubes
    mesh.apply_transform(vox.transform)
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    return mesh


@pytest.fixture(scope="module")
def head_torso_hat():
    # 実ジョブに近い条件(頭の傾き+毛のバンプ相当のノイズ)で再現する
    return make_head_torso_hat_mesh(tilt_deg=10.0, noise_amp=0.4)


def test_propagate_head_virtual_seed_lands_on_head_not_neck(head_torso_hat):
    """頭+胴+帽子モックで、頭頂寄り(実ジョブ相当)に打った頭シードの
    仮想シードが、首側(z低)や帽子側ではなく頭部相当のz範囲に着地する
    ことを検証する(対蹠アライメント探索による修正の回帰テスト。
    修正前は近傍平均逆法線1本のみのため首側z≈51付近に誤着地し、背面上部が
    帽子パーツに奪われていた)。"""
    seeds = np.array(
        [
            [0.0, -24.0, 24.0],   # 胴 正面
            [0.0, -13.0, 63.0],   # 頭 正面上部(頭頂寄り、実ジョブ相当)
            [0.0, -6.0, 74.0],    # 帽子 正面
        ]
    )
    virtual_out: list = []
    labels_from_seeds(
        head_torso_hat, seeds, seed_groups=np.array([0, 1, 2]), virtual_seeds_out=virtual_out
    )

    head_virtual = [v for v in virtual_out if v["group"] == 1]
    assert len(head_virtual) == 1
    # 頭部の妥当なz範囲(首くびれ~45より上、帽子の付け根~63より下の帯)に着地する。
    # 修正前は z≈51(首寄り)に落ちていた。
    assert 48.0 <= head_virtual[0]["z"] <= 66.0
    assert head_virtual[0]["y"] > 5.0  # 背面(y>0)であること


def test_propagate_head_reclaims_back_top_from_hat(head_torso_hat):
    """頭+胴+帽子モックで、頭の仮想シードが正しい位置に着地した結果、
    背面上部(頭頂付近)の面が帽子パーツに独占されず、頭パーツにも
    有意な面積が割り当てられることを検証する(実ジョブ実測: 修正前は
    頭10面/帽子1050面=頭がほぼ0%だったのに対し、修正後は頭が背面上部の
    相当割合(実測47%)を獲得するようになった)。"""
    seeds = np.array(
        [
            [0.0, -24.0, 24.0],
            [0.0, -13.0, 63.0],
            [0.0, -6.0, 74.0],
        ]
    )
    labels = labels_from_seeds(head_torso_hat, seeds, seed_groups=np.array([0, 1, 2]))
    centers = head_torso_hat.triangles_center
    # 背面上部(帽子の付け根より低いz帯、頭部が担うべき領域)
    back_upper_head = (centers[:, 1] > 3.0) & (centers[:, 2] > 55.0) & (centers[:, 2] < 63.0)
    assert np.any(back_upper_head)
    head_fraction = float(np.mean(labels[back_upper_head] == 1))
    # 独占されていない(=修正前のほぼ0%から明確に改善)ことを検証する
    assert head_fraction > 0.3


def test_propagate_hat_virtual_seed_tunnel_known_limitation(head_torso_hat):
    """頭+胴+帽子モックで、帽子(最小パーツ)から内向きに飛ばしたレイが頭を
    貫通して首元まで抜けてしまう既知の失敗モードを記録する回帰テスト
    (実ジョブで報告された不具合の再現)。

    不具合の実体: `compute_local_thickness` の拡散補間・平滑化により、
    帽子のような小パーツのシード面で局所肉厚が過大評価される(本モック実測:
    帽子シード面19.3mm、帽子の直径は18mm相当だが頭の厚い肉厚(約47mm)の
    滲み出しの影響を受けた値)。この過大評価により、トンネル検出の閾値
    (肉厚×3倍)が実際より緩くなり、頭を貫通して首元(z≈49)まで抜ける
    誤ったexit(距離約31mm)が、対蹠アライメントスコアで正しい候補
    (帽子自身の背面、距離約10〜16mm、score最大0.77)より高スコア
    (score=0.869)になり採用されてしまう。

    **調査の結論(本テストが「既知の限界」を記録する理由)**: この誤exit
    の所有権比率(実測1.04)は、十字モックのような「正しい対蹠面が測地的に
    遠い」正当ケース(実測1.235)や頭部の正当な対蹠面(実測1.15、実ジョブでは
    1.27)と値域が重なっており、所有権チェックの閾値調整・対蹠アライメント
    スコアとの複合スコアリング(score - alpha*ownership_ratio等)・距離
    プールの最近傍クラスタ絞り込み等、複数の方式を実装して検証したが、
    いずれも本ケースを分離するための閾値では十字モック・頭頂点シード等の
    既存正当ケースを退行させてしまった(詳細な実測比較は完了報告参照)。

    このため本修正では、この特定のケースの完全解決は見送り、代わりに
    `test_propagate_head_reclaims_back_top_from_hat` が検証する
    「頭が背面上部の相当割合を確保する」という緩和効果(多数決平滑化・
    ペア単位均衡による部分的な是正)に委ねる。本テストは、この既知の
    限界が将来のリファクタで気づかれずに悪化しないよう、現状の(望ましく
    ない)挙動を明示的に固定する回帰テストとして存在する。将来この
    tunnel/ownershipガードが改善され本テストが失敗するようになった場合は、
    ガード改善の成功を意味するため、アサーションを反転して更新してよい。"""
    seeds = np.array(
        [
            [0.0, -24.0, 24.0],   # 胴 正面
            [0.0, -13.0, 63.0],   # 頭 正面上部
            [0.0, -6.0, 74.0],    # 帽子 正面
        ]
    )
    virtual_out: list = []
    skips_out: list = []
    labels_from_seeds(
        head_torso_hat,
        seeds,
        seed_groups=np.array([0, 1, 2]),
        virtual_seeds_out=virtual_out,
        virtual_skips_out=skips_out,
    )

    hat_virtual = [v for v in virtual_out if v["group"] == 2]
    # 既知の限界: 現状は帽子の仮想シードが首元(z<55)へ誤配置されたまま
    # 採用されてしまう。誤配置されても後述のバランス緩和で頭の背面獲得は
    # 一定確保されることは別テストで検証済み。
    assert len(hat_virtual) == 1
    assert hat_virtual[0]["z"] < 55.0


# --------------------------------------------------------------------------
# 手動シード誘導: 境界の平面フィット正則化 (planar boundary regularization)
# --------------------------------------------------------------------------
def _boundary_plane_rms(mesh: trimesh.Trimesh, labels: np.ndarray) -> float:
    """境界エッジ中点の(等重み)最良フィット平面まわりのRMS(mm)。
    小さいほど境界が平面的=取付口が縫いやすい楕円に近い。"""
    adjacency = mesh.face_adjacency
    edges_v = mesh.face_adjacency_edges
    diff = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
    mids = mesh.vertices[edges_v[np.where(diff)[0]]].mean(axis=1)
    centroid = mids.mean(axis=0)
    d = mids - centroid
    cov = d.T @ d / len(d)
    _eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    return float(np.sqrt(np.mean((d @ normal) ** 2)))


def test_planar_boundary_flattens_head_torso_ring(head_torso):
    """頭+胴モックで、平面化により境界リングの平面性(最良平面まわりRMS)が
    明確に改善することを検証する(実測: 2.52mm → 0.30mm。側面のうねりが
    消えて縫いやすい平面的な楕円になる)。"""
    seeds = np.array([[0.0, -24.0, 24.0], [0.0, -14.0, 52.0]])

    labels_off = labels_from_seeds(head_torso, seeds, planar_boundaries=False)
    rms_off = _boundary_plane_rms(head_torso, labels_off)

    report: list = []
    labels_on = labels_from_seeds(
        head_torso, seeds, planar_boundaries=True, planar_fit_out=report
    )
    rms_on = _boundary_plane_rms(head_torso, labels_on)

    # 平面フィットが適用され、レポートにRMSが記録される
    assert any(r["applied"] for r in report)
    assert all("rms" in r for r in report if r["applied"])
    # 平面性が明確に改善する
    assert rms_on < rms_off
    assert rms_on < 1.0


def test_planar_boundary_tightens_arm_ring(torso_arm):
    """胴+腕モックで、平面化により境界リングの腕軸方向のばらつき(std)が
    減り、胴の腕への誤食い込みが増えないことを検証する
    (実測: std 2.62 → 1.80、over 1.0% → 1.0%)。"""
    seeds = np.array([[0.0, -15.0, 40.0], [58.0, 0.0, 40.0]])

    def boundary_axis_std(labels):
        adjacency = torso_arm.face_adjacency
        diff = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
        b_ax = (torso_arm.triangles_center[adjacency[diff][:, 0]] - _ARM_ORIGIN) @ _ARM_DIR
        return float(np.std(b_ax))

    labels_off = labels_from_seeds(torso_arm, seeds, planar_boundaries=False)
    labels_on = labels_from_seeds(torso_arm, seeds, planar_boundaries=True)

    assert boundary_axis_std(labels_on) < boundary_axis_std(labels_off)
    over_on, _ = _torso_arm_over_under(torso_arm, labels_on)
    assert over_on < 0.08


def make_cross_mesh(pitch: float = 0.8, target_faces: int = 10000) -> trimesh.Trimesh:
    """十字モック: 縦円柱r8と横円柱r8の直交交差。2シード(縦の上端・横の端)で
    分割すると境界が交差部を巻くサドル状(非平面)になり、平面フィットの
    品質ガード(rms/degenerate)の発動を検証できる。"""
    v = trimesh.creation.cylinder(radius=8.0, height=60.0, sections=24)
    h = trimesh.creation.cylinder(radius=8.0, height=60.0, sections=24)
    rot = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
    h.apply_transform(rot)
    vox = trimesh.util.concatenate([v, h]).voxelized(pitch=pitch).fill()
    mesh = vox.marching_cubes
    mesh.apply_transform(vox.transform)
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    return mesh


def test_planar_boundary_guard_skips_saddle_and_keeps_result(
):
    """サドル状(非平面)の境界を持つ十字モックでは品質ガードが発動して
    ペアがスキップされ、結果が平面化OFFと完全に一致することを検証する。"""
    mesh = make_cross_mesh()
    seeds = np.array([[0.0, 0.0, 29.0], [29.0, 0.0, 0.0]])

    report: list = []
    labels_on = labels_from_seeds(
        mesh, seeds, planar_boundaries=True, planar_fit_out=report
    )
    labels_off = labels_from_seeds(mesh, seeds, planar_boundaries=False)

    assert len(report) >= 1
    assert not any(r["applied"] for r in report)  # 全ペアがガードでスキップ
    assert np.array_equal(labels_on, labels_off)  # 結果はビット単位で不変
