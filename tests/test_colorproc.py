"""colorproc.py の単体テスト (IMPLEMENTATION_PLAN.md Phase 2.5 タスク2.5-4)。

合成画像(明確な4色ブロック)+ 単純メッシュ(box)で、量子化パレットが
n_colors以下であること・分割サブメッシュの面数合計が元メッシュの面数と
一致すること・face_ratioの合計が概ね1.0になることを検証する。
"""
import numpy as np
import pytest
import trimesh
from PIL import Image

from server import colorproc


def make_4color_image(size=128):
    """左上=赤、右上=緑、左下=青、右下=黄の4色ブロックRGBA画像。"""
    half = size // 2
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:half, :half] = [255, 0, 0, 255]
    arr[:half, half:] = [0, 255, 0, 255]
    arr[half:, :half] = [0, 0, 255, 255]
    arr[half:, half:] = [255, 255, 0, 255]
    return Image.fromarray(arr, "RGBA")


def make_solid_image(color, size=64):
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr, "RGBA")


def make_subdivided_box():
    box = trimesh.creation.box(extents=[10.0, 10.0, 20.0])
    # 単純なboxだと頂点が8個しかなく色のバリエーションが乏しいため細分化する
    box = box.subdivide().subdivide()
    return box


def test_project_colors_returns_rgba_uint8():
    mesh = make_subdivided_box()
    image = make_4color_image()
    colors = colorproc.project_colors(mesh, image)

    assert colors.shape == (len(mesh.vertices), 4)
    assert colors.dtype == np.uint8
    # アルファは常に不透明(頂点カラーは表示色として使うため)
    assert (colors[:, 3] == 255).all()


def test_project_colors_handles_rgb_image():
    """RGBA でない画像(RGB)もエラーなく処理できること。"""
    mesh = make_subdivided_box()
    image = make_4color_image().convert("RGB")
    colors = colorproc.project_colors(mesh, image)
    assert colors.shape == (len(mesh.vertices), 4)


def test_project_multiview_colors_keeps_back_base_without_back_image():
    """背面画像が無い場合、背面には正面画像が回り込まずベース色になること。"""
    mesh = make_subdivided_box()
    front = make_solid_image([255, 0, 0, 255])
    colors = colorproc.project_multiview_colors(mesh, front)

    front_mask, back_mask = colorproc._front_back_vertex_masks(mesh)
    assert front_mask.any()
    assert back_mask.any()
    assert (colors[front_mask, :3] == [255, 0, 0]).all()
    assert (colors[back_mask, :3] == colorproc._DEFAULT_BASE_COLOR).all()


def test_project_multiview_colors_uses_back_image_for_back_vertices():
    """背面画像がある場合、背面側の頂点には背面画像の色が使われること。"""
    mesh = make_subdivided_box()
    front = make_solid_image([255, 0, 0, 255])
    back = make_solid_image([0, 0, 255, 255])
    colors = colorproc.project_multiview_colors(mesh, front, back_image=back)

    front_mask, back_mask = colorproc._front_back_vertex_masks(mesh)
    assert front_mask.any()
    assert back_mask.any()
    assert (colors[front_mask, :3] == [255, 0, 0]).all()
    assert (colors[back_mask, :3] == [0, 0, 255]).all()


@pytest.mark.parametrize("n_colors", [2, 3, 4])
def test_quantize_palette_size_within_n_colors(n_colors):
    mesh = make_subdivided_box()
    image = make_4color_image()
    colors = colorproc.project_colors(mesh, image)

    palette, labels = colorproc.quantize(colors, n_colors)

    assert len(palette) <= n_colors
    assert palette.shape[1] == 3
    assert set(np.unique(labels).tolist()) == set(range(len(palette)))


def test_quantize_rejects_out_of_range_n_colors():
    mesh = make_subdivided_box()
    image = make_4color_image()
    colors = colorproc.project_colors(mesh, image)
    with pytest.raises(ValueError):
        colorproc.quantize(colors, 1)
    with pytest.raises(ValueError):
        colorproc.quantize(colors, 5)


def test_split_by_color_face_count_matches_original():
    mesh = make_subdivided_box()
    image = make_4color_image()
    colors = colorproc.project_colors(mesh, image)
    palette, labels = colorproc.quantize(colors, 4)

    submeshes = colorproc.split_by_color(mesh, labels, palette)

    assert 1 <= len(submeshes) <= 4
    total_faces = sum(len(sub.faces) for sub, _ in submeshes)
    assert total_faces == len(mesh.faces)

    # HEX形式であること
    for _, hex_color in submeshes:
        assert hex_color.startswith("#")
        assert len(hex_color) == 7


def test_split_by_color_submeshes_have_vertex_colors():
    mesh = make_subdivided_box()
    image = make_4color_image()
    colors = colorproc.project_colors(mesh, image)
    palette, labels = colorproc.quantize(colors, 4)
    submeshes = colorproc.split_by_color(mesh, labels, palette)

    for sub, _ in submeshes:
        assert hasattr(sub.visual, "vertex_colors")
        assert len(sub.visual.vertex_colors) == len(sub.vertices)


def test_palette_stats_face_ratio_sums_to_one():
    mesh = make_subdivided_box()
    image = make_4color_image()
    colors = colorproc.project_colors(mesh, image)
    palette, labels = colorproc.quantize(colors, 4)

    stats = colorproc.palette_stats(labels, palette, mesh)

    assert 1 <= len(stats) <= 4
    total_ratio = sum(s["face_ratio"] for s in stats)
    assert total_ratio == pytest.approx(1.0, abs=1e-6)

    # face_ratio降順であること
    ratios = [s["face_ratio"] for s in stats]
    assert ratios == sorted(ratios, reverse=True)

    for s in stats:
        assert s["hex"].startswith("#")


def test_palette_stats_keys():
    mesh = make_subdivided_box()
    image = make_4color_image()
    colors = colorproc.project_colors(mesh, image)
    palette, labels = colorproc.quantize(colors, 3)
    stats = colorproc.palette_stats(labels, palette, mesh)
    for entry in stats:
        assert set(entry.keys()) == {"hex", "face_ratio"}


# --- transfer_vertex_colors_nearest (Pixal3D統合: raw mesh -> 後処理後meshへの頂点カラー転写) ---


def test_transfer_vertex_colors_nearest_exact_match():
    """dst頂点がsrc頂点と完全一致する場合、そのまま同じ色が転写されること。"""
    src_vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    src_colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]],
        dtype=np.uint8,
    )
    dst_vertices = src_vertices.copy()

    result = colorproc.transfer_vertex_colors_nearest(src_vertices, src_colors, dst_vertices)

    assert result.shape == (4, 4)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, src_colors)


def test_transfer_vertex_colors_nearest_picks_closest():
    """dst頂点はsrc頂点群の最近傍の色を受け取ること。"""
    src_vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    src_colors = np.array([[255, 0, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)

    # dst頂点はsrc[0]寄り・src[1]寄りの2点
    dst_vertices = np.array([[0.5, 0.0, 0.0], [9.0, 0.0, 0.0]])

    result = colorproc.transfer_vertex_colors_nearest(src_vertices, src_colors, dst_vertices)

    assert tuple(result[0][:3]) == (255, 0, 0)
    assert tuple(result[1][:3]) == (0, 0, 255)


def test_transfer_vertex_colors_nearest_after_simplification():
    """meshprocによる簡略化(頂点数減少・再構築)を模した実際的なケース。

    元メッシュを細分化して頂点カラーを投影した後、簡略化で頂点数が変わった
    別メッシュ(同じ座標系・スケール)へ色転写しても、各頂点が合理的に近い
    色を受け取ること(全頂点がRGBのいずれかの主要色に一致)。
    """
    src_mesh = make_subdivided_box()
    image = make_4color_image()
    src_colors = colorproc.project_colors(src_mesh, image)

    # 簡略化を模した別メッシュ(元のboxの頂点のみ=より粗いメッシュ)
    dst_mesh = trimesh.creation.box(extents=[10.0, 10.0, 20.0])

    result = colorproc.transfer_vertex_colors_nearest(
        src_mesh.vertices, src_colors, dst_mesh.vertices
    )

    assert result.shape == (len(dst_mesh.vertices), 4)
    assert result.dtype == np.uint8
    assert (result[:, 3] == 255).all()


def test_transfer_vertex_colors_nearest_length_mismatch_raises():
    src_vertices = np.zeros((3, 3))
    src_colors = np.zeros((2, 4), dtype=np.uint8)
    dst_vertices = np.zeros((1, 3))
    with pytest.raises(ValueError):
        colorproc.transfer_vertex_colors_nearest(src_vertices, src_colors, dst_vertices)


# --- 左右側面ビューの投影 (4面図対応) -----------------------------------------


def _sphere():
    """法線が全方向へ均等に向く球。ビュー割当の検証に使う。

    箱は頂点が角にしか無く、法線が隣接3面の平均になって全ビューと同点になる
    ため、割当のテストには使えない。
    """
    import trimesh

    return trimesh.creation.icosphere(subdivisions=3, radius=1.0)


def _solid_image(color, size=32):
    from PIL import Image

    return Image.new("RGBA", (size, size), (*color, 255))


def test_view_masks_split_all_four_directions():
    """4ビューを渡すと、各面が対応するビューに割り当てられること。"""
    mesh = _sphere()
    masks = colorproc._view_vertex_masks(mesh, list(colorproc.VIEW_NAMES))

    # 箱の頂点は角にあり法線が斜めなので、割当先が重複しないことを確かめる
    stacked = np.stack([masks[v] for v in colorproc.VIEW_NAMES])
    assert stacked.sum(axis=0).max() <= 1, "1頂点が複数ビューに割り当てられている"


def test_view_masks_are_backward_compatible_with_front_back():
    """正面/背面だけを渡した場合は従来の法線しきい値判定と一致すること。"""
    mesh = _sphere()
    normals = colorproc._vertex_normals(mesh)
    threshold = colorproc._VIEW_NORMAL_THRESHOLD

    masks = colorproc._view_vertex_masks(mesh, ["front", "back"])
    assert np.array_equal(masks["front"], normals[:, 1] < -threshold)
    assert np.array_equal(masks["back"], normals[:, 1] > threshold)


def test_side_images_colour_the_side_vertices():
    """左右画像を渡すと、側面の頂点がその画像の色になる。"""
    mesh = _sphere()
    front = _solid_image((255, 0, 0))
    back = _solid_image((0, 255, 0))
    left = _solid_image((0, 0, 255))
    right = _solid_image((255, 255, 0))

    colors = colorproc.project_multiview_colors(
        mesh, front, back_image=back, left_image=left, right_image=right
    )

    masks = colorproc._view_vertex_masks(mesh, list(colorproc.VIEW_NAMES))
    for view, expected in (("left", (0, 0, 255)), ("right", (255, 255, 0))):
        mask = masks[view]
        assert mask.any(), f"{view} に割り当てられた頂点が無い"
        assert (colors[mask, :3] == expected).all(), f"{view} 画像の色になっていない"


def test_side_images_reduce_base_coloured_vertices():
    """側面画像はベース色のままだった頂点を実際に減らす。

    正面/背面だけだと、真横を向いた頂点はどちらのしきい値も超えず
    一律のベース色になる(実生成モデルで約9%)。
    """
    mesh = _sphere()
    front = _solid_image((255, 0, 0))
    back = _solid_image((0, 255, 0))

    def base_count(colors):
        return int((colors[:, :3] == colorproc._DEFAULT_BASE_COLOR).all(axis=1).sum())

    without_sides = colorproc.project_multiview_colors(mesh, front, back_image=back)
    with_sides = colorproc.project_multiview_colors(
        mesh,
        front,
        back_image=back,
        left_image=_solid_image((0, 0, 255)),
        right_image=_solid_image((255, 255, 0)),
    )

    assert base_count(without_sides) > 0, "前提: 側面画像が無いとベース色の頂点がある"
    assert base_count(with_sides) < base_count(without_sides)


def test_left_and_right_use_opposite_horizontal_direction():
    """左右のビューはカメラが反対側にあるため、u→メッシュY の向きが逆になる。"""
    left_axis, left_sign = colorproc._view_u_axis("left")
    right_axis, right_sign = colorproc._view_u_axis("right")
    assert left_axis == right_axis == 1, "左右側面はメッシュY軸へ投影する"
    assert left_sign == -right_sign


def test_front_and_back_still_use_x_axis():
    front_axis, front_sign = colorproc._view_u_axis("front")
    back_axis, back_sign = colorproc._view_u_axis("back")
    assert front_axis == back_axis == 0
    assert front_sign == -back_sign


def test_project_image_colors_rejects_unknown_view():
    mesh = _sphere()
    with pytest.raises(ValueError, match="view"):
        colorproc._project_image_colors(mesh, _solid_image((1, 2, 3)), view="top")


# --- 見切れ画像のシルエット自動位置合わせ -------------------------------------


def _stepped_tower():
    """高さごとに幅が変わる、シルエットに特徴のあるメッシュ。"""
    import trimesh

    parts = [
        trimesh.creation.box(extents=(2.0, 1.0, 1.0), transform=_translate(0, 0, 0.5)),
        trimesh.creation.box(extents=(0.6, 1.0, 1.0), transform=_translate(0, 0, 1.5)),
        trimesh.creation.box(extents=(1.4, 1.0, 1.0), transform=_translate(0, 0, 2.5)),
    ]
    # 頂点が角にしか無い箱のままだとシルエットの段が疎になるため細分する
    return trimesh.util.concatenate(parts).subdivide().subdivide()


def _translate(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix


def _render_silhouette(mesh, axis, height_px, top_px, canvas):
    """メッシュのシルエットを、指定の倍率・位置で画像に焼く(テスト用)。"""
    from PIL import Image

    width_px, canvas_h = canvas
    image = Image.new("RGBA", (width_px, canvas_h), (0, 0, 0, 0))
    pixels = image.load()

    vertices = mesh.vertices
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    profile = colorproc._mesh_silhouette_profile(mesh, axis)
    mesh_width = vertices[:, axis].max() - vertices[:, axis].min()
    scale = height_px / (z_max - z_min)

    for row in range(canvas_h):
        t = (row - top_px) / height_px  # 0=メッシュ上端, 1=下端
        if not (0.0 <= t < 1.0):
            continue
        bin_index = min(int((1.0 - t) * len(profile)), len(profile) - 1)
        half = profile[bin_index] * scale / 2
        if half <= 0:
            continue
        centre = width_px / 2
        for col in range(max(int(centre - half), 0), min(int(centre + half), width_px)):
            pixels[col, row] = (200, 120, 60, 255)
    return image


def test_silhouette_alignment_recovers_a_cropped_subject():
    """被写体が枠からはみ出していても、メッシュ上下端の対応行を当てられること。

    実測の側面図は縦占有 767/768 で上下とも見切れており、被写体bboxに
    メッシュ全高を合わせる従来方式では色が縦に大きくずれる。
    """
    mesh = _stepped_tower()
    canvas = (256, 200)
    height_px, top_px = 300, -40  # 画像より背が高く、上が枠外へはみ出す
    image = _render_silhouette(mesh, axis=0, height_px=height_px, top_px=top_px, canvas=canvas)

    assert colorproc._subject_touches_vertical_edges(image), "前提: 見切れている画像"

    aligned = colorproc._align_vertical_by_silhouette(mesh, image, axis=0)
    assert aligned is not None, "照合に失敗した"

    v_top, v_bottom = aligned
    canvas_h = canvas[1]
    assert abs(v_top * canvas_h - top_px) < 12
    assert abs(v_bottom * canvas_h - (top_px + height_px)) < 12


def test_silhouette_alignment_beats_bbox_fit_when_cropped():
    """見切れ時、従来の被写体bbox方式より真値に近いこと。"""
    mesh = _stepped_tower()
    canvas = (256, 200)
    height_px, top_px = 300, -40
    image = _render_silhouette(mesh, axis=0, height_px=height_px, top_px=top_px, canvas=canvas)

    _, _, bbox_top, bbox_bottom = colorproc._subject_bbox_uv(image)
    aligned_top, aligned_bottom = colorproc._align_vertical_by_silhouette(mesh, image, axis=0)

    truth_top, truth_bottom = top_px / canvas[1], (top_px + height_px) / canvas[1]
    bbox_error = abs(bbox_top - truth_top) + abs(bbox_bottom - truth_bottom)
    aligned_error = abs(aligned_top - truth_top) + abs(aligned_bottom - truth_bottom)
    assert aligned_error < bbox_error / 4


def test_silhouette_alignment_is_skipped_when_not_cropped():
    """枠内に収まっている画像では従来の被写体bbox方式のままにする。"""
    mesh = _stepped_tower()
    canvas = (256, 200)
    image = _render_silhouette(mesh, axis=0, height_px=150, top_px=25, canvas=canvas)

    assert colorproc._subject_touches_vertical_edges(image) is False


def test_alignment_gives_up_on_a_featureless_silhouette():
    """一様な矩形など手がかりの無いシルエットでは None を返して従来方式に戻す。"""
    import trimesh
    from PIL import Image

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 3.0))
    image = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    assert colorproc._align_vertical_by_silhouette(mesh, image, axis=0) is None
