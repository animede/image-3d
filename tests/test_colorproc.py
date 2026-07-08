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
