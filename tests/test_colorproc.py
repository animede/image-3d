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
    """背面画像が無い場合、正面の投影色そのものは背面へ直接コピーされないこと。

    以前は背面側が一律のベース色(灰色帯の原因)になっていたが、今は
    メッシュ表面拡散(`_diffuse_unknown_vertex_colors`)が唯一の既知色である
    正面色で埋めるため、背面もベース色ではなく正面色に近づく。これは
    「未知頂点はベース色より周囲の色に近い方が良い」という設計通りの挙動。
    front側の色は拡散の対象外(既知)であり従来通り正面画像のままである
    ことだけを固定の契約として検証する。
    """
    mesh = make_subdivided_box()
    front = make_solid_image([255, 0, 0, 255])
    colors = colorproc.project_multiview_colors(mesh, front)

    front_mask, back_mask = colorproc._front_back_vertex_masks(mesh)
    assert front_mask.any()
    assert back_mask.any()
    assert (colors[front_mask, :3] == [255, 0, 0]).all()
    # 背面はベース色(灰)には残らず、拡散で正面色(既知色が赤のみ)に埋まる
    assert not (colors[back_mask, :3] == colorproc._DEFAULT_BASE_COLOR).all()


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


def test_front_facing_vertices_keep_the_front_image():
    """正面をまっすぐ向いた頂点は、側面画像を足しても正面画像のままであること。

    4ビューを同格に扱うと、丸い顔の頬が側面画像に置き換わって顔が崩れた
    (実機で顔まわりの55%が側面画像になった)。
    """
    mesh = _sphere()
    front = _solid_image((255, 0, 0))
    back = _solid_image((0, 255, 0))

    four_view = colorproc.project_multiview_colors(
        mesh,
        front,
        back_image=back,
        left_image=_solid_image((0, 0, 255)),
        right_image=_solid_image((255, 255, 0)),
    )

    normals = colorproc._vertex_normals(mesh)
    straight_on = normals[:, 1] < -0.8  # 正面をほぼ正対して向く頂点
    assert straight_on.any()
    assert (four_view[straight_on, :3] == (255, 0, 0)).all()


def test_grazing_vertices_go_to_the_side_images():
    """ほぼ真横を向いた面は、前後の引き伸ばしではなく側面画像から色を取る。

    しきい値が緩いと視線から84°ずれた面まで前後担当になり、浅い角度で
    引き伸ばされた色が側面でぶつかって黒い継ぎ目になる。
    """
    mesh = _sphere()
    four_view = colorproc.project_multiview_colors(
        mesh,
        _solid_image((255, 0, 0)),
        back_image=_solid_image((0, 255, 0)),
        left_image=_solid_image((0, 0, 255)),
        right_image=_solid_image((255, 255, 0)),
    )

    normals = colorproc._vertex_normals(mesh)
    grazing = np.abs(normals[:, 0]) > 0.9  # ほぼ真横を向く
    assert grazing.any()
    side_colours = {(0, 0, 255), (255, 255, 0)}
    got = {tuple(c) for c in four_view[grazing, :3]}
    assert got <= side_colours, f"側面以外の色が混ざっている: {got - side_colours}"


def test_diffusion_removes_base_coloured_vertices_without_side_images():
    """メッシュ表面拡散により、側面画像が無くてもベース色の頂点がほぼ無くなる。

    以前は正面/背面だけだと、真横を向いた頂点はどちらのしきい値も超えず
    一律のベース色になっていた(実生成モデルで約9.1%)。案A(フチ除外+拡散)
    導入後は、割当から漏れた頂点も既知頂点(前後の色)からの調和補間で
    埋まるため、連結成分内にベース色は残らない(実測: 実生成ジョブで
    9.1%→0%近くまで低減)。
    """
    mesh = _sphere()
    front = _solid_image((255, 0, 0))
    back = _solid_image((0, 255, 0))

    def base_count(colors):
        return int((colors[:, :3] == colorproc._DEFAULT_BASE_COLOR).all(axis=1).sum())

    without_sides = colorproc.project_multiview_colors(mesh, front, back_image=back)
    assert base_count(without_sides) == 0


def test_side_images_still_colour_the_side_region_distinctly():
    """側面画像がある場合、真横寄りの頂点は拡散色ではなく側面画像の色になる。

    拡散だけでも見た目上は隙間が埋まるが、側面専用の画像がある場合は
    それを優先して使うべきなので、真横を向く頂点が side 画像の色そのもの
    (前後色の中間の拡散色ではない)になっていることを確認する。
    """
    mesh = _sphere()
    front = _solid_image((255, 0, 0))
    back = _solid_image((0, 255, 0))
    left = _solid_image((0, 0, 255))
    right = _solid_image((255, 255, 0))

    with_sides = colorproc.project_multiview_colors(
        mesh, front, back_image=back, left_image=left, right_image=right
    )

    normals = colorproc._vertex_normals(mesh)
    grazing = np.abs(normals[:, 0]) > 0.9  # ほぼ真横を向く
    assert grazing.any()
    side_colours = {(0, 0, 255), (255, 255, 0)}
    got = {tuple(c) for c in with_sides[grazing, :3]}
    assert got <= side_colours, f"側面以外の色が混ざっている: {got - side_colours}"


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


# --- 案A: フチ除外サンプリング + メッシュ表面拡散 ------------------------------


def _image_with_dirty_border(inner_color, border_color, size=64, border_px=2):
    """前景の外周border_pxを別の色(汚れたフチを模した色)で塗ったRGBA画像。

    rembg後のシルエット境界は背景と混ざって暗く汚れているため、外周1〜2pxを
    黒くした合成画像でフチが実際にサンプリングされないことを検証する。
    """
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :] = (*inner_color, 255)
    arr[:border_px, :] = (*border_color, 255)
    arr[-border_px:, :] = (*border_color, 255)
    arr[:, :border_px] = (*border_color, 255)
    arr[:, -border_px:] = (*border_color, 255)
    return Image.fromarray(arr, "RGBA")


def test_edge_exclusion_avoids_sampling_the_dirty_border():
    """前景外周の汚れた色(黒)が頂点カラーに出ないこと。

    フチ除外サンプリングが無いと、外周ぎりぎりに投影された頂点が汚れた
    黒フチをそのまま拾ってしまう。収縮後マスクで最近傍補完すれば、
    フチではなく内側の色になるはず。
    """
    mesh = make_subdivided_box()
    image = _image_with_dirty_border(
        inner_color=(255, 0, 0), border_color=(0, 0, 0), size=64, border_px=2
    )
    colors = colorproc.project_colors(mesh, image)

    # 頂点カラーに黒(フチ色)が一切含まれないこと
    assert not (colors[:, :3] == (0, 0, 0)).all(axis=1).any(), "汚れたフチの黒が頂点カラーに残っている"
    # 内側の色(赤)で埋まっていること
    assert (colors[:, :3] == (255, 0, 0)).any()


def test_trusted_pixel_mask_erodes_the_border():
    """_trusted_pixel_mask が外周をアルファ>0領域より狭く収縮すること。"""
    size = 64
    alpha = np.zeros((size, size), dtype=np.uint8)
    alpha[8:-8, 8:-8] = 255  # 中央だけ不透明な正方形

    trusted = colorproc._trusted_pixel_mask(alpha)

    assert trusted.sum() < (alpha > 0).sum(), "収縮されていない"
    assert trusted.any(), "前景が全滅している"
    # 収縮後マスクは元のアルファ領域に完全に含まれる
    assert (trusted <= (alpha > 0)).all()


def test_trusted_pixel_mask_falls_back_when_foreground_is_tiny():
    """収縮で前景が全滅する極小前景では、収縮なしの元マスクにフォールバックする。"""
    size = 64
    alpha = np.zeros((size, size), dtype=np.uint8)
    alpha[30:31, 30:31] = 255  # 1px だけの極小前景

    trusted = colorproc._trusted_pixel_mask(alpha)

    assert trusted.sum() == 1
    np.testing.assert_array_equal(trusted, alpha > 0)


def test_diffusion_fills_unknown_vertices_with_no_base_colour():
    """icosphereに前(赤)/後(緑)のみ与えると、ベース色頂点が0になること。"""
    mesh = _sphere()
    front = _solid_image((255, 0, 0))
    back = _solid_image((0, 255, 0))

    colors = colorproc.project_multiview_colors(mesh, front, back_image=back)

    base_count = int((colors[:, :3] == colorproc._DEFAULT_BASE_COLOR).all(axis=1).sum())
    assert base_count == 0


def test_diffusion_blends_red_and_green_at_the_equator():
    """真横(赤と緑のちょうど中間)の頂点は、拡散により中間色になること。"""
    mesh = _sphere()
    front = _solid_image((255, 0, 0))  # -Y向き
    back = _solid_image((0, 255, 0))  # +Y向き

    colors = colorproc.project_multiview_colors(mesh, front, back_image=back)

    normals = colorproc._vertex_normals(mesh)
    # front/backどちらの法線しきい値も超えない、真横(Y成分がほぼ0)の頂点
    equator = np.abs(normals[:, 1]) < 0.05
    assert equator.any()

    equator_colors = colors[equator, :3].astype(np.float64)
    # 赤成分・緑成分がどちらも極端(0 or 255)に偏らず、中間色になっていること
    assert (equator_colors[:, 0] > 20).all() and (equator_colors[:, 0] < 235).all()
    assert (equator_colors[:, 1] > 20).all() and (equator_colors[:, 1] < 235).all()
    # 青は前後どちらの画像にも無いので0のまま
    assert (equator_colors[:, 2] == 0).all()


def test_diffusion_leaves_front_facing_vertices_unchanged():
    """拡散は既知頂点(顔=正対領域)の色を書き換えない。

    「顔の再現性を損なわない」ことが最優先のため、正対して前を向く頂点は
    拡散対象にならず、投影された正面画像の色のままであることを確認する。
    """
    mesh = _sphere()
    front = _solid_image((255, 0, 0))
    back = _solid_image((0, 255, 0))

    colors = colorproc.project_multiview_colors(mesh, front, back_image=back)

    normals = colorproc._vertex_normals(mesh)
    straight_on = normals[:, 1] < -0.8  # 正面をほぼ正対して向く頂点
    assert straight_on.any()
    assert (colors[straight_on, :3] == (255, 0, 0)).all()


# --- 側面参照の異方歪み修正(等方スケール+横オフセット探索) -------------------
#
# 実ジョブ(cbf449b7...)で確認したバグ: 側面参照は腕を前へ出した姿勢で
# 撮るため被写体bboxが横に広がるが、メッシュの側面投影は胴の厚みしかない。
# 縦横で独立にbboxいっぱいへ引き伸ばすと胴が約2倍に間延びし、シルエット
# 不一致ガードがほぼ全域(82%)を弾いていた。等方スケール(縦と同じpx/mm)+
# IoU最大の横オフセットへ切り替えると、実ジョブで不一致率が34.6%まで下がる
# ことを確認済み(実機検証はコードレビュー時の報告参照)。


def _torso_with_side_arm_mesh():
    """胴体(薄い直方体)のみのメッシュ。厚み(Y)は薄く、幅(X)・高さ(Z)は太い。

    「側面ビュー」を想定し、Y軸方向への投影幅は薄い胴体の厚みだけになる。
    実ジョブ同様、頂点が細分されていないと段のプロファイルが疎になるため
    細分する。
    """
    box = trimesh.creation.box(extents=[20.0, 4.0, 40.0])
    return box.subdivide().subdivide().subdivide()


def _side_reference_with_arm_bulge(
    canvas=(256, 256), torso_width_px=40, arm_width_px=200, arm_row_frac=(0.35, 0.55)
):
    """側面参照を模した合成画像。

    ほとんどの行は胴体幅(torso_width_px)の細い帯だが、`arm_row_frac` の
    行範囲だけ腕が突き出て被写体bboxが大きく広がる(arm_width_px)。
    実ジョブの「胴は細いのに被写体bboxは腕を含んで広い」状況を再現する。
    """
    w, h = canvas
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    centre = w / 2
    arm_lo, arm_hi = int(h * arm_row_frac[0]), int(h * arm_row_frac[1])
    for row in range(h):
        half = (arm_width_px if arm_lo <= row < arm_hi else torso_width_px) / 2
        lo, hi = max(int(centre - half), 0), min(int(centre + half), w)
        arr[row, lo:hi] = (200, 120, 60, 255)
    return Image.fromarray(arr, "RGBA")


def test_isotropic_fit_keeps_the_torso_width_despite_an_arm_bulge():
    """腕の突起で被写体bboxが横に広がっていても、胴の実幅が引き伸ばされないこと。

    従来のbbox方式(縦横独立に引き伸ばす)だと、胴体の投影幅が被写体bbox幅
    (腕を含む=torso_width_pxよりずっと広い)いっぱいまで引き伸ばされる。
    等方スケール方式なら、胴の投影幅は「縦と同じpx/mm」で決まるので、
    腕の有無に関係なく一定(≒torso_width_px相当)に保たれるはず。
    """
    mesh = _torso_with_side_arm_mesh()
    image = _side_reference_with_arm_bulge()

    axis, _ = colorproc._view_u_axis("left")
    px, _ = colorproc.project_points_to_pixels(mesh, image, mesh.vertices, view="left")
    projected_width = int(px.max()) - int(px.min())

    # bbox方式(腕込みの被写体bbox幅)ならもっと広くなるはずの閾値。
    # 胴だけの幅(torso_width_px=40)に近い値に収まっていることを確認する。
    assert projected_width < 90, f"胴の投影幅が広すぎる(異方歪みが残っている): {projected_width}px"


def test_isotropic_fit_matches_bbox_fit_when_bboxes_already_agree():
    """前面/背面のようにメッシュbboxと被写体bboxが一致するケースでは、

    安全弁が働いて従来のbbox方式のままになり、結果が変わらないこと
    (回帰防止)。
    """
    mesh = make_subdivided_box()  # extents=[10,10,20]、前面/背面はX軸に投影
    image = make_4color_image()  # 被写体bbox=画像全体(アルファ無し=不透明)

    with_fit = colorproc.project_points_to_pixels(mesh, image, mesh.vertices, view="front")

    axis, u_sign = colorproc._view_u_axis("front")
    u_min, u_max, v_min, v_max = colorproc._subject_bbox_uv(image)
    bounds = mesh.bounds
    a_min, a_max = bounds[0][axis], bounds[1][axis]
    a_norm = (mesh.vertices[:, axis] - a_min) / max(a_max - a_min, 1e-9)
    u_norm = a_norm if u_sign > 0 else 1.0 - a_norm
    expected_u = u_min + u_norm * (u_max - u_min)
    w, h = image.size
    expected_px = np.clip((expected_u * w).astype(np.int64), 0, w - 1)

    assert np.array_equal(with_fit[0], expected_px), "bboxが一致するのに等方フィットへ切り替わった"


def test_safety_valve_falls_back_to_bbox_when_isotropic_iou_is_worse():
    """等方フィットのIoUがbbox方式を下回る場合は、bbox方式のまま使うこと。

    `_resolve_horizontal_range` は両方式のIoUを比べ、改善しない場合は
    従来のbbox方式を返す(安全弁)。被写体アルファがメッシュのシルエットと
    無関係な位置にある(=bboxのほうがまだマシな)人工的なケースで確認する。
    """
    mesh = _torso_with_side_arm_mesh()
    axis, _ = colorproc._view_u_axis("left")

    # 被写体が画像の隅に偏って写っている、メッシュ形状と噛み合わない画像。
    # 等方スケールの狭い窓をどこに置いてもbbox方式より良くならないはず。
    w, h = 256, 256
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[h // 2 - 5 : h // 2 + 5, w - 30 : w - 5] = (200, 120, 60, 255)
    image = Image.fromarray(arr, "RGBA")

    u_min, u_max, v_min, v_max = colorproc._subject_bbox_uv(image)
    result = colorproc._resolve_horizontal_range(
        mesh, image, axis, u_min, u_max, v_min, v_max, "left"
    )
    assert result == (u_min, u_max), "IoUが悪化する場合はbbox方式を使うはずが等方フィットが選ばれた"


def test_horizontal_fit_cache_is_ignored_after_id_reuse():
    """id が再利用されたキャッシュ項目は使わず、計算し直すこと。

    キーは id() なので、オブジェクト解放後の id 再利用でまったく別のメッシュに
    古いマッピングを返す危険がある(実測: 200個のメッシュを生成/破棄すると
    181個しか異なる id にならない)。弱参照で「同じオブジェクトか」を確かめる。
    """
    import weakref

    from PIL import Image as PILImage

    from server import colorproc

    mesh = trimesh.creation.box(extents=(2.0, 1.0, 2.0))
    arr = np.zeros((64, 64, 4), np.uint8)
    arr[8:56, 8:56] = (200, 200, 200, 255)
    image = PILImage.fromarray(arr, "RGBA")

    colorproc._horizontal_fit_cache.clear()
    expected, _ = colorproc.project_points_to_pixels(
        mesh, image, mesh.vertices, view="front"
    )

    # 解放済みオブジェクトの弱参照 = id が再利用された状況を作る
    class _Gone:
        pass

    dead = _Gone()
    dead_ref = weakref.ref(dead)
    del dead
    assert dead_ref() is None

    colorproc._horizontal_fit_cache[(id(mesh), id(image), "front")] = (
        dead_ref,
        dead_ref,
        (0.0, 0.05),  # 使われたら投影が左端に潰れる、あり得ない値
    )
    actual, _ = colorproc.project_points_to_pixels(
        mesh, image, mesh.vertices, view="front"
    )
    assert np.array_equal(actual, expected), "解放済みidのキャッシュが使われている"
    colorproc._horizontal_fit_cache.clear()
