"""texgen アトラスを参照画像の全解像度で上書きする処理のテスト (server/texrefine.py)。

GPU不要。texgen の出力に見立てた「UV+平坦なテクスチャ付きの球」を作り、
正面から見える面だけが参照画像の色に置き換わることを確認する。
"""
import numpy as np
import pytest
import trimesh
from PIL import Image

from server import texrefine
from server.texture import sample_vertex_colors_from_texture

TEXGEN_GREY = 128
REFERENCE_RED = (220, 30, 40)


def _sphere_with_uv(texture_size: int = 512) -> trimesh.Trimesh:
    """球体に「面ごとのチャート」UVと単色テクスチャを付け、texgen 出力を模す。

    球面UV(緯度経度)にすると経度0°をまたぐ面のUV三角形がアトラス全幅に
    広がり、その面のサンプルが無関係なテクセルへ飛び散る。実際の texgen は
    xatlas で切り開いたチャートを並べるので、こちらに合わせて面ごとに
    独立したセルへ割り当てる。
    """
    mesh = trimesh.creation.uv_sphere(radius=50, count=[24, 24])
    mesh.unmerge_vertices()  # 面ごとに頂点を独立させ、チャートを切り離す

    n_faces = len(mesh.faces)
    grid = int(np.ceil(np.sqrt(n_faces)))
    cell = 1.0 / grid
    # セル内に収まる三角形(端に寄せるとチャート外周にかかるので余白を取る)
    corners = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8]]) * cell

    uv = np.zeros((len(mesh.vertices), 2))
    index = np.arange(n_faces)
    origin = np.column_stack([index % grid, index // grid]) * cell
    for k in range(3):
        uv[mesh.faces[:, k]] = origin + corners[k]

    texture = Image.new("RGB", (texture_size, texture_size), (TEXGEN_GREY,) * 3)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.SimpleMaterial(image=texture)
    )
    return mesh


def _circular_reference(size: int = 256, color=REFERENCE_RED) -> Image.Image:
    """球のシルエットに合わせた円形の参照画像(円の外は透明)。"""
    yy, xx = np.mgrid[0:size, 0:size]
    centre = (size - 1) / 2
    inside = (xx - centre) ** 2 + (yy - centre) ** 2 <= (size * 0.48) ** 2
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[inside, :3] = color
    arr[inside, 3] = 255
    return Image.fromarray(arr, "RGBA")


def _vertex_colors_by_facing(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """(正面を正対して向く頂点の色, 背面を向く頂点の色) を返す。"""
    colors = sample_vertex_colors_from_texture(mesh)[:, :3].astype(int)
    normals = np.asarray(mesh.vertex_normals)
    front = normals @ np.array([0.0, -1.0, 0.0])
    return colors[front > 0.9], colors[front < -0.9]


def test_front_facing_texels_take_the_reference_colour():
    """正面を正対して向くテクセルは、参照画像の色に置き換わる。"""
    mesh = _sphere_with_uv()
    stats = texrefine.refine_texture_with_references(mesh, {"front": _circular_reference()})

    assert stats.applied, stats.reason
    front, _ = _vertex_colors_by_facing(mesh)
    assert len(front) > 0
    assert np.allclose(front.mean(axis=0), REFERENCE_RED, atol=12)


def test_back_facing_texels_keep_the_texgen_colour():
    """参照が正面だけなら、背面は texgen の色のまま残す。"""
    mesh = _sphere_with_uv()
    texrefine.refine_texture_with_references(mesh, {"front": _circular_reference()})

    _, back = _vertex_colors_by_facing(mesh)
    assert len(back) > 0
    assert np.allclose(back.mean(axis=0), TEXGEN_GREY, atol=6)


def test_back_reference_paints_the_back():
    """背面参照を渡せば背面もその色になる。"""
    mesh = _sphere_with_uv()
    blue = (20, 40, 210)
    stats = texrefine.refine_texture_with_references(
        mesh,
        {"front": _circular_reference(), "back": _circular_reference(color=blue)},
    )

    assert stats.applied, stats.reason
    front, back = _vertex_colors_by_facing(mesh)
    assert np.allclose(front.mean(axis=0), REFERENCE_RED, atol=12)
    assert np.allclose(back.mean(axis=0), blue, atol=12)


def test_full_resolution_detail_survives():
    """512に潰れた texgen では出せない細かい模様が、参照から転写される。

    これがこの処理の存在理由なので、模様の本数で確かめる。
    """
    size = 512
    ref = np.asarray(_circular_reference(size)).copy()
    stripe = (np.arange(size) // 4) % 2 == 0
    ref[:, stripe, :3] = np.where(
        ref[:, stripe, 3:4] > 0, np.array([250, 250, 250], dtype=np.uint8), 0
    )
    mesh = _sphere_with_uv(texture_size=1024)
    stats = texrefine.refine_texture_with_references(
        mesh, {"front": Image.fromarray(ref, "RGBA")}
    )

    assert stats.applied, stats.reason
    front, _ = _vertex_colors_by_facing(mesh)
    # 白い縞(G=250)と赤い地(G=30)が両方残っているか。平均化されると中間に寄る。
    green = front[:, 1]
    assert (green > 200).sum() > 5, "白い縞が消えている"
    assert (green < 80).sum() > 5, "赤い地が消えている"


def test_dirty_silhouette_edge_is_not_sampled():
    """シルエット境界の汚れた画素は上書きに使わない(黒筋の原因)。"""
    size = 256
    ref = np.asarray(_circular_reference(size)).copy()
    opaque = ref[..., 3] > 0
    from scipy.ndimage import binary_erosion

    edge = opaque & ~binary_erosion(opaque, iterations=3)
    ref[edge, :3] = 0  # 黒く汚れたフチ

    mesh = _sphere_with_uv()
    texrefine.refine_texture_with_references(mesh, {"front": Image.fromarray(ref, "RGBA")})

    colors = sample_vertex_colors_from_texture(mesh)[:, :3].astype(int)
    normals = np.asarray(mesh.vertex_normals)
    refined = colors[normals @ np.array([0.0, -1.0, 0.0]) > 0.9]
    assert refined.max() > 100, "フチの黒がそのまま乗っている"
    assert np.allclose(refined.mean(axis=0), REFERENCE_RED, atol=12)


def _two_plates_mesh(texture_size: int = 256) -> trimesh.Trimesh:
    """正面(-Y)を向く2枚の板。小さい板が大きい板の手前(帽子の垂れの模型)。

    どちらも法線は正面ビューへ正対しているので、可視性を見ない投影では
    後ろの板も参照画像(=手前の板を覆う色)を拾ってしまう。
    UVは左半分=後ろの板、右半分=手前の板に分ける。
    """
    # 後ろの大きい板 (y=+10)、手前の小さい板 (y=0)。Z-up、正面 -Y。
    back = np.array([[-50, 10, -50], [50, 10, -50], [50, 10, 50], [-50, 10, 50]], float)
    front = np.array([[-20, 0, -20], [20, 0, -20], [20, 0, 20], [-20, 0, 20]], float)
    vertices = np.vstack([back, front])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    uv = np.array(
        [[0.05, 0.05], [0.45, 0.05], [0.45, 0.95], [0.05, 0.95],
         [0.55, 0.05], [0.95, 0.05], [0.95, 0.95], [0.55, 0.95]]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    texture = Image.new("RGB", (texture_size, texture_size), (TEXGEN_GREY,) * 3)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.SimpleMaterial(image=texture)
    )
    return mesh


def test_occluded_surfaces_are_not_painted_from_the_reference():
    """手前の板に隠れた面は、正面を向いていても参照から塗らない。

    これを怠ると、帽子の垂れや耳の裏に隠れた頭側面が、そこを覆っている
    帽子の赤・耳の黒を拾って筋になる(実ジョブ46b64850で確認した欠陥)。
    """
    mesh = _two_plates_mesh()
    ref = Image.new("RGBA", (256, 256), (*REFERENCE_RED, 255))
    stats = texrefine.refine_texture_with_references(mesh, {"front": ref})

    assert stats.applied, stats.reason
    assert stats.occluded_sample_ratio > 0.0
    tex = np.asarray(texrefine._extract_texture_image(mesh.visual).convert("RGB")).astype(int)
    h, w = tex.shape[:2]
    front_plate = tex[h // 2, int(w * 0.75)]  # 手前の板の中央
    hidden_centre = tex[h // 2, int(w * 0.25)]  # 後ろの板のうち隠れている中央部
    assert np.allclose(front_plate, REFERENCE_RED, atol=10), "手前の板が塗られていない"
    assert np.allclose(hidden_centre, TEXGEN_GREY, atol=10), "隠れた面に参照色が乗っている"


def test_visible_parts_of_a_partially_occluded_surface_are_still_painted():
    """部分的に隠れた面でも、見えている部分は参照から塗る。"""
    mesh = _two_plates_mesh()
    ref = Image.new("RGBA", (256, 256), (*REFERENCE_RED, 255))
    texrefine.refine_texture_with_references(mesh, {"front": ref})

    tex = np.asarray(texrefine._extract_texture_image(mesh.visual).convert("RGB")).astype(int)
    h, w = tex.shape[:2]
    # 後ろの板の外周(手前の板 40/100 の外側)は見えている。
    # UV左半分の端 (u=0.08 -> 板の左端付近) をサンプルする。
    visible_edge = tex[h // 2, int(w * 0.08)]
    assert np.allclose(visible_edge, REFERENCE_RED, atol=10), "見えている外周が塗られていない"


def test_grazing_angles_are_left_to_texgen():
    """視線に対し浅い角度の面は上書きしない(投影が伸びて信用できないため)。"""
    assert texrefine._blend_weight(np.array([0.0]))[0] == 0.0
    assert texrefine._blend_weight(np.array([texrefine.MIN_CONFIDENCE_DOT]))[0] == 0.0
    assert texrefine._blend_weight(np.array([1.0]))[0] == 1.0
    mid = texrefine._blend_weight(
        np.array([(texrefine.MIN_CONFIDENCE_DOT + texrefine.FULL_CONFIDENCE_DOT) / 2])
    )[0]
    assert 0.0 < mid < 1.0


@pytest.mark.parametrize(
    "references, fragment",
    [({}, "参照画像"), ({"nonsense": None}, "参照画像")],
)
def test_declines_without_usable_references(references, fragment):
    stats = texrefine.refine_texture_with_references(_sphere_with_uv(), references)
    assert stats.applied is False
    assert fragment in stats.reason


def test_declines_when_the_mesh_has_no_texture():
    mesh = trimesh.creation.uv_sphere(radius=50, count=[16, 16])
    stats = texrefine.refine_texture_with_references(mesh, {"front": _circular_reference()})
    assert stats.applied is False
    assert "UV" in stats.reason


def _eye_fixture(synth_eye_x: int, ref_eye_x: int, size: int = 256):
    """「メッシュ側の目」と「参照側の目」の位置が異なる合成ビュー/参照を作る。"""
    yy, xx = np.mgrid[0:size, 0:size]

    def disc(cx, cy, r):
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

    synth = np.full((size, size, 3), 255, np.uint8)
    synth[disc(synth_eye_x, 128, 14)] = 20
    mask = disc(128, 128, 110)

    ref = np.full((size, size, 4), 255, np.uint8)
    ref[disc(128, 128, 110), :3] = (230, 200, 170)  # 毛
    ref[disc(ref_eye_x, 128, 14), :3] = 20  # 目
    return synth, mask, Image.fromarray(ref, "RGBA")


def test_reference_features_move_rigidly_to_the_mesh_position():
    """参照の目は形のままメッシュ側の位置へ動き、元の位置は毛で埋まる。

    ずれ(10px/256=3.9%)は上限 FEATURE_MATCH_MAX_FRACTION(5%) 未満に収める。
    実ジョブのずれは10〜25px/1024(1〜2.5%)。
    """
    synth, mask, reference = _eye_fixture(synth_eye_x=100, ref_eye_x=110)
    corrected, moved = texrefine.align_reference_features(synth, mask, reference)

    assert moved == 1
    out = np.asarray(corrected)
    assert out[128, 100, :3].mean() < 60, "目がメッシュ側の位置に来ていない"
    assert out[128, 121, :3].mean() > 150, "元の位置に目が残っている(二重写し)"
    # 形が保たれている(移動先で円がそのまま): 縁の少し内側も暗い
    assert out[128 - 10, 100, :3].mean() < 60


def test_features_without_a_match_stay_put():
    """メッシュ側に対応が無い(ずれ上限超)特徴は動かさない。"""
    synth, mask, reference = _eye_fixture(synth_eye_x=40, ref_eye_x=210)
    corrected, moved = texrefine.align_reference_features(synth, mask, reference)

    assert moved == 0
    assert np.array_equal(np.asarray(corrected), np.asarray(reference.convert("RGBA")))


def test_shadow_deepening_darkens_baked_shine_only():
    """黒締めは「暗い無彩色」(目の照り)だけに効き、デニムや毛には効かない。"""
    tex = np.array(
        [[[50, 45, 42],     # 目の照り(暗い無彩色) -> 締まる
          [26, 30, 67],     # デニム(暗いが有彩色) -> ほぼ不変
          [229, 196, 160]]],  # 毛(明るい) -> 不変
        dtype=np.float32,
    )
    out = texrefine.deepen_neutral_shadows(tex)

    assert out[0, 0].mean() < tex[0, 0].mean() * 0.65, "目の照りが締まっていない"
    assert np.allclose(out[0, 1], tex[0, 1], atol=6), "デニムまで暗くなっている"
    assert np.allclose(out[0, 2], tex[0, 2]), "明るい毛が変わっている"


def test_refine_applies_shadow_deepening_to_the_atlas():
    """精細化の出力では、参照から転写された暗い無彩色が締められている。"""
    mesh = _sphere_with_uv()
    shine = (50, 45, 42)
    stats = texrefine.refine_texture_with_references(
        mesh, {"front": _circular_reference(color=shine)}
    )

    assert stats.applied, stats.reason
    front, _ = _vertex_colors_by_facing(mesh)
    assert front.mean(axis=0).mean() < np.mean(shine) * 0.65


def test_declines_when_uv_count_mismatches():
    mesh = _sphere_with_uv()
    mesh.visual.uv = np.asarray(mesh.visual.uv)[:-5]
    stats = texrefine.refine_texture_with_references(mesh, {"front": _circular_reference()})
    assert stats.applied is False
    assert "一致しません" in stats.reason
