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


def _two_plates_mesh(texture_size: int = 256, texture: "Image.Image | None" = None) -> trimesh.Trimesh:
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
    if texture is None:
        texture = Image.new("RGB", (texture_size, texture_size), (TEXGEN_GREY,) * 3)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.SimpleMaterial(image=texture)
    )
    return mesh


GREEN = (30, 160, 60)


def _two_tone_reference(size: int = 256) -> Image.Image:
    """全体は緑、手前の板の足元(中央40%)だけ赤の参照画像。

    「隠れた面が手前の板の色(赤)を透過して拾っていないか」を色で判別できる。
    """
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., :3] = GREEN
    arr[..., 3] = 255
    lo, hi = int(size * 0.30), int(size * 0.70)
    arr[lo:hi, lo:hi, :3] = REFERENCE_RED
    return Image.fromarray(arr, "RGBA")


def test_occluded_surfaces_are_not_painted_from_the_reference():
    """手前の板に隠れた面は、正面を向いていても参照から塗らない。

    これを怠ると、帽子の垂れや耳の裏に隠れた頭側面が、そこを覆っている
    帽子の赤・耳の黒を拾って筋になる(実ジョブ46b64850で確認した欠陥)。
    """
    mesh = _two_plates_mesh()
    stats = texrefine.refine_texture_with_references(mesh, {"front": _two_tone_reference()})

    assert stats.applied, stats.reason
    assert stats.occluded_sample_ratio > 0.0
    tex = np.asarray(texrefine._extract_texture_image(mesh.visual).convert("RGB")).astype(int)
    h, w = tex.shape[:2]
    front_plate = tex[h // 2, int(w * 0.75)]  # 手前の板の中央
    hidden_centre = tex[h // 2, int(w * 0.25)]  # 後ろの板のうち隠れている中央部
    assert np.allclose(front_plate, REFERENCE_RED, atol=10), "手前の板が塗られていない"
    # 隠れた面は「手前の板の赤」を透過して拾ってはいけない。texgen が平坦なら
    # 同じ板の見えている部分(緑)から調和拡散で埋まる。
    assert hidden_centre[0] < 120, "隠れた面が遮蔽物(赤)を透過して拾っている"
    assert hidden_centre[1] > 80, "隠れた面が同じ板の周囲の色(緑)で埋まっていない"


def test_visible_parts_of_a_partially_occluded_surface_are_still_painted():
    """部分的に隠れた面でも、見えている部分は参照から塗る。"""
    mesh = _two_plates_mesh()
    texrefine.refine_texture_with_references(mesh, {"front": _two_tone_reference()})

    tex = np.asarray(texrefine._extract_texture_image(mesh.visual).convert("RGB")).astype(int)
    h, w = tex.shape[:2]
    # 後ろの板の外周(手前の板 40/100 の外側)は見えている。
    # UV左半分の端 (u=0.08 -> 板の左端付近) をサンプルする。
    visible_edge = tex[h // 2, int(w * 0.08)]
    assert np.allclose(visible_edge, GREEN, atol=12), "見えている外周が塗られていない"


def test_fill_leaves_textured_hidden_areas_to_texgen():
    """texgen が描き込みを持つ隠れ領域は拡散で潰さない(平坦ゲート)。"""
    size = 256
    base = np.full((size, size, 3), TEXGEN_GREY, np.uint8)
    # 後ろの板のチャート(左半分)に市松模様 = texgen に本物の情報がある想定
    yy, xx = np.mgrid[0:size, 0:size]
    checker = (((yy // 8) + (xx // 8)) % 2 == 0) & (xx < size // 2)
    base[checker] = (40, 40, 40)
    mesh = _two_plates_mesh(texture=Image.fromarray(base, "RGB"))
    stats = texrefine.refine_texture_with_references(mesh, {"front": _two_tone_reference()})

    tex = np.asarray(texrefine._extract_texture_image(mesh.visual).convert("RGB")).astype(int)
    h, w = tex.shape[:2]
    patch = tex[h // 2 - 8 : h // 2 + 8, int(w * 0.25) - 8 : int(w * 0.25) + 8]
    assert patch.std() > 20, "隠れ領域の texgen の描き込みが拡散で潰された"


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

    ずれ(8px/256=3.1%)は上限 FEATURE_MATCH_MAX_FRACTION(3.5%) 未満に収める。
    実ジョブの正当なずれは10〜25px/1024(1〜2.5%)。
    """
    synth, mask, reference = _eye_fixture(synth_eye_x=102, ref_eye_x=110)
    corrected, moved = texrefine.align_reference_features(synth, mask, reference)

    assert moved == 1
    out = np.asarray(corrected)
    assert out[128, 102, :3].mean() < 60, "目がメッシュ側の位置に来ていない"
    assert out[128, 121, :3].mean() > 150, "元の位置に目が残っている(二重写し)"
    # 形が保たれている(移動先で円がそのまま): 縁の少し内側も暗い
    assert out[128 - 10, 102, :3].mean() < 60


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


def test_features_do_not_match_dissimilar_shapes():
    """コンパクトな目は細長い髪の影と対応しない(誤マッチで目が消える再発防止)。"""
    size = 256
    yy, xx = np.mgrid[0:size, 0:size]

    synth = np.full((size, size, 3), 255, np.uint8)
    # メッシュ側にあるのは細長い影だけ(texgenが顔を描かなかった状況)
    synth[(np.abs(yy - 120) < 2) & (np.abs(xx - 120) < 30)] = 20
    mask = (xx - 128) ** 2 + (yy - 128) ** 2 <= 110 ** 2

    ref = np.full((size, size, 4), 255, np.uint8)
    eye = (xx - 128) ** 2 + (yy - 128) ** 2 <= 12 ** 2  # コンパクトな目
    ref[eye, :3] = 20
    reference = Image.fromarray(ref, "RGBA")

    corrected, moved = texrefine.align_reference_features(synth, mask, reference)
    assert moved == 0, "目が髪の影と誤対応して移動している"
    assert np.array_equal(np.asarray(corrected), np.asarray(reference.convert("RGBA")))


# --- シルエット不一致ガード ---------------------------------------------------
#
# 側面参照はTポーズのままだと腕が胴を隠すため、腕を前へ出した姿勢で作らざるを
# 得ない。腕だけメッシュと食い違うので、その領域から転写しないようにする。


def _silhouette_pair(size=256, ref_extra=None, synth_extra=None):
    """胴(共通)+腕(位置違い)のシルエット対を作る。"""
    yy, xx = np.mgrid[0:size, 0:size]
    torso = (np.abs(xx - 128) < 30) & (np.abs(yy - 128) < 70)

    synth = torso.copy()
    if synth_extra is not None:
        synth |= synth_extra(xx, yy)

    ref = np.zeros((size, size, 4), np.uint8)
    mask = torso.copy()
    if ref_extra is not None:
        mask |= ref_extra(xx, yy)
    ref[mask, :3] = REFERENCE_RED
    ref[mask, 3] = 255
    return synth, Image.fromarray(ref, "RGBA")


def test_silhouette_guard_keeps_the_agreeing_parts():
    """胴が一致していれば胴は信頼される。"""
    synth, ref = _silhouette_pair()
    agree, ratio = texrefine.silhouette_agreement_mask(synth, ref)
    assert ratio < 0.05, "同一シルエットなのに不一致と判定されている"
    assert agree[128, 128], "一致している胴の中心が捨てられている"


def test_silhouette_guard_rejects_a_differently_posed_arm():
    """メッシュは腕が横、参照は腕が前 → その領域は信頼しない。"""
    synth, ref = _silhouette_pair(
        synth_extra=lambda x, y: (np.abs(y - 80) < 8) & (x > 150) & (x < 230),  # 横へ
        ref_extra=lambda x, y: (np.abs(y - 150) < 8) & (x > 150) & (x < 230),   # 別位置
    )
    agree, ratio = texrefine.silhouette_agreement_mask(synth, ref)
    assert ratio > 0.05, "ポーズ違いが不一致として検出されていない"
    assert not agree[80, 190], "メッシュ側の腕の位置が信頼されたままになっている"
    assert not agree[150, 190], "参照側の腕の位置が信頼されたままになっている"
    assert agree[128, 128], "一致している胴まで捨てている"


def test_silhouette_guard_tolerates_the_point_sampled_synth_holes():
    """合成ビューの点描き由来の細かい穴は不一致に数えない。"""
    synth, ref = _silhouette_pair()
    rng = np.random.default_rng(0)
    holes = rng.random(synth.shape) < 0.3
    _, ratio = texrefine.silhouette_agreement_mask(synth & ~holes, ref)
    assert ratio < 0.05, "サンプル穴を不一致と誤判定している"


def test_side_views_do_not_take_face_on_surfaces():
    """側面参照を足しても、正面を正対して向く面は正面参照のまま。

    側面参照はシルエットが一致していても内部の特徴が数pxずれることがあり、
    顔のような細かい領域では黒い塊として見える(実ジョブ24d313abで確認)。
    正面・背面が dot>=0.5 で見えている面は側面に渡さない。
    """
    mesh = _sphere_with_uv()
    front = _circular_reference(color=REFERENCE_RED)
    side = _circular_reference(color=(20, 220, 40))  # 側面は緑
    stats = texrefine.refine_texture_with_references(
        mesh, {"front": front, "left": side, "right": side}
    )
    assert stats.applied, stats.reason

    colors = sample_vertex_colors_from_texture(mesh)[:, :3].astype(int)
    normals = np.asarray(mesh.vertex_normals)
    facing = normals @ np.array([0.0, -1.0, 0.0])
    front_on = colors[facing > 0.9]
    assert np.allclose(front_on.mean(axis=0), REFERENCE_RED, atol=15), "正対面が側面色で塗られた"

    # 真横を向く面は側面参照が担当する
    sideways = colors[np.abs(normals @ np.array([1.0, 0.0, 0.0])) > 0.95]
    assert sideways[:, 1].mean() > sideways[:, 0].mean(), "真横が側面参照で塗られていない"


def test_head_base_is_found_below_the_narrowing_above_the_shoulders():
    """Tポーズの肩から上で幅が落ちる位置を頭の付け根とする。"""
    # 胴+横に張り出した腕、その上に細い頭、というTポーズ状の形
    torso = trimesh.creation.box(extents=(20, 10, 60))
    torso.apply_translation([0, 0, 30])
    arms = trimesh.creation.box(extents=(100, 8, 8))
    arms.apply_translation([0, 0, 55])
    head = trimesh.creation.box(extents=(16, 16, 20))
    head.apply_translation([0, 0, 70])
    mesh = trimesh.util.concatenate([torso, arms, head])

    base = texrefine.head_base_height(mesh)
    assert 55 < base < 65, f"頭の付け根が肩と頭頂の間にない: {base}"


def _narrow_topped_mesh_with_uv(texture_size: int = 512) -> trimesh.Trimesh:
    """上半分が細い「頭付き」の球。頭の付け根が中腹に検出される。

    素の球は真横を向く面が赤道付近にしか無く、頭の上下で挙動を比べられない。
    """
    mesh = _sphere_with_uv(texture_size)
    vertices = np.asarray(mesh.vertices).copy()
    upper = vertices[:, 2] > 0
    vertices[upper, :2] *= 0.35
    mesh.vertices = vertices
    return mesh


def test_side_views_are_not_used_above_the_head_base():
    """頭の付け根より上では側面参照を使わない(顔が側面のずれで壊れるため)。"""
    mesh = _narrow_topped_mesh_with_uv()
    base = texrefine.head_base_height(mesh)
    vertices = np.asarray(mesh.vertices)
    assert vertices[:, 2].min() < base < vertices[:, 2].max(), f"付け根が範囲外: {base}"

    front = _circular_reference(color=REFERENCE_RED)
    side = _circular_reference(color=(20, 220, 40))
    stats = texrefine.refine_texture_with_references(
        mesh, {"front": front, "left": side, "right": side}
    )
    assert stats.applied, stats.reason

    colors = sample_vertex_colors_from_texture(mesh)[:, :3].astype(int)
    normals = np.asarray(mesh.vertex_normals)
    sideways = np.abs(normals @ np.array([1.0, 0.0, 0.0])) > 0.9
    above = sideways & (vertices[:, 2] > base + 5)
    below = sideways & (vertices[:, 2] < base - 5)
    assert above.sum() > 20 and below.sum() > 20, "頭の上下に真横向きの頂点が必要"
    # 付け根より下の真横は側面(緑)が担当する
    assert colors[below][:, 1].mean() > colors[below][:, 0].mean() + 10, "側面が使われていない"
    # 頭の真横は側面を使わない。転写されず texgen(無彩色)のまま残ってよい。
    above_rgb = colors[above].mean(axis=0)
    assert above_rgb[1] <= above_rgb[0] + 10, f"頭に側面(緑)が使われている: {above_rgb}"
