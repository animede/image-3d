"""server/texture.py の単体テスト (SPEC.md §3.9 / FR-10, IMPLEMENTATION_PLAN.md Phase 3c)。

`sample_vertex_colors_from_texture` はGPU不要の純関数のため、合成UVメッシュ
(単純な平面2枚=正方形)+合成テクスチャ(既知の色ブロック画像)で
UV→ピクセル対応が正しいことを検証する。
"""
import numpy as np
import pytest
import trimesh
from PIL import Image

from server import texture


def make_uv_quad() -> trimesh.Trimesh:
    """(0,0)-(1,0)-(1,1)-(0,1) の正方形(2三角形)平面メッシュ。UVは頂点位置と同じ。"""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # uv (0,0) -> 左下
            [1.0, 0.0, 0.0],  # uv (1,0) -> 右下
            [1.0, 1.0, 0.0],  # uv (1,1) -> 右上
            [0.0, 1.0, 0.0],  # uv (0,1) -> 左上
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # 4色ブロックテクスチャ: 画像座標は左上原点。
    # uv v=0(下)は画像下端、v=1(上)は画像上端に対応する慣例(OpenGL式)。
    size = 64
    half = size // 2
    tex = np.zeros((size, size, 4), dtype=np.uint8)
    tex[:half, :half] = [255, 0, 0, 255]  # 画像上半分・左 = 赤 (uv v近く1、u近く0 -> 左上頂点)
    tex[:half, half:] = [0, 255, 0, 255]  # 画像上半分・右 = 緑 (右上頂点)
    tex[half:, :half] = [0, 0, 255, 255]  # 画像下半分・左 = 青 (左下頂点)
    tex[half:, half:] = [255, 255, 0, 255]  # 画像下半分・右 = 黄 (右下頂点)
    tex_image = Image.fromarray(tex, "RGBA")

    material = trimesh.visual.texture.SimpleMaterial(image=tex_image)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex_image, material=material)
    return mesh


def test_sample_vertex_colors_matches_uv_corners():
    mesh = make_uv_quad()
    colors = texture.sample_vertex_colors_from_texture(mesh)

    assert colors.shape == (4, 4)
    assert colors.dtype == np.uint8

    # 頂点0: uv(0,0) -> 左下 -> 画像下半分・左 = 青
    assert tuple(colors[0][:3]) == (0, 0, 255)
    # 頂点1: uv(1,0) -> 右下 -> 画像下半分・右 = 黄
    assert tuple(colors[1][:3]) == (255, 255, 0)
    # 頂点2: uv(1,1) -> 右上 -> 画像上半分・右 = 緑
    assert tuple(colors[2][:3]) == (0, 255, 0)
    # 頂点3: uv(0,1) -> 左上 -> 画像上半分・左 = 赤
    assert tuple(colors[3][:3]) == (255, 0, 0)

    # アルファは常に不透明
    assert (colors[:, 3] == 255).all()


def test_sample_vertex_colors_without_texture_returns_white():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    colors = texture.sample_vertex_colors_from_texture(mesh)
    assert colors.shape == (len(mesh.vertices), 4)
    assert (colors == 255).all()


def test_is_available_returns_bool():
    # 環境に依存するが、常にbool型であることは保証される(実ロードはしない)。
    result = texture.is_available()
    assert isinstance(result, bool)


# --- 材質の金属度 (glTF metallicFactor) ---------------------------------------


def _textured_box():
    """テクスチャ付きメッシュ(paint出力相当)を作る。"""
    import numpy as np
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial

    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=PBRMaterial(baseColorTexture=Image.new("RGB", (4, 4), (200, 150, 100))),
    )
    return mesh


def test_glb_without_metallic_factor_defaults_to_metal():
    """前提の確認: 未指定だと trimesh は metallicFactor を書き出さない。

    glTF 2.0 の既定値は 1.0(完全な金属)なので、書き出さない = 金属になる。
    """
    import json
    import struct

    data = _textured_box().export(file_type="glb")
    length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + length])
    assert "metallicFactor" not in gltf["materials"][0]["pbrMetallicRoughness"]


def test_ensure_non_metallic_sets_zero():
    from server import texture

    mesh = _textured_box()
    assert texture.ensure_non_metallic(mesh) is True
    assert mesh.visual.material.metallicFactor == 0.0


def test_ensure_non_metallic_is_written_to_glb():
    """暗く沈む原因だった metallicFactor が GLB に 0 として入ること。"""
    import json
    import struct

    from server import texture

    mesh = _textured_box()
    texture.ensure_non_metallic(mesh)
    data = mesh.export(file_type="glb")
    length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + length])
    assert gltf["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 0.0


def test_ensure_non_metallic_keeps_explicit_value():
    """明示的に指定された金属度は尊重する(勝手に上書きしない)。"""
    from server import texture

    mesh = _textured_box()
    mesh.visual.material.metallicFactor = 1.0
    assert texture.ensure_non_metallic(mesh) is False
    assert mesh.visual.material.metallicFactor == 1.0


def test_ensure_non_metallic_ignores_meshes_without_material():
    import trimesh

    from server import texture

    assert texture.ensure_non_metallic(trimesh.creation.box()) is False


def test_ensure_non_metallic_converts_simple_material():
    """texgen が返す SimpleMaterial は metallicFactor を持たない。

    そのまま GLB にすると baseColorFactor と roughnessFactor は書かれるのに
    metallicFactor だけ落ち、glTF既定の 1.0(金属)が適用されて暗くなる。
    """
    import json
    import struct

    import numpy as np
    import trimesh
    from PIL import Image
    from trimesh.visual.material import SimpleMaterial

    from server import texture

    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=SimpleMaterial(image=Image.new("RGB", (4, 4), (200, 150, 100))),
    )
    assert not hasattr(mesh.visual.material, "metallicFactor")

    assert texture.ensure_non_metallic(mesh) is True

    data = mesh.export(file_type="glb")
    length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + length])
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 0.0
    assert "baseColorTexture" in pbr, "テクスチャが失われた"
    assert gltf["images"], "テクスチャ画像が失われた"


# --- 背面参照画像対応 (paint()の複数画像入力) ---------------------------------


def _fake_wrapper_with_recording_pipeline(monkeypatch):
    """paint()の内部ロジックのみを検証するため、_load_pipelineをモックに差し替える。

    実パイプラインはGPU/hy3dgenロードを要求するため、テストでは
    「pipeline(mesh, image_or_list) がどう呼ばれたか」だけを記録する
    軽量な偽パイプラインで置き換える(単体テストの方針: GPU不要)。
    """
    wrapper = texture.TexturePipelineWrapper()
    calls: list = []

    def fake_pipeline(mesh, image_arg):
        calls.append(image_arg)
        textured = trimesh.creation.box()
        textured.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(textured.vertices), 2)),
            material=trimesh.visual.material.SimpleMaterial(
                image=Image.new("RGB", (4, 4), (10, 20, 30))
            ),
        )
        return textured

    monkeypatch.setattr(wrapper, "_load_pipeline", lambda: fake_pipeline)
    return wrapper, calls


def test_paint_passes_single_image_when_no_back_image(monkeypatch):
    """back_imageを渡さない場合は従来通り画像1枚(リストでなく単体)を渡す。"""
    wrapper, calls = _fake_wrapper_with_recording_pipeline(monkeypatch)
    mesh = trimesh.creation.box()
    front = Image.new("RGBA", (8, 8), (255, 0, 0, 255))

    wrapper.paint(mesh, front)

    assert len(calls) == 1
    assert calls[0] is front  # リスト化されていないこと(後方互換)


def test_paint_passes_image_list_when_back_image_given(monkeypatch):
    """back_imageを渡すと [front, back] のリストでパイプラインへ渡す。"""
    wrapper, calls = _fake_wrapper_with_recording_pipeline(monkeypatch)
    mesh = trimesh.creation.box()
    front = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    back = Image.new("RGBA", (8, 8), (0, 0, 255, 255))

    wrapper.paint(mesh, front, back_image=back)

    assert len(calls) == 1
    assert calls[0] == [front, back]


def test_paint_truncates_beyond_max_ref_images(monkeypatch, caplog):
    """MAX_REF_IMAGES(=5、hunyuanpaint/unet/modules.py:456のmax_num_ref_image由来)
    を超える枚数を渡そうとしても例外にせず、上限で切り詰めて警告を出す。

    現状の呼び出し元(front + back の最大2枚)では発生しないが、将来の
    拡張に対する安全弁として、内部ヘルパーで直接大量の画像を渡した場合の
    挙動を検証する。
    """
    wrapper, calls = _fake_wrapper_with_recording_pipeline(monkeypatch)
    mesh = trimesh.creation.box()
    front = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    back = Image.new("RGBA", (8, 8), (0, 0, 255, 255))

    # paint()のシグネチャはfront+back限定なので、切り詰めロジック自体を
    # MAX_REF_IMAGESを一時的に下げて2枚のケースで検証する。
    monkeypatch.setattr(texture, "MAX_REF_IMAGES", 1)
    with caplog.at_level("WARNING"):
        wrapper.paint(mesh, front, back_image=back)

    assert len(calls) == 1
    assert calls[0] is front  # 2枚目(back)が切り詰められ、1枚扱いになる
    assert any("Truncating" in rec.message for rec in caplog.records)


# --- multiview_utils.camera_info_ref モンキーパッチのフォールバック ------------


class _FakeMultiviewModelWrongType:
    """Multiview_Diffusion_Netではない偽の属性(想定外の構造)。"""


class _FakePipelineWithWrongModelType:
    def __init__(self):
        self.models = {"multiview_model": _FakeMultiviewModelWrongType()}


class _FakePipelineMissingModels:
    """`models`属性自体を持たない偽パイプライン(想定と違う構造)。"""


def test_patch_falls_back_gracefully_when_model_type_unexpected(caplog):
    """ラップ対象が想定と異なる型でも例外を出さず警告になる(graceful degradation)。"""
    fake_pipeline = _FakePipelineWithWrongModelType()
    with caplog.at_level("WARNING"):
        texture._patch_multiview_ref_camera_info(fake_pipeline)
    assert any(
        "multi-reference" in rec.message or "camera_info_ref" in rec.message
        for rec in caplog.records
    )


def test_patch_falls_back_gracefully_when_models_attribute_missing(caplog):
    """`pipeline.models`自体が無い(属性エラー)場合も例外にせず警告になる。"""
    fake_pipeline = _FakePipelineMissingModels()
    with caplog.at_level("WARNING"):
        texture._patch_multiview_ref_camera_info(fake_pipeline)
    assert any(
        "multi-reference" in rec.message or "camera_info_ref" in rec.message
        for rec in caplog.records
    )
