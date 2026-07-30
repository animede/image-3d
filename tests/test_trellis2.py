"""TRELLIS.2ジェネレータ統合のGPU不要テスト。

実モデルのロード・生成はGPU/16GB重みが必要なためここでは行わず、
以下のGPU不要な純ロジックのみを検証する:
  - ジェネレータのバリデーション(アルファ無し画像拒否、マルチビュー許容)
  - _build_generator("trellis2") の解決
  - 互換スタブ(_ensure_shims)の解決挙動と xformers スタブの数値等価性
    (ブロック対角アテンション vs 系列ごとのSDPAループ)
  - nvdiffrast フェイクパッケージの pixal3d_raster への転送
  - /api/health の texgen_available (provides_texture ジェネレータで true)
  - 事前テクスチャ付きraw mesh → texgenスキップ → texrefine直接適用のE2E
    (テクスチャ付きメッシュを metadata で返すスタブジェネレータ使用)

実際の生成成功経路(25秒生成+4分GLB化)は実モデル検証でカバーする
(data/spikes/trellis2-hybrid-20260730 参照)。
"""
import importlib.util
import io
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from server.generators.base import GenerationParams
from server.generators.mock import MockGenerator
from server.generators.trellis2 import (
    PRETEXTURED_MESH_KEY,
    _SHIMS_DIR,
    Trellis2Generator,
    _ensure_shims,
    _has_meaningful_alpha,
)


def make_rgba_image(size=64, with_alpha=True) -> Image.Image:
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :] = [200, 50, 50, 255]
    if with_alpha:
        arr[: size // 4, :, 3] = 0  # 上1/4を透明(背景除去済みを模す)
    return Image.fromarray(arr, "RGBA")


# --- ジェネレータバリデーション ------------------------------------------------


def test_generator_name_and_texture_capability():
    gen = Trellis2Generator()
    assert gen.name == "trellis2"
    # jobs.py / /api/health が texgen 無しでも paint を提供する判断に使う
    assert gen.provides_texture is True


def test_image_without_alpha_rejected_without_loading_pipeline():
    gen = Trellis2Generator()
    image = Image.new("RGB", (64, 64), (200, 50, 50))
    with pytest.raises(RuntimeError, match="アルファ"):
        gen.generate(image, GenerationParams())
    assert gen._pipeline is None


def test_multiview_is_ignored_not_rejected():
    """extra_views はエラーにしない (texrefine の参照として jobs.py が使うため)。

    アルファ無し画像で呼び、マルチビューのValueErrorではなくアルファの
    RuntimeErrorに到達する(=マルチビューチェックで弾かれていない)ことを確認。
    """
    gen = Trellis2Generator()
    image = Image.new("RGB", (64, 64), (200, 50, 50))
    with pytest.raises(RuntimeError, match="アルファ"):
        gen.generate(
            image,
            GenerationParams(),
            extra_views={"back": make_rgba_image(), "left": make_rgba_image()},
        )
    assert gen._pipeline is None


def test_has_meaningful_alpha():
    assert _has_meaningful_alpha(make_rgba_image(with_alpha=True))
    assert not _has_meaningful_alpha(make_rgba_image(with_alpha=False))
    assert not _has_meaningful_alpha(Image.new("RGB", (8, 8)))


def test_build_generator_resolves_trellis2(monkeypatch):
    from server import config
    from server import main as main_module

    monkeypatch.setattr(config, "GENERATOR", "trellis2")
    gen = main_module._build_generator()
    assert isinstance(gen, Trellis2Generator)


# --- 互換スタブ ------------------------------------------------------------


@pytest.fixture()
def clean_shims():
    """スタブが sys.path / sys.modules に残らないように後始末する。"""
    yield
    while str(_SHIMS_DIR) in sys.path:
        sys.path.remove(str(_SHIMS_DIR))
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in ("xformers", "nvdiffrast"):
            module = sys.modules[name]
            module_file = getattr(module, "__file__", "") or ""
            if str(_SHIMS_DIR) in module_file:
                del sys.modules[name]


def test_ensure_shims_adds_path_only_when_missing(clean_shims, monkeypatch):
    # 本物の xformers / nvdiffrast が両方 import できる状況ではパスを足さない
    monkeypatch.setitem(sys.modules, "xformers", types.ModuleType("xformers"))
    monkeypatch.setitem(sys.modules, "nvdiffrast", types.ModuleType("nvdiffrast"))
    before = list(sys.path)
    _ensure_shims()
    assert sys.path == before


def test_ensure_shims_provides_importable_stubs(clean_shims):
    # このテスト環境 (.venv) には本物が無い前提 (あればスキップ)
    for name in ("xformers", "nvdiffrast"):
        if name in sys.modules and str(_SHIMS_DIR) not in (
            getattr(sys.modules[name], "__file__", "") or ""
        ):
            pytest.skip(f"本物の {name} が導入されている環境")
    pytest.importorskip("torch")
    _ensure_shims()
    assert str(_SHIMS_DIR) in sys.path

    import xformers.ops as xops

    assert str(_SHIMS_DIR) in xops.__file__
    mask = xops.fmha.BlockDiagonalMask.from_seqlens([3, 5])
    assert mask.q_seqlen == [3, 5] and mask.kv_seqlen == [3, 5]

    import nvdiffrast.torch as dr

    # pixal3d_raster (drtkシム) の3 APIへ転送されていること
    from server.generators import pixal3d_raster

    assert dr.rasterize is pixal3d_raster.rasterize
    assert dr.interpolate is pixal3d_raster.interpolate
    assert dr.RasterizeCudaContext is pixal3d_raster.RasterizeCudaContext


def _load_xformers_ops_stub():
    """sys.path を汚さずスタブの ops モジュールを直接ロードする。"""
    path = _SHIMS_DIR / "xformers" / "ops" / "__init__.py"
    spec = importlib.util.spec_from_file_location("_trellis2_test_xops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_xformers_stub_matches_blockwise_sdpa():
    """ブロック対角アテンションが「系列ごとのSDPAループ」と一致すること。"""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    xops = _load_xformers_ops_stub()

    torch.manual_seed(0)
    q_lens, kv_lens = [3, 7, 1], [4, 2, 5]
    H, C = 2, 8
    q = torch.randn(1, sum(q_lens), H, C)
    k = torch.randn(1, sum(kv_lens), H, C)
    v = torch.randn(1, sum(kv_lens), H, C)

    mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_lens, kv_lens)
    out = xops.memory_efficient_attention(q, k, v, mask)
    assert out.shape == (1, sum(q_lens), H, C)

    # 参照実装: ブロックごとに個別のSDPA
    expected = []
    q0 = k0 = 0
    for ql, kl in zip(q_lens, kv_lens):
        qb = q[0, q0 : q0 + ql].transpose(0, 1)[None]  # [1, H, ql, C]
        kb = k[0, k0 : k0 + kl].transpose(0, 1)[None]
        vb = v[0, k0 : k0 + kl].transpose(0, 1)[None]
        ob = F.scaled_dot_product_attention(qb, kb, vb)
        expected.append(ob[0].transpose(0, 1))  # [ql, H, C]
        q0 += ql
        k0 += kl
    expected = torch.cat(expected, dim=0)[None]

    assert torch.allclose(out, expected, atol=1e-5)


def test_xformers_stub_single_sequence_no_mask():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    xops = _load_xformers_ops_stub()
    torch.manual_seed(1)
    q = torch.randn(1, 6, 2, 4)
    k = torch.randn(1, 6, 2, 4)
    v = torch.randn(1, 6, 2, 4)
    out = xops.memory_efficient_attention(q, k, v, None)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    assert torch.allclose(out, expected, atol=1e-5)


# --- /api/health -------------------------------------------------------------


def test_health_texgen_available_with_provides_texture(tmp_path, monkeypatch):
    from server import config, texture
    from server import main as main_module

    data_dir = tmp_path / "data"
    (data_dir / "jobs").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "JOBS_DIR", data_dir / "jobs")
    monkeypatch.setattr(texture, "is_available", lambda: False)

    class _StubGen(MockGenerator):
        name = "trellis2"
        provides_texture = True

    monkeypatch.setattr(main_module.job_manager, "generator", _StubGen())
    with TestClient(main_module.app) as client:
        data = client.get("/api/health").json()
    assert data["generator"] == "trellis2"
    assert data["texgen_available"] is True


# --- 事前テクスチャ付き paint 経路のE2E(スタブ使用、GPU不要) -------------------


def _make_textured_mesh() -> trimesh.Trimesh:
    """UV+ベースカラーテクスチャ付きの単位ボックス (生成スケールを模す)。"""
    mesh = trimesh.creation.box(extents=(0.4, 0.2, 1.0))
    mesh = mesh.unwrap() if hasattr(mesh, "unwrap") else mesh
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        # unwrap が使えない環境向けの単純平面UV
        v = mesh.vertices
        span = v.max(axis=0) - v.min(axis=0)
        uv = (v[:, :2] - v.min(axis=0)[:2]) / np.maximum(span[:2], 1e-9)
    atlas = Image.new("RGB", (32, 32), (10, 200, 30))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=atlas)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh


class PretexturedMockGenerator(MockGenerator):
    """trellis2 の「頂点カラー+metadata[pretextured_mesh]」出力を模すスタブ。"""

    name = "pretextured-mock"
    provides_texture = True

    def generate(self, image, params, extra_views=None):
        mesh = super().generate(image, params, extra_views)
        colors = np.full((len(mesh.vertices), 4), [10, 200, 30, 255], dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        mesh.metadata[PRETEXTURED_MESH_KEY] = _make_textured_mesh()
        return mesh


@pytest.fixture()
def pretextured_client(tmp_path, monkeypatch):
    from server import config
    from server import main as main_module

    data_dir = tmp_path / "data"
    jobs_dir = data_dir / "jobs"
    jobs_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "JOBS_DIR", jobs_dir)

    main_module.job_manager.jobs = {}
    monkeypatch.setattr(main_module.job_manager, "generator", PretexturedMockGenerator())

    # texgen パイプラインには触れてはならない (trellis2 環境には hy3dgen が無い)
    def _fail_get_pipeline(self):
        raise AssertionError("pretextured経路で texgen パイプラインが取得された")

    monkeypatch.setattr(
        main_module.job_manager.__class__,
        "_get_texture_pipeline",
        _fail_get_pipeline,
        raising=True,
    )
    with TestClient(main_module.app) as c:
        yield c


def _wait_for_completion(client, job_id, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200
        job = res.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def make_png_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_pretextured_paint_with_refine(pretextured_client, monkeypatch):
    """paint+refine: texgenを通らず、スケール調整済みメッシュへtexrefineを直接適用。"""
    from server import texrefine

    refined_meshes = []

    def _fake_refine(mesh, references, **kwargs):
        refined_meshes.append((mesh, dict(references)))
        return types.SimpleNamespace(applied=True, reason=None)

    monkeypatch.setattr(
        texrefine, "refine_texture_with_references", _fake_refine, raising=True
    )

    res = pretextured_client.post(
        "/api/jobs",
        files={"image": ("test.png", make_png_bytes(), "image/png")},
        data={
            "params": '{"texture_mode": "paint", "texture_refine": true, '
            '"remove_bg": false, "target_height_mm": 100, "seed": 7}'
        },
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    job = _wait_for_completion(pretextured_client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["textured"] is True
    assert not any("フォールバック" in w for w in job["warnings"]), job["warnings"]

    # texrefine は1回だけ、正面参照付きで呼ばれている
    assert len(refined_meshes) == 1
    mesh, references = refined_meshes[0]
    assert "front" in references

    # 事前テクスチャメッシュは meshproc 済みメッシュの高さに合わせて
    # スケール・接地されている (スタブは高さ1.0の単位ボックス)
    z = mesh.bounds[:, 2]
    assert z[1] - z[0] == pytest.approx(100.0, rel=0.05)
    assert z[0] == pytest.approx(0.0, abs=1.0)

    # 精細化前アトラスが保存されている (texrefine 単体再実行用)
    from server import config

    job_dir = config.JOBS_DIR / job_id
    assert (job_dir / "texture_texgen.png").exists()

    # ビューア用GLBはテクスチャ付きメッシュ由来
    res = pretextured_client.get(f"/api/jobs/{job_id}/model.glb")
    assert res.status_code == 200
    assert len(res.content) > 0


def test_pretextured_paint_without_refine_skips_texrefine(pretextured_client, monkeypatch):
    """texture_refine=false なら texrefine を呼ばず、生成テクスチャをそのまま使う。"""
    from server import texrefine

    called = []
    monkeypatch.setattr(
        texrefine,
        "refine_texture_with_references",
        lambda *a, **k: called.append(1),
        raising=True,
    )

    res = pretextured_client.post(
        "/api/jobs",
        files={"image": ("test.png", make_png_bytes(), "image/png")},
        data={"params": '{"texture_mode": "paint", "texture_refine": false, "remove_bg": false}'},
    )
    job = _wait_for_completion(pretextured_client, res.json()["job_id"])
    assert job["status"] == "completed", job.get("error")
    assert job["textured"] is True
    assert called == []


def test_pretextured_mesh_not_used_without_paint(pretextured_client):
    """texture_mode=none では事前テクスチャは使われず、通常の頂点カラー経路。"""
    res = pretextured_client.post(
        "/api/jobs",
        files={"image": ("test.png", make_png_bytes(), "image/png")},
        data={"params": '{"remove_bg": false}'},
    )
    job = _wait_for_completion(pretextured_client, res.json()["job_id"])
    assert job["status"] == "completed", job.get("error")
    assert job["textured"] is False
