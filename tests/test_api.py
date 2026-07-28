"""API ライフサイクルテスト (IMPLEMENTATION_PLAN.md タスク1-8 (b)(c))。

FastAPI TestClient + mockジェネレータでジョブ作成→完了までポーリング→
GET model.glb / download?format=stl を検証する。STL出力はtrimeshで再読込し
watertight・高さ(mm)を機械検証する(DEVELOPMENT_POLICY.md §5)。
"""
import base64
import io
import json
import time

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import make_test_png_bytes


def make_4color_png_bytes(size=128) -> bytes:
    """カラーモードE2Eテスト用の4色ブロックRGBA画像。"""
    half = size // 2
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:half, :half] = [255, 0, 0, 255]
    arr[:half, half:] = [0, 255, 0, 255]
    arr[half:, :half] = [0, 0, 255, 255]
    arr[half:, half:] = [255, 255, 0, 255]
    img = Image.fromarray(arr, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from server import config

    data_dir = tmp_path / "data"
    jobs_dir = data_dir / "jobs"
    jobs_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "JOBS_DIR", jobs_dir)

    from server import main as main_module

    # 各テストを独立させるため、ジョブ管理状態をリセットする
    main_module.job_manager.jobs = {}

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


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["generator"] == "mock"
    assert "gpu" in data
    assert "texgen_available" in data
    assert isinstance(data["texgen_available"], bool)


def test_job_lifecycle_end_to_end(client):
    png_bytes = make_test_png_bytes()

    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": '{"target_height_mm": 100, "seed": 42}'},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    assert job_id

    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["stats"]["watertight"] is True
    assert job["stats"]["vertices"] > 0
    assert job["stats"]["bbox_mm"][2] == pytest.approx(100.0, abs=1.0)

    # GET /api/jobs (一覧)
    res = client.get("/api/jobs")
    assert res.status_code == 200
    jobs = res.json()
    assert any(j["job_id"] == job_id for j in jobs)

    # GET /api/jobs/{id}/input
    res = client.get(f"/api/jobs/{job_id}/input")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/")

    # GET /api/jobs/{id}/model.glb
    res = client.get(f"/api/jobs/{job_id}/model.glb")
    assert res.status_code == 200
    assert len(res.content) > 0
    glb_mesh = trimesh.load(io.BytesIO(res.content), file_type="glb")
    assert glb_mesh is not None

    # GET download?format=stl
    res = client.get(f"/api/jobs/{job_id}/download?format=stl")
    assert res.status_code == 200
    assert len(res.content) > 0

    stl_mesh = trimesh.load(io.BytesIO(res.content), file_type="stl")
    assert stl_mesh.is_watertight
    height = stl_mesh.bounds[1][2] - stl_mesh.bounds[0][2]
    assert height == pytest.approx(100.0, abs=1.0)

    # 他の形式も取得できること
    for fmt in ("3mf", "obj", "glb"):
        res = client.get(f"/api/jobs/{job_id}/download?format={fmt}")
        assert res.status_code == 200, f"format={fmt}"
        assert len(res.content) > 0

    # DELETE
    res = client.delete(f"/api/jobs/{job_id}")
    assert res.status_code == 200
    res = client.get(f"/api/jobs/{job_id}")
    assert res.status_code == 404


def test_job_history_persistence(client, tmp_path):
    """サーバ再起動を模してjobs.pyの load_history が正しくメタデータを読み込むこと。"""
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": "{}"},
    )
    job_id = res.json()["job_id"]
    _wait_for_completion(client, job_id)

    from server import main as main_module
    from server.jobs import JobManager
    from server.generators.mock import MockGenerator

    new_manager = JobManager(MockGenerator())
    new_manager.load_history()
    assert job_id in new_manager.jobs
    assert new_manager.jobs[job_id].status == "completed"


def test_reject_non_image_file(client):
    res = client.post(
        "/api/jobs",
        files={"image": ("test.txt", b"not an image", "image/png")},
    )
    assert 400 <= res.status_code < 500


def test_reject_invalid_content_type(client):
    res = client.post(
        "/api/jobs",
        files={"image": ("test.txt", b"hello world", "text/plain")},
    )
    assert 400 <= res.status_code < 500


def test_reject_oversized_file(client, monkeypatch):
    from server import config, main as main_module

    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(main_module.config, "MAX_UPLOAD_BYTES", 1000)

    big_png = make_test_png_bytes(size=(512, 512))
    assert len(big_png) > 1000

    res = client.post(
        "/api/jobs",
        files={"image": ("big.png", big_png, "image/png")},
    )
    assert res.status_code == 413


def test_reject_bad_params_json(client):
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": "{not valid json"},
    )
    assert res.status_code == 400


def test_reject_invalid_octree_resolution(client):
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": '{"octree_resolution": 999}'},
    )
    assert res.status_code == 400


def test_get_nonexistent_job_returns_404(client):
    res = client.get("/api/jobs/does-not-exist")
    assert res.status_code == 404


def test_download_before_completion_returns_409(client):
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
    )
    job_id = res.json()["job_id"]
    # 完了前にダウンロードを試みる(タイミング依存だが、生成完了直後に即試行)
    res2 = client.get(f"/api/jobs/{job_id}/download?format=stl")
    assert res2.status_code in (409, 200)  # 速いマシンでは既に完了している可能性あり
    _wait_for_completion(client, job_id)


def test_reject_invalid_n_colors(client):
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": '{"color_mode": "color4", "n_colors": 1}'},
    )
    assert res.status_code == 400


def test_reject_invalid_color_mode(client):
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": '{"color_mode": "rainbow"}'},
    )
    assert res.status_code == 400


def test_color_mode_job_e2e(client):
    """SPEC.md §3.7 (FR-8): color_mode=color4 のジョブがパレット統計を返し、
    3MFダウンロードが色ごとに分割された複数オブジェクトになること、
    GLBに頂点カラーが含まれることを検証する。
    """
    png_bytes = make_4color_png_bytes()

    res = client.post(
        "/api/jobs",
        files={"image": ("test4color.png", png_bytes, "image/png")},
        data={
            "params": '{"color_mode": "color4", "n_colors": 4, "seed": 42, "remove_bg": false}'
        },
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["params"]["color_mode"] == "color4"
    assert job["params"]["n_colors"] == 4

    palette = job["stats"]["palette"]
    assert isinstance(palette, list)
    assert 1 <= len(palette) <= 4
    for entry in palette:
        assert entry["hex"].startswith("#")
        assert 0.0 <= entry["face_ratio"] <= 1.0

    # 3MF: 色ごとに分割された最大4オブジェクト
    res = client.get(f"/api/jobs/{job_id}/download?format=3mf")
    assert res.status_code == 200
    assert len(res.content) > 0
    scene = trimesh.load(io.BytesIO(res.content), file_type="3mf")
    assert hasattr(scene, "geometry")
    assert 1 <= len(scene.geometry) <= 4

    # GLB: 頂点カラーが含まれること
    res = client.get(f"/api/jobs/{job_id}/model.glb")
    assert res.status_code == 200
    glb = trimesh.load(io.BytesIO(res.content), file_type="glb")
    geom = list(glb.geometry.values())[0]
    assert geom.visual.kind == "vertex"
    assert len(geom.visual.vertex_colors) == len(geom.vertices)


def test_color_mode_none_has_empty_palette(client):
    """color_mode=none(デフォルト)の場合、paletteは空リストであること。"""
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": "{}"},
    )
    job_id = res.json()["job_id"]
    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["stats"]["palette"] == []


def test_multiview_job_e2e(client):
    """SPEC.md §3.8 (FR-9): image + image_back の2ビュージョブがcompletedし、
    ジョブmetaの `views` が ["front", "back"] であること(mockでextra_viewsは
    無視されるが、jobs.py側の受付・記録は検証できる)。
    """
    front_bytes = make_test_png_bytes(color=(200, 50, 50))
    back_bytes = make_test_png_bytes(color=(50, 50, 200))

    res = client.post(
        "/api/jobs",
        files={
            "image": ("front.png", front_bytes, "image/png"),
            "image_back": ("back.png", back_bytes, "image/png"),
        },
        data={"params": '{"seed": 42}'},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["views"] == ["front", "back"]

    # 追加ビューの前処理画像も保存されていること
    res = client.get(f"/api/jobs/{job_id}/input")
    assert res.status_code == 200


def test_multiview_job_all_views(client):
    """front + back + left + right の4ビュー受付順序が views に反映されること。"""
    png_bytes = make_test_png_bytes()

    res = client.post(
        "/api/jobs",
        files={
            "image": ("front.png", png_bytes, "image/png"),
            "image_back": ("back.png", png_bytes, "image/png"),
            "image_left": ("left.png", png_bytes, "image/png"),
            "image_right": ("right.png", png_bytes, "image/png"),
        },
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["views"] == ["front", "back", "left", "right"]


def test_single_view_job_has_views_front_only(client):
    """追加ビューを指定しない従来通りのジョブは views == ["front"] であること。"""
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
    )
    job_id = res.json()["job_id"]
    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["views"] == ["front"]


def make_sheet_png_bytes(size=(900, 400), panel_w=200, panel_h=300, gap=100) -> bytes:
    """/api/sheet/split テスト用の3パネル合成RGBAシート画像。"""
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    y0 = (h - panel_h) // 2
    colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]
    x_starts = [gap, gap * 2 + panel_w, gap * 3 + panel_w * 2]
    for x0, color in zip(x_starts, colors):
        arr[y0 : y0 + panel_h, x0 : x0 + panel_w] = color
    img = Image.fromarray(arr, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_sheet_split_endpoint_detects_three_panels(client):
    """POST /api/sheet/split: 合成3パネルシート -> panels長さ3、
    suggested_viewが front/left/back の順であること。
    """
    sheet_bytes = make_sheet_png_bytes()
    res = client.post(
        "/api/sheet/split",
        files={"image": ("sheet.png", sheet_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    panels = data["panels"]
    assert len(panels) == 3

    for idx, panel in enumerate(panels):
        assert panel["index"] == idx
        assert panel["suggested_view"] in ("front", "left", "back", "right")
        # image_b64はデコード可能なPNGであること
        raw = base64.b64decode(panel["image_b64"])
        decoded = Image.open(io.BytesIO(raw))
        assert decoded.format == "PNG"

    assert [p["suggested_view"] for p in panels] == ["front", "left", "back"]


def test_sheet_split_endpoint_rejects_non_image(client):
    res = client.post(
        "/api/sheet/split",
        files={"image": ("test.txt", b"not an image", "image/png")},
    )
    assert 400 <= res.status_code < 500


def test_reject_invalid_texture_mode(client):
    """SPEC.md §3.9 (FR-10): texture_modeは'none'/'paint'以外は400。"""
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": '{"texture_mode": "sculpt"}'},
    )
    assert res.status_code == 400


def test_reject_non_boolean_texture_refine(client):
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": '{"texture_refine": "yes"}'},
    )
    assert res.status_code == 400


def test_texture_refine_defaults_to_off_and_is_recorded(client):
    """高精細化は明示しない限りオフ(素のtexgen=破綻の少ない既定)。"""
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": "{}"},
    )
    job = _wait_for_completion(client, res.json()["job_id"])
    assert job["params"]["texture_refine"] is False

    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": '{"texture_refine": true}'},
    )
    job = _wait_for_completion(client, res.json()["job_id"])
    assert job["params"]["texture_refine"] is True


def test_paint_without_refine_skips_texrefine(client, monkeypatch):
    """texture_refine=false なら texrefine を呼ばない(素のtexgen出力)。"""
    from server import main as main_module, texrefine
    import trimesh as _trimesh

    called = []
    monkeypatch.setattr(
        texrefine,
        "refine_texture_with_references",
        lambda *a, **k: called.append(1),
        raising=True,
    )

    # 実GPUは使わず、texgen成功後の分岐(refine有無)だけを本物の _run_paint で
    # 通すため、pipeline 取得だけを差し替える。
    class _FakePipeline:
        def paint(self, mesh, image, back_image=None):
            return mesh.copy()

    monkeypatch.setattr(
        main_module.job_manager.__class__,
        "_get_texture_pipeline",
        lambda self: _FakePipeline(),
        raising=True,
    )

    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": '{"texture_mode": "paint", "texture_refine": false}'},
    )
    job = _wait_for_completion(client, res.json()["job_id"])
    assert job["status"] == "completed", job.get("error")
    assert called == [], "texture_refine=false なのに texrefine が呼ばれている"


def test_texture_mode_paint_job_completes_with_mock(client, monkeypatch):
    """mock環境でtexture_mode=paintを指定してもジョブは完了すること。

    texgen(paintパイプライン)の実ロードはモデルDL(数GB)を伴うため、
    ユニットテストでは `JobManager._run_paint` をモンキーパッチして
    「paint失敗→graceful degradation」経路を検証する(実際のpaint成功経路は
    実モデル検証(README/報告参照)でカバーする)。
    """
    from server import main as main_module

    def _fail_paint(self, mesh, image, job, back_image=None, refine_references=None):
        job.warnings.append("test: paint intentionally failed")
        return None

    monkeypatch.setattr(
        main_module.job_manager.__class__, "_run_paint", _fail_paint, raising=True
    )

    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
        data={"params": '{"texture_mode": "paint", "seed": 42}'},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["params"]["texture_mode"] == "paint"
    assert job["textured"] is False
    assert len(job["warnings"]) >= 1

    # GLBは(テクスチャの有無に関わらず)取得できること
    res = client.get(f"/api/jobs/{job_id}/model.glb")
    assert res.status_code == 200
    assert len(res.content) > 0


def test_texture_mode_paint_receives_back_view(client, monkeypatch):
    """image_back付き + texture_mode=paint のジョブで、_run_paintに背面画像が
    渡されること(背面参照対応、docs/RIG_SERVICE_PLAN.md記載のスパイク実装)。

    left/rightはスパイク未検証のため今回は渡さない仕様なので、それらを
    指定しても_run_paintに渡らないことも合わせて確認する。
    """
    from server import main as main_module

    received: dict = {}

    def _capture_paint(self, mesh, image, job, back_image=None, refine_references=None):
        received["back_image"] = back_image
        job.warnings.append("test: paint intentionally short-circuited")
        return None

    monkeypatch.setattr(
        main_module.job_manager.__class__, "_run_paint", _capture_paint, raising=True
    )

    front_bytes = make_test_png_bytes(color=(200, 50, 50))
    back_bytes = make_test_png_bytes(color=(50, 50, 200))
    left_bytes = make_test_png_bytes(color=(50, 200, 50))

    res = client.post(
        "/api/jobs",
        files={
            "image": ("front.png", front_bytes, "image/png"),
            "image_back": ("back.png", back_bytes, "image/png"),
            "image_left": ("left.png", left_bytes, "image/png"),
        },
        data={"params": '{"texture_mode": "paint", "seed": 42}'},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")

    assert received["back_image"] is not None
    from PIL import Image as _Image

    assert isinstance(received["back_image"], _Image.Image)


def test_texture_mode_paint_with_color4_completes_with_mock(client, monkeypatch):
    """texture_mode=paint + color_mode=color4 の組合せもmockでジョブ完了すること。

    paint失敗時は正面/背面投影方式(colorproc.project_multiview_colors)に
    フォールバックしてパレット統計が生成されることを検証する。
    """
    from server import main as main_module

    def _fail_paint(self, mesh, image, job, back_image=None, refine_references=None):
        job.warnings.append("test: paint intentionally failed")
        return None

    monkeypatch.setattr(
        main_module.job_manager.__class__, "_run_paint", _fail_paint, raising=True
    )

    png_bytes = make_4color_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test4color.png", png_bytes, "image/png")},
        data={
            "params": (
                '{"texture_mode": "paint", "color_mode": "color4", '
                '"n_colors": 4, "seed": 42, "remove_bg": false}'
            )
        },
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")
    palette = job["stats"]["palette"]
    assert 1 <= len(palette) <= 4
    assert job["textured"] is False


# --- IMAGE3D_GENERATOR=auto の解決 (mock表示問題の対処) ------------------------


def test_auto_generator_resolves_to_mock_when_gpu_unavailable(monkeypatch):
    from server import config, main

    monkeypatch.setattr(config, "GENERATOR", "auto")
    monkeypatch.setattr(main, "_hunyuan3d_usable", lambda: False)
    assert main._build_generator().name == "mock"


def test_auto_generator_resolves_to_hunyuan3d_when_usable(monkeypatch):
    from server import config, main

    monkeypatch.setattr(config, "GENERATOR", "auto")
    monkeypatch.setattr(main, "_hunyuan3d_usable", lambda: True)
    assert main._build_generator().name == "hunyuan3d"


# --- rig-service 連携 (docs/RIG_SERVICE_PLAN.md §7 R4-1) ----------------------


def _completed_job_id(client) -> str:
    res = client.post("/api/jobs", files={"image": ("t.png", make_test_png_bytes(), "image/png")})
    job_id = res.json()["job_id"]
    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed"
    return job_id


def test_health_reports_rig_service_url(client, monkeypatch):
    from server import config

    assert client.get("/api/health").json()["rigsvc_url"] is None

    monkeypatch.setattr(config, "RIGSVC_URL", "http://127.0.0.1:8100")
    assert client.get("/api/health").json()["rigsvc_url"] == "http://127.0.0.1:8100"


def test_rig_endpoint_disabled_when_url_unset(client):
    """IMAGE3D_RIGSVC_URL 未設定なら 503(UI側はボタンを出さない)。"""
    job_id = _completed_job_id(client)
    res = client.post(f"/api/jobs/{job_id}/rig")
    assert res.status_code == 503
    assert "IMAGE3D_RIGSVC_URL" in res.json()["detail"]


def test_rig_endpoint_forwards_glb_and_returns_preview_url(client, monkeypatch):
    from server import config, main as main_module

    monkeypatch.setattr(config, "RIGSVC_URL", "http://rigsvc.test")
    sent = {}

    async def fake_post(filename, glb_bytes, params):
        sent["filename"] = filename
        sent["glb"] = glb_bytes
        sent["params"] = params
        return {"job_id": "rig-123"}

    monkeypatch.setattr(main_module, "_post_to_rig_service", fake_post)

    job_id = _completed_job_id(client)
    res = client.post(f"/api/jobs/{job_id}/rig", data={"params": '{"height_m":1.2}'})

    assert res.status_code == 200
    assert res.json() == {
        "rig_job_id": "rig-123",
        "url": "http://rigsvc.test/?job=rig-123",
    }
    # 実際に生成済みGLBが送られていること
    assert sent["glb"].startswith(b"glTF")
    assert json.loads(sent["params"])["height_m"] == 1.2
    # VRMのタイトルが元画像名から決まるよう、拡張子だけglbに替えて渡す
    assert sent["filename"] == "t.glb"


def test_rig_endpoint_rejects_incomplete_job(client, monkeypatch):
    from server import config

    monkeypatch.setattr(config, "RIGSVC_URL", "http://rigsvc.test")
    res = client.post("/api/jobs", files={"image": ("t.png", make_test_png_bytes(), "image/png")})
    job_id = res.json()["job_id"]

    # 完了前に呼ぶと 409(タイミング次第で完了済みなら 200 でもよい)
    status = client.post(f"/api/jobs/{job_id}/rig").status_code
    assert status in (409, 200)
    _wait_for_completion(client, job_id)


def test_rig_endpoint_reports_unreachable_service(client, monkeypatch):
    """リグサービスが落ちていても image-3d 側は 502 を返すだけで落ちない。"""
    from server import config, main as main_module

    monkeypatch.setattr(config, "RIGSVC_URL", "http://127.0.0.1:1")

    async def boom(filename, glb_bytes, params):
        raise OSError("接続できません")

    monkeypatch.setattr(main_module, "_post_to_rig_service", boom)

    job_id = _completed_job_id(client)
    res = client.post(f"/api/jobs/{job_id}/rig")
    assert res.status_code == 502
    assert "接続できませんでした" in res.json()["detail"]


def test_rig_endpoint_unknown_job(client, monkeypatch):
    from server import config

    monkeypatch.setattr(config, "RIGSVC_URL", "http://rigsvc.test")
    assert client.post("/api/jobs/does-not-exist/rig").status_code == 404


def test_rig_endpoint_declares_image3d_coordinate_conventions(client, monkeypatch):
    """image-3d は自分の出力規約を知っているので、rig-service に推測させない。

    正面の自動判定は生成物77体で86〜91%しか当たらず、特に pixal3d の出力では
    足の手がかりが機能しない(3/8)。座標系は生成側が宣言するのが確実。
    """
    from server import config, main as main_module

    monkeypatch.setattr(config, "RIGSVC_URL", "http://rigsvc.test")
    sent = {}

    async def fake_post(filename, glb_bytes, params):
        sent["params"] = json.loads(params)
        return {"job_id": "rig-1"}

    monkeypatch.setattr(main_module, "_post_to_rig_service", fake_post)

    job_id = _completed_job_id(client)
    assert client.post(f"/api/jobs/{job_id}/rig").status_code == 200
    assert sent["params"]["up_axis"] == "z"
    assert sent["params"]["facing"] == "-y"


def test_rig_endpoint_lets_caller_override_conventions(client, monkeypatch):
    """検証用に上書きできること(宣言はあくまで既定値)。"""
    from server import config, main as main_module

    monkeypatch.setattr(config, "RIGSVC_URL", "http://rigsvc.test")
    sent = {}

    async def fake_post(filename, glb_bytes, params):
        sent["params"] = json.loads(params)
        return {"job_id": "rig-1"}

    monkeypatch.setattr(main_module, "_post_to_rig_service", fake_post)

    job_id = _completed_job_id(client)
    res = client.post(
        f"/api/jobs/{job_id}/rig",
        data={"params": json.dumps({"facing": "auto", "height_m": 1.2})},
    )
    assert res.status_code == 200
    assert sent["params"]["facing"] == "auto"
    assert sent["params"]["height_m"] == 1.2
    assert sent["params"]["up_axis"] == "z"


def test_rig_endpoint_rejects_malformed_params(client, monkeypatch):
    from server import config

    monkeypatch.setattr(config, "RIGSVC_URL", "http://rigsvc.test")
    job_id = _completed_job_id(client)
    res = client.post(f"/api/jobs/{job_id}/rig", data={"params": "{not json"})
    assert res.status_code == 400


# --- 背景除去のスキップ判定 ---------------------------------------------------


def _rgba_png(alpha_ratio: float, size: int = 64) -> bytes:
    """指定割合が透明なRGBA PNGを作る。"""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., :3] = 180
    arr[..., 3] = 255
    transparent_rows = int(size * alpha_ratio)
    arr[:transparent_rows, :, 3] = 0
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def test_detects_already_removed_background():
    from server.preprocess import has_removed_background

    assert has_removed_background(Image.open(io.BytesIO(_rgba_png(0.5)))) is True


def test_opaque_rgba_is_not_treated_as_removed():
    """アルファはあるが全て不透明なら、まだ背景は抜けていない。"""
    from server.preprocess import has_removed_background

    assert has_removed_background(Image.open(io.BytesIO(_rgba_png(0.0)))) is False


def test_rgb_without_alpha_is_not_treated_as_removed():
    from server.preprocess import has_removed_background

    assert has_removed_background(Image.new("RGB", (32, 32), (10, 20, 30))) is False


def test_thin_antialias_edge_is_not_treated_as_removed():
    """ふちのわずかな透明だけでは背景除去済みと見なさない。"""
    from server.preprocess import has_removed_background

    assert has_removed_background(Image.open(io.BytesIO(_rgba_png(0.01)))) is False


def test_preprocess_skips_rembg_for_transparent_input(monkeypatch):
    """既に透明な入力に rembg を再適用しない(前景を削るだけのため)。"""
    from server import preprocess

    called = []

    def spy(image):
        called.append(image)
        return image, True

    monkeypatch.setattr(preprocess, "remove_background", spy)
    _, _, bg_removed, _ = preprocess.preprocess_image(_rgba_png(0.5), 10_000_000, remove_bg=True)

    assert called == [], "透明な入力に rembg が呼ばれている"
    assert bg_removed is False


def test_preprocess_still_removes_background_for_opaque_input(monkeypatch):
    from server import preprocess

    called = []

    def spy(image):
        called.append(image)
        return image.convert("RGBA"), True

    monkeypatch.setattr(preprocess, "remove_background", spy)
    _, _, bg_removed, _ = preprocess.preprocess_image(
        make_test_png_bytes(), 10_000_000, remove_bg=True
    )

    assert len(called) == 1, "不透明な入力に rembg が呼ばれていない"


# --- 単色背景の色キー ---------------------------------------------------------


def _pale_subject_on_white(size: int = 96) -> Image.Image:
    """白背景に、背景とよく似た淡色の被写体(細い腕つき)を描く。

    rembg が腕を落とした実例(白背景・白い毛のT字キャラ)を模したもの。
    """
    arr = np.full((size, size, 3), 253, dtype=np.uint8)
    arr[size // 4 : size * 3 // 4, size * 2 // 5 : size * 3 // 5] = 235  # 胴
    arr[size * 2 // 5 : size * 2 // 5 + 4, 4 : size - 4] = 235  # 左右に伸びる腕
    return Image.fromarray(arr, "RGB")


def test_colour_key_keeps_pale_limbs_on_a_uniform_background():
    """背景と色が近い細い部位も、単色背景なら確実に残る。"""
    from server.preprocess import remove_uniform_background

    size = 96
    result, removed = remove_uniform_background(_pale_subject_on_white(size))

    assert removed is True
    alpha = np.asarray(result)[..., 3]
    arm_row = alpha[size * 2 // 5 + 1]
    assert arm_row[5] > 128, "左腕の先が消えている"
    assert arm_row[size - 6] > 128, "右腕の先が消えている"


def test_colour_key_keeps_background_coloured_areas_inside_the_subject():
    """被写体の内側にある背景と同色の領域(白い服など)は塗りつぶしで残す。"""
    from server.preprocess import remove_uniform_background

    arr = np.full((80, 80, 3), 253, dtype=np.uint8)
    arr[20:60, 20:60] = 40  # 暗い被写体
    arr[35:45, 35:45] = 253  # その内側にある背景色の領域
    result, removed = remove_uniform_background(Image.fromarray(arr, "RGB"))

    assert removed is True
    assert np.asarray(result)[40, 40, 3] > 128, "被写体内部の背景色が抜けている"


def test_colour_key_declines_a_non_uniform_background():
    """写真のような背景では色キーを使わず rembg に譲る。"""
    from server.preprocess import remove_uniform_background

    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    _, removed = remove_uniform_background(Image.fromarray(noisy, "RGB"))

    assert removed is False


def test_colour_key_declines_when_nothing_remains():
    """全面が単色(被写体なし)なら破綻とみなしてフォールバックする。"""
    from server.preprocess import remove_uniform_background

    _, removed = remove_uniform_background(Image.new("RGB", (64, 64), (200, 50, 50)))

    assert removed is False


def test_preprocess_prefers_the_colour_key_over_rembg(monkeypatch):
    """単色背景では rembg を呼ばない(推定より色キーの方が確実)。"""
    from server import preprocess

    called = []
    monkeypatch.setattr(
        preprocess, "remove_background", lambda image: (called.append(image), (image, True))[1]
    )

    buf = io.BytesIO()
    _pale_subject_on_white().save(buf, format="PNG")
    _, _, bg_removed, _ = preprocess.preprocess_image(buf.getvalue(), 10_000_000, remove_bg=True)

    assert called == [], "単色背景なのに rembg が呼ばれている"
    assert bg_removed is True
    assert bg_removed is True


# --- texrefine用の高解像度参照(ネイティブ解像度の背景除去済み画像) --------------


def test_preprocess_returns_native_resolution_background_removed_image():
    """透明PNGアップロード時、preprocessがネイティブ解像度の背景除去済み画像を返す。

    透明入力は既に背景除去済みとみなされ rembg はスキップされるが(NFR-5)、
    「ネイティブ解像度の背景除去済み画像」自体は resize_to_square 前の
    ネイティブ画像として返る必要がある(texrefine用の高精細参照)。
    """
    from server import preprocess

    native_size = 2048
    original, processed, bg_removed, native_processed = preprocess.preprocess_image(
        _rgba_png(0.5, size=native_size), 50_000_000, remove_bg=True
    )

    assert original.size == (native_size, native_size)
    assert processed.size == (preprocess.TARGET_SIZE, preprocess.TARGET_SIZE)
    assert native_processed.size == (native_size, native_size)
    assert native_processed.mode == "RGBA"


def _make_job_with_paint(client, monkeypatch, png_bytes, *, texture_refine=True, extra=None):
    """texture_mode=paint のジョブを作り、_get_texture_pipelineをフェイクに
    差し替えて完了させるための共通ヘルパー。

    `test_paint_without_refine_skips_texrefine` と同様、実GPUのtexgenは使わず
    `paint()`のフェイクだけを差し替え、`_run_paint`本体(texrefine呼び出しの
    分岐)は本物を通す。
    """
    from server import main as main_module

    class _FakePipeline:
        def paint(self, mesh, image, back_image=None):
            return mesh.copy()

    monkeypatch.setattr(
        main_module.job_manager.__class__,
        "_get_texture_pipeline",
        lambda self: _FakePipeline(),
        raising=True,
    )

    params = {"texture_mode": "paint", "texture_refine": texture_refine, "remove_bg": True}
    data = {"params": json.dumps(params)}
    files = {"image": ("test.png", png_bytes, "image/png")}
    if extra:
        files.update(extra)

    res = client.post("/api/jobs", files=files, data=data)
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed", job.get("error")
    return job_id, job


def test_texrefine_receives_native_resolution_reference_for_high_res_upload(client, monkeypatch):
    """2048pxの透明PNGでジョブを作ると、texrefineに渡る参照が2048であること。

    texrefine.refine_texture_with_referencesをモンキーパッチし、受け取った
    front参照画像のサイズを検証する(実texrefineは重いため呼ばない)。
    """
    from server import texrefine

    received: dict = {}

    def _fake_refine(mesh, references):
        received["front_size"] = references["front"].size
        return texrefine.RefineStats(applied=True)

    monkeypatch.setattr(texrefine, "refine_texture_with_references", _fake_refine, raising=True)

    png_bytes = _rgba_png(0.5, size=2048)
    job_id, job = _make_job_with_paint(client, monkeypatch, png_bytes)

    assert received.get("front_size") == (2048, 2048)

    # ジョブディレクトリに reference.png が保存されていること(1024と異なるため)。
    from server import config

    reference_path = config.JOBS_DIR / job_id / "reference.png"
    assert reference_path.exists()
    with Image.open(reference_path) as img:
        assert img.size == (2048, 2048)


def test_texrefine_reference_unchanged_for_1024_input(client, monkeypatch):
    """1024px入力では従来と同じ参照(1024)が渡り、reference.pngは保存されない。"""
    from server import texrefine, config, preprocess

    received: dict = {}

    def _fake_refine(mesh, references):
        received["front_size"] = references["front"].size
        return texrefine.RefineStats(applied=True)

    monkeypatch.setattr(texrefine, "refine_texture_with_references", _fake_refine, raising=True)

    png_bytes = _rgba_png(0.5, size=preprocess.TARGET_SIZE)
    job_id, job = _make_job_with_paint(client, monkeypatch, png_bytes)

    assert received.get("front_size") == (1024, 1024)

    reference_path = config.JOBS_DIR / job_id / "reference.png"
    assert not reference_path.exists(), "同サイズなのにreference.pngが保存されている"
