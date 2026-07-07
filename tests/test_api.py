"""API ライフサイクルテスト (IMPLEMENTATION_PLAN.md タスク1-8 (b)(c))。

FastAPI TestClient + mockジェネレータでジョブ作成→完了までポーリング→
GET model.glb / download?format=stl を検証する。STL出力はtrimeshで再読込し
watertight・高さ(mm)を機械検証する(DEVELOPMENT_POLICY.md §5)。
"""
import base64
import io
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


def test_texture_mode_paint_job_completes_with_mock(client, monkeypatch):
    """mock環境でtexture_mode=paintを指定してもジョブは完了すること。

    texgen(paintパイプライン)の実ロードはモデルDL(数GB)を伴うため、
    ユニットテストでは `JobManager._run_paint` をモンキーパッチして
    「paint失敗→graceful degradation」経路を検証する(実際のpaint成功経路は
    実モデル検証(README/報告参照)でカバーする)。
    """
    from server import main as main_module

    def _fail_paint(self, mesh, image, job):
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


def test_texture_mode_paint_with_color4_completes_with_mock(client, monkeypatch):
    """texture_mode=paint + color_mode=color4 の組合せもmockでジョブ完了すること。

    paint失敗時は従来の正面投影方式(colorproc.project_colors)に
    フォールバックしてパレット統計が生成されることを検証する。
    """
    from server import main as main_module

    def _fail_paint(self, mesh, image, job):
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
