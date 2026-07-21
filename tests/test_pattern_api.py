"""型紙生成API のテスト (SPEC.md §3.12 / FR-13, Phase 4a〜4c)。

`server/pattern/` 自体は純粋モジュール単体テスト(test_pattern_segment.py /
test_pattern_parts.py 等)で検証済み。ここではmainのアダプタ(2段階フロー・
ジョブディレクトリ接続・バリデーション・ステータスコード)をmock
ジェネレータ+TestClientで検証する。

Phase 4cで型紙生成は2段階になった:
    1. `POST /api/jobs/{id}/pattern/parts` — パーツ自動分解
    2. `POST /api/jobs/{id}/pattern` — パーツ単位のパネル分割→平坦化→SVG
       (パーツ分解が未実行なら内部で自動実行される)
"""
from __future__ import annotations

import time

import pytest
import trimesh
from fastapi.testclient import TestClient

from tests.conftest import make_test_png_bytes


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from server import config

    data_dir = tmp_path / "data"
    jobs_dir = data_dir / "jobs"
    jobs_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "JOBS_DIR", jobs_dir)

    from server import main as main_module

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


def _create_completed_job(client, remove_bg: bool = True) -> str:
    png_bytes = make_test_png_bytes()
    files = {"image": ("test.png", png_bytes, "image/png")}
    data = {}
    if not remove_bg:
        import json as _json

        data["params"] = _json.dumps({"remove_bg": False})
    res = client.post("/api/jobs", files=files, data=data)
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    job = _wait_for_completion(client, job_id)
    assert job["status"] == "completed"
    return job_id


# --- Phase 4c: ステップ1 パーツ分解 ------------------------------------------
def test_pattern_parts_e2e(client):
    job_id = _create_completed_job(client)

    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"n_parts": 0})
    assert res.status_code == 200
    data = res.json()
    assert data["job_id"] == job_id
    assert data["n_parts_requested"] == 0
    assert data["n_parts_actual"] >= 1
    assert len(data["parts"]) == data["n_parts_actual"]
    for part in data["parts"]:
        assert "part_id" in part
        assert "n_faces" in part
        assert "area_mm2" in part
        assert "volume_mm3" in part
        assert "mean_thickness_mm" in part
        assert "connected" in part
        assert "watertight_after_cap" in part

    # pattern_parts.json 取得
    res_json = client.get(f"/api/jobs/{job_id}/pattern_parts.json")
    assert res_json.status_code == 200
    assert res_json.json()["n_parts_actual"] == data["n_parts_actual"]

    # pattern_parts_preview.glb 取得。trimeshで再読込できること。
    res_glb = client.get(f"/api/jobs/{job_id}/pattern_parts_preview.glb")
    assert res_glb.status_code == 200
    assert res_glb.headers["content-type"] == "model/gltf-binary"

    import io

    loaded = trimesh.load(io.BytesIO(res_glb.content), file_type="glb")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        assert len(meshes) >= 1
    else:
        assert len(loaded.faces) > 0


def test_pattern_parts_llm_guidance_used_when_endpoint_configured(client, monkeypatch):
    """`IMAGE3D_LLM_ENDPOINT`設定+`detect_parts`成功時、guidance="llm"となり
    パーツにLLMが返した名前が対応付けられて pattern_parts.json に載ることを
    monkeypatchで検証する(実LLMサーバは呼ばない)。"""
    from server import config, llm_parts

    monkeypatch.setattr(config, "IMAGE3D_LLM_ENDPOINT", "http://fake-llm.invalid:1234")

    # mockジェネレータの入力画像は128x128の不透明正方形が1024x1024
    # キャンバス中央にレターボックス配置される(remove_bg=False かつ
    # `resize_to_square` の仕様)。中央付近を上下2分するbboxにすることで、
    # 投影後に両パーツへ有意な面積が振り分けられるようにする。
    fake_parts = [
        {"name": "頭", "bbox": [0.0, 0.0, 1.0, 0.5]},
        {"name": "胴体", "bbox": [0.0, 0.5, 1.0, 1.0]},
    ]

    def fake_detect_parts(image_rgba, endpoint, timeout=60):
        return fake_parts

    monkeypatch.setattr(llm_parts, "detect_parts", fake_detect_parts)

    # remove_bg=Falseで作成し、入力画像を全域不透明のまま保つ(rembgが
    # 単色テスト画像を被写体なしと判定してほぼ全画素透明にしてしまい、
    # bboxが有意なチャンクを作れなくなるのを避けるため)。
    job_id = _create_completed_job(client, remove_bg=False)
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"n_parts": 0, "use_llm": True})
    assert res.status_code == 200
    data = res.json()

    assert data["guidance"] == "llm"
    assert data["use_llm"] is True
    assert data["use_image"] is False
    assert data["n_llm_parts_detected"] == 2

    detected_names = {p["name"] for p in fake_parts}
    part_names_in_result = {p.get("name") for p in data["parts"] if p.get("name")}
    # 少なくとも1つはLLM検出名がパーツに対応付けられていること
    assert part_names_in_result & detected_names

    # pattern_parts.json にも guidance/name が保存されていること
    res_json = client.get(f"/api/jobs/{job_id}/pattern_parts.json")
    assert res_json.status_code == 200
    saved = res_json.json()
    assert saved["guidance"] == "llm"


def test_pattern_parts_falls_back_to_color_when_llm_endpoint_unset(client, monkeypatch):
    """`IMAGE3D_LLM_ENDPOINT`が未設定の場合、`use_llm=true`を指定しても
    LLM誘導は使われず、色領域誘導(guidance="color")にフォールバックする
    ことを検証する(既定のテストジョブ画像は不透明単色PNGのため
    color_regions=1になりうるが、いずれにせよLLMは使われない)。"""
    from server import config

    monkeypatch.setattr(config, "IMAGE3D_LLM_ENDPOINT", "")

    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"n_parts": 0, "use_llm": True})
    assert res.status_code == 200
    data = res.json()
    assert data["guidance"] != "llm"
    assert data["use_llm"] is False


def test_pattern_parts_falls_back_to_color_when_llm_detect_fails(client, monkeypatch):
    """LLMエンドポイントは設定されているが `detect_parts` が失敗(None)した
    場合、色領域誘導へフォールバックすることを検証する。"""
    from server import config, llm_parts

    monkeypatch.setattr(config, "IMAGE3D_LLM_ENDPOINT", "http://fake-llm.invalid:1234")

    def fake_detect_parts_fail(image_rgba, endpoint, timeout=60):
        return None

    monkeypatch.setattr(llm_parts, "detect_parts", fake_detect_parts_fail)

    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"n_parts": 0, "use_llm": True})
    assert res.status_code == 200
    data = res.json()
    assert data["guidance"] != "llm"
    assert data["use_llm"] is False


@pytest.mark.parametrize("bad_value", ["true", 1, None])
def test_pattern_parts_rejects_bad_use_llm(client, bad_value):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"n_parts": 0, "use_llm": bad_value})
    if bad_value is None:
        # JSONのnullはbodyのuse_llmキー自体は渡るが、bool型チェックでNoneは弾かれる
        assert res.status_code == 400
    else:
        assert res.status_code == 400


def test_pattern_parts_with_hint(client):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"n_parts": 3})
    assert res.status_code == 200
    data = res.json()
    assert data["n_parts_requested"] == 3
    # ヒントは厳密保証ではないが、mockのトーラス結び目(単一の塊)でも
    # くびれ誘導の階層分割により2パーツ以上に応えられること
    assert data["n_parts_actual"] >= 2


@pytest.mark.parametrize("bad_value", [1, 11, -2, "3", True, 2.5])
def test_pattern_parts_rejects_bad_n_parts(client, bad_value):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"n_parts": bad_value})
    assert res.status_code == 400


def test_pattern_parts_rejects_incomplete_job(client):
    png_bytes = make_test_png_bytes()
    res = client.post("/api/jobs", files={"image": ("test.png", png_bytes, "image/png")})
    job_id = res.json()["job_id"]

    from server import main as main_module

    job = main_module.job_manager.get_job(job_id)
    job.status = "generating"

    res_parts = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"n_parts": 0})
    assert res_parts.status_code == 409


def test_pattern_parts_404s(client):
    res = client.post("/api/jobs/does-not-exist/pattern/parts", json={"n_parts": 0})
    assert res.status_code == 404

    job_id = _create_completed_job(client)
    assert client.get(f"/api/jobs/{job_id}/pattern_parts.json").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/pattern_parts_preview.glb").status_code == 404


# --- 手動シード誘導 (guidance="manual") --------------------------------------
def _job_bbox_mm(client, job_id):
    """mockジェネレータのモデルはXY中心が原点、Z底面がz=0に接地(meshproc._scale_to_height)。
    `bbox_mm`(各軸の全長)から、頭頂付近・足元付近など妥当なシード座標を組み立てる
    のに使う。"""
    res = client.get(f"/api/jobs/{job_id}")
    assert res.status_code == 200
    return res.json()["stats"]["bbox_mm"]


def test_pattern_parts_manual_seeds_e2e(client):
    job_id = _create_completed_job(client)
    bbox = _job_bbox_mm(client, job_id)
    z_top = bbox[2] * 0.9
    z_bottom = bbox[2] * 0.1

    seeds = [
        {"x": 0.0, "y": 0.0, "z": z_top, "name": "頭"},
        {"x": 0.0, "y": 0.0, "z": z_bottom, "name": "胴体"},
    ]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 200
    data = res.json()

    assert data["guidance"] == "manual"
    assert data["use_image"] is False
    assert data["use_llm"] is False
    assert data["n_parts_actual"] == 2
    assert len(data["parts"]) == 2
    names = {p.get("name") for p in data["parts"]}
    assert names == {"頭", "胴体"}
    assert data["seeds"][0]["name"] == "頭"
    assert data["seeds"][1]["name"] == "胴体"

    # pattern_parts.json にも guidance="manual" とseedsが保存されること
    res_json = client.get(f"/api/jobs/{job_id}/pattern_parts.json")
    assert res_json.status_code == 200
    saved = res_json.json()
    assert saved["guidance"] == "manual"
    assert len(saved["seeds"]) == 2

    # pattern_parts_preview.glb も既存形式のまま出力される
    res_glb = client.get(f"/api/jobs/{job_id}/pattern_parts_preview.glb")
    assert res_glb.status_code == 200
    assert res_glb.headers["content-type"] == "model/gltf-binary"


def test_pattern_parts_manual_seeds_default_names(client):
    """nameを省略/空文字にした場合、part_1のような自動名が振られる。"""
    job_id = _create_completed_job(client)
    bbox = _job_bbox_mm(client, job_id)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.9},
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.1, "name": ""},
    ]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 200
    data = res.json()
    assert data["seeds"][0]["name"] == "part_1"
    assert data["seeds"][1]["name"] == "part_2"


def test_pattern_parts_manual_seeds_ignore_n_parts_and_guidance_flags(client):
    """seeds指定時はn_parts/use_image/use_llmを渡しても無視される。"""
    job_id = _create_completed_job(client)
    bbox = _job_bbox_mm(client, job_id)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.9, "name": "頭"},
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.1, "name": "胴体"},
    ]
    res = client.post(
        f"/api/jobs/{job_id}/pattern/parts",
        json={"seeds": seeds, "n_parts": 6, "use_image": True, "use_llm": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["guidance"] == "manual"
    assert data["n_parts_actual"] == 2


@pytest.mark.parametrize("bad_seeds", [[], [{"x": 0, "y": 0, "z": 0}]])
def test_pattern_parts_rejects_too_few_seeds(client, bad_seeds):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": bad_seeds})
    assert res.status_code == 400


def test_pattern_parts_rejects_too_many_seeds(client):
    """シード総数の上限は48(同名マージ対応で緩和済み)。49個は400。"""
    job_id = _create_completed_job(client)
    seeds = [{"x": float(i), "y": 0.0, "z": 0.0, "name": f"p{i % 4}"} for i in range(49)]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 400


def test_pattern_parts_rejects_too_many_unique_names(client):
    """ユニーク名(=パーツ数)はプレビューパレットの20色に合わせ最大20。
    名前なしシードは自動名(part_N)が振られ各々ユニークになるため、
    21個の無名シードはパーツ数21となり400。"""
    job_id = _create_completed_job(client)
    seeds = [{"x": float(i), "y": 0.0, "z": 0.0} for i in range(21)]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 400


def test_pattern_parts_rejects_single_unique_name(client):
    """全シードが同名(=パーツ数1)は分割にならないため400。"""
    job_id = _create_completed_job(client)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": 10.0, "name": "胴体"},
        {"x": 0.0, "y": 0.0, "z": 90.0, "name": "胴体"},
    ]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 400


def test_pattern_parts_same_name_seeds_merge(client):
    """同名シード(「胴体」を正面と背面に1点ずつ)が1パーツに統合され、
    パーツ数=ユニーク名数になることを検証する。"""
    job_id = _create_completed_job(client)
    bbox = _job_bbox_mm(client, job_id)
    # mockモデルはXY中心原点なので、y正負を正面/背面に見立てる
    seeds = [
        {"x": 0.0, "y": bbox[1] * 0.4, "z": bbox[2] * 0.3, "name": "胴体"},
        {"x": 0.0, "y": -bbox[1] * 0.4, "z": bbox[2] * 0.3, "name": "胴体"},
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.9, "name": "頭"},
    ]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 200
    data = res.json()
    assert data["guidance"] == "manual"
    assert data["n_parts_requested"] == 2  # ユニーク名数
    assert data["n_parts_actual"] == 2
    names = {p.get("name") for p in data["parts"]}
    assert names == {"胴体", "頭"}
    # seeds記録は全点(3点)がそのまま残る(再現性のため)
    assert len(data["seeds"]) == 3


# --- 背面への仮想シード自動伝播 (propagate_back) ------------------------------
def test_pattern_parts_propagate_back_fields_in_response(client):
    """既定(propagate_back=true)のレスポンス/pattern_parts.json に
    `propagate_back`/`n_virtual_seeds`/`virtual_seeds` が記録されることを検証する。"""
    job_id = _create_completed_job(client)
    bbox = _job_bbox_mm(client, job_id)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.9, "name": "頭"},
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.1, "name": "胴体"},
    ]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 200
    data = res.json()
    assert data["propagate_back"] is True
    assert isinstance(data["n_virtual_seeds"], int)
    assert data["n_virtual_seeds"] == len(data["virtual_seeds"])
    for vs in data["virtual_seeds"]:
        assert {"x", "y", "z", "name"} <= set(vs.keys())
    # 仮想シードを得られなかったグループの棄却理由が記録される(診断用)
    assert isinstance(data["virtual_seed_skips"], list)
    for sk in data["virtual_seed_skips"]:
        assert {"name", "reason"} <= set(sk.keys())
        assert sk["reason"] in ("no_hit", "tunnel", "taken", "ownership", "balance")

    # pattern_parts.json にも保存される
    res_json = client.get(f"/api/jobs/{job_id}/pattern_parts.json")
    assert res_json.status_code == 200
    saved = res_json.json()
    assert saved["propagate_back"] is True
    assert saved["n_virtual_seeds"] == data["n_virtual_seeds"]
    assert saved["virtual_seed_skips"] == data["virtual_seed_skips"]


def test_pattern_parts_propagate_back_false_disables_virtual_seeds(client):
    """propagate_back=false では仮想シードが生成されない(従来動作)。"""
    job_id = _create_completed_job(client)
    bbox = _job_bbox_mm(client, job_id)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.9, "name": "頭"},
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.1, "name": "胴体"},
    ]
    res = client.post(
        f"/api/jobs/{job_id}/pattern/parts",
        json={"seeds": seeds, "propagate_back": False},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["propagate_back"] is False
    assert data["n_virtual_seeds"] == 0
    assert data["virtual_seeds"] == []


@pytest.mark.parametrize("bad_value", ["true", 1, None])
def test_pattern_parts_rejects_bad_propagate_back(client, bad_value):
    job_id = _create_completed_job(client)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": 10.0, "name": "a"},
        {"x": 0.0, "y": 0.0, "z": 90.0, "name": "b"},
    ]
    res = client.post(
        f"/api/jobs/{job_id}/pattern/parts",
        json={"seeds": seeds, "propagate_back": bad_value},
    )
    assert res.status_code == 400


# --- 境界の平面フィット正則化 (planar_boundaries) -----------------------------
def test_pattern_parts_planar_boundaries_fields_in_response(client):
    """既定(planar_boundaries=true)のレスポンス/pattern_parts.json に
    `planar_boundaries`/`planar_fit`(ペアごとの適用結果)が記録される。"""
    job_id = _create_completed_job(client)
    bbox = _job_bbox_mm(client, job_id)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.9, "name": "頭"},
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.1, "name": "胴体"},
    ]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 200
    data = res.json()
    assert data["planar_boundaries"] is True
    assert isinstance(data["planar_fit"], list)
    for entry in data["planar_fit"]:
        assert "parts" in entry and "applied" in entry
        # 適用されたペアにはロバストフィットのアンカー情報が記録される
        if entry["applied"]:
            assert "anchor_rms" in entry and "n_anchors" in entry

    res_json = client.get(f"/api/jobs/{job_id}/pattern_parts.json")
    assert res_json.status_code == 200
    saved = res_json.json()
    assert saved["planar_boundaries"] is True
    assert saved["planar_fit"] == data["planar_fit"]


def test_pattern_parts_planar_boundaries_false_disables(client):
    """planar_boundaries=false では平面化が行われず planar_fit が空になる
    (従来動作)。"""
    job_id = _create_completed_job(client)
    bbox = _job_bbox_mm(client, job_id)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.9, "name": "頭"},
        {"x": 0.0, "y": 0.0, "z": bbox[2] * 0.1, "name": "胴体"},
    ]
    res = client.post(
        f"/api/jobs/{job_id}/pattern/parts",
        json={"seeds": seeds, "planar_boundaries": False},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["planar_boundaries"] is False
    assert data["planar_fit"] == []


@pytest.mark.parametrize("bad_value", ["true", 1, None])
def test_pattern_parts_rejects_bad_planar_boundaries(client, bad_value):
    job_id = _create_completed_job(client)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": 10.0, "name": "a"},
        {"x": 0.0, "y": 0.0, "z": 90.0, "name": "b"},
    ]
    res = client.post(
        f"/api/jobs/{job_id}/pattern/parts",
        json={"seeds": seeds, "planar_boundaries": bad_value},
    )
    assert res.status_code == 400


@pytest.mark.parametrize(
    "bad_seed",
    [
        {"x": "0", "y": 0.0, "z": 0.0},
        {"x": None, "y": 0.0, "z": 0.0},
        {"y": 0.0, "z": 0.0},
    ],
)
def test_pattern_parts_rejects_non_numeric_seed_coords(client, bad_seed):
    job_id = _create_completed_job(client)
    seeds = [bad_seed, {"x": 1.0, "y": 1.0, "z": 1.0}]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 400


def test_pattern_parts_rejects_nan_seed_coord(client):
    """NaNは標準JSONでは表現できないが(json.dumpsが送出できない)、Python
    のjsonデコーダはリテラルNaNを非標準拡張として受け付けてしまうため、
    サーバ側の有限値チェック(math.isfinite)で確実に弾かれることを
    生JSON文字列送信で検証する。"""
    job_id = _create_completed_job(client)
    body = (
        '{"seeds": [{"x": NaN, "y": 0.0, "z": 0.0}, {"x": 1.0, "y": 1.0, "z": 1.0}]}'
    )
    res = client.post(
        f"/api/jobs/{job_id}/pattern/parts",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 400


def test_pattern_parts_rejects_non_string_seed_name(client):
    job_id = _create_completed_job(client)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "name": 123},
        {"x": 1.0, "y": 1.0, "z": 1.0},
    ]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 400


def test_pattern_parts_manual_seeds_duplicate_face_returns_400(client):
    """同一面にスナップする(=事実上同一点の)シードはlabels_from_seedsが
    ValueErrorを送出し、APIは400として返す。"""
    job_id = _create_completed_job(client)
    seeds = [
        {"x": 0.0, "y": 0.0, "z": 1.0, "name": "a"},
        {"x": 0.0, "y": 0.0, "z": 1.0, "name": "b"},
    ]
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": seeds})
    assert res.status_code == 400


def test_pattern_parts_manual_seeds_not_dict_item_rejected(client):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"seeds": ["notadict", {"x": 1, "y": 1, "z": 1}]})
    assert res.status_code == 400


# --- Phase 4c: ステップ2 型紙生成(parts構造) --------------------------------
def _assert_pattern_response_shape(data, job_id):
    assert data["job_id"] == job_id
    assert data["n_parts"] >= 1
    assert len(data["parts"]) == data["n_parts"]
    assert data["n_panels_total"] >= 2
    assert 0 <= data["n_panels_flattened"] <= data["n_panels_total"]

    n_panels_seen = 0
    for part in data["parts"]:
        assert "part_id" in part
        assert "part_label" in part
        assert "color_hex" in part
        assert "stats" in part
        assert "panels" in part
        assert part["n_panels"] == len(part["panels"])
        for panel in part["panels"]:
            n_panels_seen += 1
            assert "panel_id" in panel
            assert "panel_no" in panel
            assert "n_faces" in panel
            assert "area_mm2" in panel
            assert "boundary_loops" in panel
            assert "disk_topology" in panel
            assert "flatten_failed" in panel
            assert "has_attachment_opening" in panel
            if not panel["flatten_failed"]:
                assert "distortion" in panel
                assert "edge_length_ratio_mean" in panel["distortion"]
    assert n_panels_seen == data["n_panels_total"]


def test_pattern_generation_e2e(client):
    job_id = _create_completed_job(client)

    # パーツ分解→型紙生成の2段階フロー
    res_parts = client.post(f"/api/jobs/{job_id}/pattern/parts", json={"n_parts": 0})
    assert res_parts.status_code == 200

    res = client.post(f"/api/jobs/{job_id}/pattern", json={"n_panels": 4})
    assert res.status_code == 200
    data = res.json()
    assert data["n_panels_max_per_part"] == 4
    _assert_pattern_response_shape(data, job_id)
    # 事前のパーツ分解結果が使われている
    assert data["n_parts"] == res_parts.json()["n_parts_actual"]

    # mockのトーラス結び目は全パネルが平坦化可能(円盤位相修復込み)であるべき
    assert data["n_panels_flattened"] == data["n_panels_total"]

    # pattern.json取得(parts構造)
    res_json = client.get(f"/api/jobs/{job_id}/pattern.json")
    assert res_json.status_code == 200
    assert res_json.json()["n_panels_total"] == data["n_panels_total"]
    assert "parts" in res_json.json()

    # pattern_preview.glb取得。trimeshで再読込できること。
    res_glb = client.get(f"/api/jobs/{job_id}/pattern_preview.glb")
    assert res_glb.status_code == 200
    assert res_glb.headers["content-type"] == "model/gltf-binary"

    import io

    loaded = trimesh.load(io.BytesIO(res_glb.content), file_type="glb")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        assert len(meshes) >= 1
    else:
        assert len(loaded.faces) > 0


def test_pattern_generation_autoruns_parts_decomposition(client):
    """パーツ分解が未実行でも pattern POST が内部で自動実行する。"""
    job_id = _create_completed_job(client)

    res = client.post(f"/api/jobs/{job_id}/pattern", json={})
    assert res.status_code == 200
    _assert_pattern_response_shape(res.json(), job_id)

    # 自動実行されたパーツ分解の成果物も保存されている
    assert client.get(f"/api/jobs/{job_id}/pattern_parts.json").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/pattern_parts_preview.glb").status_code == 200


def test_pattern_generation_uses_defaults_with_empty_body(client):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern", json={})
    assert res.status_code == 200
    data = res.json()
    assert data["n_panels_max_per_part"] == 4
    assert data["use_colors"] is True
    assert data["seam_allowance_mm"] == 7.0


def test_pattern_generation_rejects_incomplete_job(client):
    png_bytes = make_test_png_bytes()
    res = client.post(
        "/api/jobs",
        files={"image": ("test.png", png_bytes, "image/png")},
    )
    job_id = res.json()["job_id"]

    # 完了前(queued直後)にリクエストするとレースになりうるため、
    # 明示的にjobオブジェクトのstatusを差し替える。
    from server import main as main_module

    job = main_module.job_manager.get_job(job_id)
    job.status = "generating"

    res_pattern = client.post(f"/api/jobs/{job_id}/pattern", json={"n_panels": 4})
    assert res_pattern.status_code == 409


def test_pattern_generation_rejects_out_of_range_n_panels(client):
    job_id = _create_completed_job(client)

    # Phase 4cでn_panelsは「パーツあたりの最大パネル数」(2〜6)
    res = client.post(f"/api/jobs/{job_id}/pattern", json={"n_panels": 1})
    assert res.status_code == 400

    res2 = client.post(f"/api/jobs/{job_id}/pattern", json={"n_panels": 7})
    assert res2.status_code == 400


def test_pattern_generation_rejects_bad_smooth_iterations(client):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern", json={"smooth_iterations": -1})
    assert res.status_code == 400


def test_pattern_json_and_glb_404_before_generation(client):
    job_id = _create_completed_job(client)
    res = client.get(f"/api/jobs/{job_id}/pattern.json")
    assert res.status_code == 404
    res2 = client.get(f"/api/jobs/{job_id}/pattern_preview.glb")
    assert res2.status_code == 404
    res3 = client.get(f"/api/jobs/{job_id}/pattern.svg")
    assert res3.status_code == 404


def test_pattern_generation_nonexistent_job_404(client):
    res = client.post("/api/jobs/does-not-exist/pattern", json={"n_panels": 4})
    assert res.status_code == 404


# --- Phase 4b+4c: 実寸SVG(パーツグループ化) ---------------------------------
def test_pattern_generation_produces_svg_with_part_groups(client):
    job_id = _create_completed_job(client)

    res = client.post(
        f"/api/jobs/{job_id}/pattern", json={"n_panels": 4, "seam_allowance_mm": 7.0}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["seam_allowance_mm"] == 7.0

    res_svg = client.get(f"/api/jobs/{job_id}/pattern.svg")
    assert res_svg.status_code == 200
    assert res_svg.headers["content-type"] == "image/svg+xml"

    import xml.etree.ElementTree as ET

    root = ET.fromstring(res_svg.content)
    assert root.tag.endswith("svg")
    assert root.attrib["width"].endswith("mm")

    svg_text = res_svg.content.decode("utf-8")
    # パーツグループ化: 「部位N-PM」ラベル・data-part-id属性・パーツ対応表
    assert "部位1-P1" in svg_text
    assert 'data-part-id="' in svg_text
    assert "パーツ対応表" in svg_text
    # 注: mockのトーラス結び目は単一パーツ(取付口なし)になるため、
    # 取付口ラベルの検証は tests/test_pattern_parts.py の雪だるま形状で行う。


def test_pattern_generation_default_seam_allowance(client):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern", json={})
    assert res.status_code == 200
    assert res.json()["seam_allowance_mm"] == 7.0


@pytest.mark.parametrize("bad_value", [0, 31, -5, "7", True])
def test_pattern_generation_rejects_out_of_range_seam_allowance(client, bad_value):
    job_id = _create_completed_job(client)
    res = client.post(f"/api/jobs/{job_id}/pattern", json={"seam_allowance_mm": bad_value})
    assert res.status_code == 400


# --- /api/health の llm_parts_available (SPEC.md §3.12 第3層誘導) ------------
def test_health_reports_llm_parts_available_false_by_default(client, monkeypatch):
    from server import config

    monkeypatch.setattr(config, "IMAGE3D_LLM_ENDPOINT", "")
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert "llm_parts_available" in data
    assert data["llm_parts_available"] is False


def test_health_reports_llm_parts_available_true_when_endpoint_configured(client, monkeypatch):
    from server import config

    monkeypatch.setattr(config, "IMAGE3D_LLM_ENDPOINT", "http://fake-llm.invalid:1234")
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["llm_parts_available"] is True
