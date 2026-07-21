"""FastAPIエントリポイント (SPEC.md §5 API仕様)。"""
from __future__ import annotations

import base64
import io
import json
import logging
import platform
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, sheet
from .generators.base import GenerationParams
from .generators.mock import MockGenerator
from .jobs import EXPORT_FORMATS, EXTRA_VIEW_LABELS, STATUS_COMPLETED, JobManager
from .preprocess import InvalidImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _hunyuan3d_usable() -> bool:
    """hunyuan3dジェネレータが動作可能か(モデルの実ロードはせず判定)。"""
    import importlib.util

    if importlib.util.find_spec("hy3dgen") is None:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _build_generator():
    name = config.GENERATOR
    if name == "auto":
        if _hunyuan3d_usable():
            name = "hunyuan3d"
        else:
            name = "mock"
            logger.warning(
                "IMAGE3D_GENERATOR=auto: GPU/hy3dgen が利用できないため mock で起動します。"
                "アップロード画像は3D化されず、テスト用形状が返ります。"
            )
    if name == "mock":
        return MockGenerator()
    if name == "hunyuan3d":
        from .generators.hunyuan3d import Hunyuan3DGenerator

        return Hunyuan3DGenerator()
    if name == "pixal3d":
        # Pixal3Dは専用venv (.venv-pixal3d) での起動が前提のため、autoでは解決せず
        # IMAGE3D_GENERATOR=pixal3d の明示指定でのみ使用する (SPEC.md §3.3)。
        from .generators.pixal3d import Pixal3DGenerator

        return Pixal3DGenerator()
    raise ValueError(f"Unknown generator: {name}")


job_manager = JobManager(_build_generator())


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    job_manager.load_history()
    await job_manager.start_worker()
    yield
    await job_manager.stop_worker()


app = FastAPI(title="Image-3D", lifespan=lifespan)


def _parse_params(params_json: Optional[str]) -> GenerationParams:
    data = {}
    if params_json:
        try:
            data = json.loads(params_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"paramsのJSONが不正です: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="paramsはJSONオブジェクトである必要があります。")

    defaults = GenerationParams(
        steps=config.DEFAULT_STEPS,
        guidance_scale=config.DEFAULT_GUIDANCE_SCALE,
        octree_resolution=config.DEFAULT_OCTREE_RESOLUTION,
        seed=None,
        remove_bg=config.DEFAULT_REMOVE_BG,
        target_height_mm=config.DEFAULT_TARGET_HEIGHT_MM,
        max_faces=config.DEFAULT_MAX_FACES,
        color_mode="none",
        n_colors=4,
        texture_mode="none",
    )

    steps = data.get("steps", defaults.steps)
    guidance_scale = data.get("guidance_scale", defaults.guidance_scale)
    octree_resolution = data.get("octree_resolution", defaults.octree_resolution)
    seed = data.get("seed", defaults.seed)
    remove_bg = data.get("remove_bg", defaults.remove_bg)
    target_height_mm = data.get("target_height_mm", defaults.target_height_mm)
    max_faces = data.get("max_faces", defaults.max_faces)
    color_mode = data.get("color_mode", defaults.color_mode)
    n_colors = data.get("n_colors", defaults.n_colors)
    texture_mode = data.get("texture_mode", defaults.texture_mode)

    if octree_resolution not in config.ALLOWED_OCTREE_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"octree_resolutionは{sorted(config.ALLOWED_OCTREE_RESOLUTIONS)}のいずれかである必要があります。",
        )
    if not isinstance(steps, int) or steps <= 0:
        raise HTTPException(status_code=400, detail="stepsは正の整数である必要があります。")
    if not isinstance(target_height_mm, (int, float)) or target_height_mm <= 0:
        raise HTTPException(status_code=400, detail="target_height_mmは正の数である必要があります。")
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise HTTPException(status_code=400, detail="max_facesは正の整数である必要があります。")
    if color_mode not in ("none", "color4"):
        raise HTTPException(
            status_code=400, detail="color_modeは'none'または'color4'である必要があります。"
        )
    if not isinstance(n_colors, int) or not (2 <= n_colors <= 4):
        raise HTTPException(status_code=400, detail="n_colorsは2〜4の整数である必要があります。")
    if texture_mode not in ("none", "paint"):
        raise HTTPException(
            status_code=400, detail="texture_modeは'none'または'paint'である必要があります。"
        )

    return GenerationParams(
        steps=steps,
        guidance_scale=guidance_scale,
        octree_resolution=octree_resolution,
        seed=seed,
        remove_bg=remove_bg,
        target_height_mm=target_height_mm,
        max_faces=max_faces,
        color_mode=color_mode,
        n_colors=n_colors,
        texture_mode=texture_mode,
    )


async def _read_and_validate_upload(image: UploadFile, label: str) -> bytes:
    if image.content_type not in config.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"対応していないファイル形式です({label}: {image.content_type})。PNG/JPEG/WebPを使用してください。",
        )

    data = await image.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"ファイルサイズが上限({config.MAX_UPLOAD_BYTES // (1024 * 1024)}MB)を超えています({label})。",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail=f"空のファイルです({label})。")

    try:
        from .preprocess import load_and_validate_image

        load_and_validate_image(data, config.MAX_UPLOAD_BYTES)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=f"{label}: {exc}") from exc

    return data


@app.post("/api/jobs")
async def create_job(
    image: UploadFile = File(...),
    params: Optional[str] = Form(None),
    image_back: Optional[UploadFile] = File(None),
    image_left: Optional[UploadFile] = File(None),
    image_right: Optional[UploadFile] = File(None),
):
    data = await _read_and_validate_upload(image, "image")

    gen_params = _parse_params(params)

    # 追加ビュー(SPEC.md §3.8 / FR-9): 任意のmultipartフィールド
    # image_back / image_left / image_right を受け付ける。
    extra_uploads = {"back": image_back, "left": image_left, "right": image_right}
    extra_images: dict[str, bytes] = {}
    for view, upload in extra_uploads.items():
        if upload is None:
            continue
        extra_images[view] = await _read_and_validate_upload(upload, f"image_{view}")

    job = await job_manager.create_job(
        data,
        gen_params,
        original_filename=image.filename,
        extra_images=extra_images or None,
    )
    return {"job_id": job.job_id}


@app.get("/api/jobs")
async def list_jobs():
    return [job.to_dict() for job in job_manager.list_jobs()]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/input")
async def get_job_input(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.input_image_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="入力画像がまだありません。")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/model.glb")
async def get_job_model_glb(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    if job.status != STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail=f"ジョブは未完了です(status={job.status})。")
    path = job.model_path("glb")
    if not path.exists():
        raise HTTPException(status_code=404, detail="モデルファイルが見つかりません。")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{job_id}.glb")


_DOWNLOAD_MEDIA_TYPES = {
    "stl": "model/stl",
    "3mf": "model/3mf",
    "obj": "text/plain",
    "glb": "model/gltf-binary",
}


@app.get("/api/jobs/{job_id}/download")
async def download_job_model(job_id: str, format: str = "stl"):
    fmt = format.lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(
            status_code=400, detail=f"formatは{sorted(EXPORT_FORMATS)}のいずれかである必要があります。"
        )
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    if job.status != STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail=f"ジョブは未完了です(status={job.status})。")

    # カラーモード時、3MFは色ごとに分割されたマルチオブジェクト版を返す
    if fmt == "3mf" and job.is_color_mode():
        color_path = job.model_color_3mf_path()
        if color_path.exists():
            return FileResponse(
                color_path,
                media_type=_DOWNLOAD_MEDIA_TYPES[fmt],
                filename=f"{job_id}_color.3mf",
            )

    path = job.model_path(fmt)
    if not path.exists():
        raise HTTPException(status_code=404, detail="モデルファイルが見つかりません。")
    return FileResponse(
        path,
        media_type=_DOWNLOAD_MEDIA_TYPES[fmt],
        filename=f"{job_id}.{fmt}",
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    ok = job_manager.delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    return {"deleted": True}


# --- ぬいぐるみ型紙生成 (SPEC.md §3.12 / FR-13, Phase 4a〜4c) ----------------
# `server/pattern/` は純粋モジュール(numpy/scipy/trimeshのみに依存し、
# server内の他モジュールを一切importしない)。ここではジョブディレクトリ・
# パラメータバリデーション等アプリ固有の事情を扱う薄いアダプタとして接続する。
# Phase 4c: 型紙生成は2段階(①パーツ分解 → ②パーツ単位のパネル分割+平坦化+SVG)。
_PATTERN_MIN_PANELS = 2   # パーツあたりの最大パネル数の下限
_PATTERN_MAX_PANELS = 6   # パーツあたりの最大パネル数の上限
_PATTERN_DEFAULT_PANELS = 4
_PATTERN_PARTS_MIN = 2
_PATTERN_PARTS_MAX = 10
_PATTERN_MIN_SEAM_ALLOWANCE_MM = 1
_PATTERN_MAX_SEAM_ALLOWANCE_MM = 30
_PATTERN_DEFAULT_SEAM_ALLOWANCE_MM = 7.0
_PATTERN_ARAP_ITERATIONS = 10
_PATTERN_DEFAULT_SMOOTH_ITERATIONS = 10
# 取付口(パーツ切断面のリム)との照合許容距離(mm)。リム頂点とパネル境界
# 頂点は同一頂点由来のため実質0だが、浮動小数点誤差を見込む。
_PATTERN_ATTACHMENT_TOLERANCE_MM = 0.01
# 手動シード誘導 (guidance="manual") のシード数の範囲。
# 同じ名前のシードは同一パーツに統合される(正面+背面に打つ等)ため、
# シード総数はパーツ数上限より多く取れる。ユニーク名(=パーツ数)は
# プレビューパレットが20色のため2〜20に制限する。
_PATTERN_SEEDS_MIN = 2
_PATTERN_SEEDS_MAX = 48
_PATTERN_SEED_PARTS_MIN = 2
_PATTERN_SEED_PARTS_MAX = 20


def _is_finite_number(v) -> bool:
    import math

    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _pattern_vertex_colors(mesh):
    import numpy as np
    import trimesh

    visual = getattr(mesh, "visual", None)
    if isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind == "vertex":
        return np.asarray(visual.vertex_colors)
    return None


def _pattern_load_prepared_mesh(job, smooth_iterations: int):
    """ジョブのGLBを読み込み、型紙用前処理を適用したメッシュを返す。"""
    import trimesh

    from .pattern import prepare_mesh

    model_path = job.model_path("glb")
    loaded = trimesh.load(model_path, file_type="glb", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    else:
        mesh = loaded
    return prepare_mesh(mesh, smooth_iterations=smooth_iterations)


def _pattern_load_image_rgba(job):
    """ジョブの背景除去済み入力画像(input.png)をRGBA ndarrayとして読み込む。

    存在しない場合は None を返す(画像誘導なしにフォールバック)。
    pattern モジュールはPILをimportしない境界規約のため、画像のロード・
    ndarray化はこのアダプタ側で行う。
    """
    import numpy as np
    from PIL import Image

    path = job.input_image_path()
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGBA")
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        logger.exception("Failed to load input image for pattern image-guided decomposition (job %s)", job.job_id)
        return None


def _pattern_llm_guided_labels(job, image_rgba):
    """LLMパーツ検出を試み、成功すれば `(image_labels, part_names, llm_parts)` を返す。

    失敗(endpoint未設定・通信エラー・不正応答等)した場合は
    `(None, None, None)` を返し、呼び出し側は色領域誘導へフォールバックする。
    """
    from . import llm_parts as llm_parts_module
    from .pattern.parts import labels_from_bboxes

    if not llm_parts_module.is_available(config.IMAGE3D_LLM_ENDPOINT):
        return None, None, None

    detected = llm_parts_module.detect_parts(
        image_rgba, config.IMAGE3D_LLM_ENDPOINT, timeout=config.IMAGE3D_LLM_TIMEOUT
    )
    if not detected:
        logger.info("LLM part detection unavailable/failed for job %s; falling back", job.job_id)
        return None, None, None

    bboxes = [tuple(p["bbox"]) for p in detected]
    part_names = {i: p["name"] for i, p in enumerate(detected)}
    alpha_mask = image_rgba[:, :, 3]
    image_labels = labels_from_bboxes(bboxes, alpha_mask)
    return image_labels, part_names, detected


def _pattern_run_parts(
    job,
    prepared,
    n_parts: int,
    smooth_iterations: int,
    save: bool = True,
    use_image: bool = True,
    use_llm: bool = True,
    seeds: Optional[list[dict]] = None,
    propagate_back: bool = True,
    planar_boundaries: bool = True,
):
    """パーツ分解を実行し、成果物(json/labels/preview GLB)を保存して返す。

    誘導の優先順位(SPEC.md §3.12 第3層誘導。手動シードが最優先):
        0. `seeds` 指定あり → 手動シード誘導 (`guidance="manual"`)。
           ユーザーが3Dビューア上でクリックした点から測地距離ベースで
           パーツを分割する。同じ名前のシードは同一パーツに統合される
           (マルチソース。正面+背面に打つと背面の割当が改善する)。
           `propagate_back=True`(既定)の場合はさらに、各シードの反対側
           スキンに同グループの仮想シードを自動生成し、正面のみのシードでも
           背面の割当を改善する(誤配置ガード付き、`labels_from_seeds` 参照)。
           `use_image`/`use_llm` は無視する。
        1. `use_llm=True` かつ `IMAGE3D_LLM_ENDPOINT` 設定あり かつ検出成功
           → LLMのbbox誘導 (`guidance="llm"`、パーツに `name` が付く)
        2. 上記が使えず `use_image=True` かつ入力画像あり
           → 色領域誘導 (`guidance="color"`)
        3. どちらも使えない → ジオメトリのみ (`guidance="geometry"`)

    Args:
        seeds: `[{"x": float, "y": float, "z": float, "name": str}, ...]`
            (任意)。座標はビューア表示中モデルと同じローカル座標系(mm)。
            `create_job_pattern_parts` でバリデーション済みのものを渡す想定。

    Returns:
        (labels, parts_meta, result_dict)
    """
    import json as json_module

    import numpy as np

    from .pattern import build_preview_mesh, decompose_parts
    from .pattern.parts import _IMG_SIGNIFICANT_CHUNK_FRACTION_LLM, extract_image_regions

    if seeds:
        seed_points = np.array([[float(s["x"]), float(s["y"]), float(s["z"])] for s in seeds])
        seed_names = [s.get("name") or None for s in seeds]
        manual_info: dict = {}
        labels, parts_meta = decompose_parts(
            prepared,
            seed_points=seed_points,
            seed_names=seed_names,
            seed_propagate_opposite=propagate_back,
            seed_planar_boundaries=planar_boundaries,
            manual_info_out=manual_info,
        )
        virtual_seeds = manual_info.get("virtual_seeds", [])
        result = {
            "job_id": job.job_id,
            # 同名シードは統合されるため、要求パーツ数=ユニーク名数
            "n_parts_requested": len({s["name"] for s in seeds}),
            "n_parts_actual": int(len(parts_meta)),
            "smooth_iterations": smooth_iterations,
            "guidance": "manual",
            "use_image": False,
            "use_llm": False,
            "n_image_regions": 0,
            "n_llm_parts_detected": 0,
            "seeds": seeds,
            "propagate_back": bool(propagate_back),
            "n_virtual_seeds": int(len(virtual_seeds)),
            "virtual_seeds": virtual_seeds,
            # 仮想シードを得られなかったグループの代表的な棄却理由
            # (no_hit/tunnel/taken/ownership/balance。トライアルの診断用)
            "virtual_seed_skips": manual_info.get("virtual_seed_skips", []),
            "planar_boundaries": bool(planar_boundaries),
            "planar_fit": manual_info.get("planar_fit", []),
            "parts": parts_meta,
        }
        if save:
            np.save(job.pattern_parts_labels_path(), labels)
            job.pattern_parts_json_path().write_text(
                json_module.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            preview_mesh = build_preview_mesh(prepared, labels)
            job.pattern_parts_preview_glb_path().write_bytes(preview_mesh.export(file_type="glb"))
        return labels, parts_meta, result

    image_rgba = _pattern_load_image_rgba(job) if (use_image or use_llm) else None

    guidance = "geometry"
    n_image_regions = 0
    image_labels = None
    part_names = None
    significant_chunk_fraction = _IMG_SIGNIFICANT_CHUNK_FRACTION_LLM  # 使われるのはllm時のみ
    llm_parts_detected = None

    if use_llm and image_rgba is not None:
        try:
            image_labels, part_names, llm_parts_detected = _pattern_llm_guided_labels(job, image_rgba)
        except Exception:
            logger.exception("LLM part detection raised for job %s; falling back", job.job_id)
            image_labels, part_names, llm_parts_detected = None, None, None
        if image_labels is not None:
            guidance = "llm"

    color_image_rgba = None
    if guidance != "llm" and use_image and image_rgba is not None:
        color_image_rgba = image_rgba
        try:
            _region_label_img, n_image_regions = extract_image_regions(color_image_rgba, seed=0)
            if n_image_regions > 0:
                guidance = "color"
            else:
                color_image_rgba = None
        except Exception:
            logger.exception("extract_image_regions failed for job %s; falling back to geometry only", job.job_id)
            color_image_rgba = None
            n_image_regions = 0

    if guidance == "llm":
        labels, parts_meta = decompose_parts(
            prepared,
            n_parts_hint=n_parts,
            seed=0,
            image_labels=image_labels,
            significant_chunk_fraction=significant_chunk_fraction,
            part_names=part_names,
        )
    elif guidance == "color":
        labels, parts_meta = decompose_parts(
            prepared, n_parts_hint=n_parts, seed=0, image_rgba=color_image_rgba
        )
    else:
        labels, parts_meta = decompose_parts(prepared, n_parts_hint=n_parts, seed=0)

    result = {
        "job_id": job.job_id,
        "n_parts_requested": n_parts,
        "n_parts_actual": int(len(parts_meta)),
        "smooth_iterations": smooth_iterations,
        "guidance": guidance,
        "use_image": bool(guidance == "color"),
        "use_llm": bool(guidance == "llm"),
        "n_image_regions": int(n_image_regions),
        "n_llm_parts_detected": int(len(llm_parts_detected)) if llm_parts_detected else 0,
        "parts": parts_meta,
    }

    if save:
        np.save(job.pattern_parts_labels_path(), labels)
        job.pattern_parts_json_path().write_text(
            json_module.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        preview_mesh = build_preview_mesh(prepared, labels)
        job.pattern_parts_preview_glb_path().write_bytes(preview_mesh.export(file_type="glb"))

    return labels, parts_meta, result


@app.post("/api/jobs/{job_id}/pattern/parts")
async def create_job_pattern_parts(job_id: str, body: Optional[dict] = None):
    """パーツ自動分解 (SPEC.md §3.12「2段階構成 (4c)」の1段目)。

    body: `{"n_parts": 0(自動) | 2〜10, "use_image": true(既定), "use_llm": true(既定),
    "seeds": [{"x": float, "y": float, "z": float, "name": str}, ...],
    "propagate_back": true(既定), "planar_boundaries": true(既定)}`。
    `planar_boundaries=true` の場合、隣接パーツペアの境界へ平面をフィットし、
    曖昧帯(背面・肩の上など誘導信号の弱い区間)を平面で再割当して境界を
    平面的な楕円に整える(縫いやすい取付口。ペアごとの適用/スキップと理由は
    結果の `planar_fit` に記録)。
    誘導の優先順位:
        0. `seeds` が指定された場合、最優先で手動シード誘導を使う
           (`guidance="manual"`)。ユーザーが3Dビューア上でクリックした
           2〜48点から測地距離ベースでパーツを分割する。**同じ名前の
           シードは同一パーツに統合される**(マルチソース。「胴体」を正面と
           背面に1点ずつ打つと背面の割当が改善する)。ユニークな名前の数
           (=パーツ数)は2〜20であること。`propagate_back=true`(既定)の
           場合、各シードの反対側スキン(逆法線レイの first exit)に同グループの
           仮想シードを自動生成し、正面のみのシードでも背面の割当を改善する
           (誤配置ガード付き。結果の `n_virtual_seeds`/`virtual_seeds` に記録)。
           `n_parts`/`use_image`/`use_llm` は無視される。
        1. `use_llm=true` かつ `IMAGE3D_LLM_ENDPOINT` 設定あり かつ検出成功
           → マルチモーダルLLMが検出した部位のbboxでパーツ分解を誘導する
           (`guidance="llm"`。パーツ名が付き、`部位N: <名前>` として表示できる)。
        2. 上記が使えない場合、`use_image=true` かつジョブの背景除去済み
           入力画像(input.png)が存在すれば、その画像の色領域を投影して
           パーツ分解を誘導する(`guidance="color"`。くびれの浅い頭・胴等の
           分離補助。3D形状だけでは不十分な場合の対策)。
        3. どちらも使えない場合はジオメトリのみ(`guidance="geometry"`)。

    `seeds` の座標系は、ビューアに表示中のモデル(型紙前処理後のprepared
    mesh)のローカル座標(Z-up, mm)。`prepare_mesh` は平行移動・スケールを
    行わないため、ビューア表示用GLBの座標をそのまま渡してよい。
    同期実行(十数秒、LLM使用時は+数秒〜十数秒)。`pattern_parts.json`
    (パーツ統計。`guidance`/`use_image`/`use_llm`/`n_image_regions`/
    `n_llm_parts_detected`、手動シード時は`seeds`を含む)と
    `pattern_parts_preview.glb`(パーツ色分け)をジョブディレクトリへ保存する。
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    if job.status != STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail=f"ジョブは未完了です(status={job.status})。")

    body = body or {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="リクエストボディはJSONオブジェクトである必要があります。")

    n_parts = body.get("n_parts", 0)
    if isinstance(n_parts, bool) or not isinstance(n_parts, int) or not (
        n_parts == 0 or _PATTERN_PARTS_MIN <= n_parts <= _PATTERN_PARTS_MAX
    ):
        raise HTTPException(
            status_code=400,
            detail=f"n_partsは0(自動)または{_PATTERN_PARTS_MIN}〜{_PATTERN_PARTS_MAX}の整数である必要があります。",
        )

    use_image = body.get("use_image", True)
    if not isinstance(use_image, bool):
        raise HTTPException(status_code=400, detail="use_imageはbool値である必要があります。")

    use_llm = body.get("use_llm", True)
    if not isinstance(use_llm, bool):
        raise HTTPException(status_code=400, detail="use_llmはbool値である必要があります。")

    seeds_raw = body.get("seeds")
    seeds: Optional[list[dict]] = None
    if seeds_raw is not None:
        if not isinstance(seeds_raw, list) or not (
            _PATTERN_SEEDS_MIN <= len(seeds_raw) <= _PATTERN_SEEDS_MAX
        ):
            raise HTTPException(
                status_code=400,
                detail=f"seedsは{_PATTERN_SEEDS_MIN}〜{_PATTERN_SEEDS_MAX}個のリストである必要があります。",
            )
        seeds = []
        for i, item in enumerate(seeds_raw):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail=f"seeds[{i}]はオブジェクトである必要があります。")
            coords = {}
            for axis in ("x", "y", "z"):
                v = item.get(axis)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not _is_finite_number(v):
                    raise HTTPException(
                        status_code=400,
                        detail=f"seeds[{i}].{axis}は有限の数値である必要があります。",
                    )
                coords[axis] = float(v)
            name = item.get("name")
            if name is not None and not isinstance(name, str):
                raise HTTPException(status_code=400, detail=f"seeds[{i}].nameは文字列である必要があります。")
            name = name.strip() if isinstance(name, str) else None
            seeds.append({**coords, "name": name or f"part_{i + 1}"})

        # 同じ名前のシードは同一パーツに統合されるため、パーツ数=ユニーク名数。
        # プレビューパレット(20色)に収まるようパーツ数を2〜20に制限する。
        n_unique_names = len({s["name"] for s in seeds})
        if not (_PATTERN_SEED_PARTS_MIN <= n_unique_names <= _PATTERN_SEED_PARTS_MAX):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"seedsのユニークな名前(=パーツ数)は{_PATTERN_SEED_PARTS_MIN}〜"
                    f"{_PATTERN_SEED_PARTS_MAX}である必要があります(同じ名前のシードは同一パーツに統合されます)。"
                ),
            )

    propagate_back = body.get("propagate_back", True)
    if not isinstance(propagate_back, bool):
        raise HTTPException(status_code=400, detail="propagate_backはbool値である必要があります。")

    planar_boundaries = body.get("planar_boundaries", True)
    if not isinstance(planar_boundaries, bool):
        raise HTTPException(status_code=400, detail="planar_boundariesはbool値である必要があります。")

    model_path = job.model_path("glb")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="モデルファイルが見つかりません。")

    def _run() -> dict:
        prepared = _pattern_load_prepared_mesh(job, _PATTERN_DEFAULT_SMOOTH_ITERATIONS)
        _labels, _meta, result = _pattern_run_parts(
            job,
            prepared,
            n_parts,
            _PATTERN_DEFAULT_SMOOTH_ITERATIONS,
            use_image=use_image,
            use_llm=use_llm,
            seeds=seeds,
            propagate_back=propagate_back,
            planar_boundaries=planar_boundaries,
        )
        return result

    import asyncio

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _run)
    except ValueError as exc:
        # labels_from_seeds: シード重複等のユーザー起因エラーは400として返す
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pattern parts decomposition failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"パーツ分解に失敗しました: {exc}") from exc

    return result


@app.post("/api/jobs/{job_id}/pattern")
async def create_job_pattern(job_id: str, body: Optional[dict] = None):
    """型紙生成 (SPEC.md §3.12「2段階構成 (4c)」の2段目)。

    事前に `POST .../pattern/parts` が実行済みならそのパーツ分解を再利用し、
    未実行なら内部で自動実行(n_parts=auto)する。パーツごとにパネル分割→
    平坦化→SVG出力を行う。`n_panels` は「パーツあたりの最大パネル数」
    (2〜6、デフォルト4)。取付口が多いパーツでは円盤位相の確保を優先して
    実パネル数がこれを超えることがある。
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    if job.status != STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail=f"ジョブは未完了です(status={job.status})。")

    body = body or {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="リクエストボディはJSONオブジェクトである必要があります。")

    n_panels = body.get("n_panels", _PATTERN_DEFAULT_PANELS)
    use_colors = body.get("use_colors", True)
    smooth_iterations = body.get("smooth_iterations", _PATTERN_DEFAULT_SMOOTH_ITERATIONS)
    seam_allowance_mm = body.get("seam_allowance_mm", _PATTERN_DEFAULT_SEAM_ALLOWANCE_MM)

    if isinstance(n_panels, bool) or not isinstance(n_panels, int) or not (
        _PATTERN_MIN_PANELS <= n_panels <= _PATTERN_MAX_PANELS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"n_panelsは{_PATTERN_MIN_PANELS}〜{_PATTERN_MAX_PANELS}の整数である必要があります。",
        )
    if not isinstance(use_colors, bool):
        raise HTTPException(status_code=400, detail="use_colorsはbool値である必要があります。")
    if isinstance(smooth_iterations, bool) or not isinstance(smooth_iterations, int) or not (
        0 <= smooth_iterations <= 50
    ):
        raise HTTPException(status_code=400, detail="smooth_iterationsは0〜50の整数である必要があります。")
    if isinstance(seam_allowance_mm, bool) or not isinstance(seam_allowance_mm, (int, float)) or not (
        _PATTERN_MIN_SEAM_ALLOWANCE_MM <= seam_allowance_mm <= _PATTERN_MAX_SEAM_ALLOWANCE_MM
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"seam_allowance_mmは{_PATTERN_MIN_SEAM_ALLOWANCE_MM}〜"
                f"{_PATTERN_MAX_SEAM_ALLOWANCE_MM}の数値である必要があります。"
            ),
        )
    seam_allowance_mm = float(seam_allowance_mm)

    model_path = job.model_path("glb")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="モデルファイルが見つかりません。")

    def _run_pattern() -> dict:
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree

        from .pattern import (
            build_pattern_svg,
            build_preview_mesh,
            cap_part,
            flatten_panel,
            panel_stats,
            segment_part_panels,
        )
        from .pattern.preview import PALETTE_HEX

        prepared = _pattern_load_prepared_mesh(job, smooth_iterations)
        prepared_colors = _pattern_vertex_colors(prepared)

        # --- ステージ1: パーツ分解(実行済みならそれを再利用) ---------------
        part_labels = None
        parts_meta = None
        labels_path = job.pattern_parts_labels_path()
        parts_json_path = job.pattern_parts_json_path()
        if labels_path.exists() and parts_json_path.exists():
            try:
                saved = json.loads(parts_json_path.read_text(encoding="utf-8"))
                # 平滑化条件が違うとメッシュ(面数・形状)が変わりラベルが
                # 合わないため、一致する場合のみ再利用する。
                if saved.get("smooth_iterations") == smooth_iterations:
                    candidate = np.load(labels_path)
                    if len(candidate) == len(prepared.faces):
                        part_labels = candidate
                        parts_meta = saved.get("parts")
            except Exception:
                part_labels = None
                parts_meta = None

        if part_labels is None or parts_meta is None:
            part_labels, parts_meta, _ = _pattern_run_parts(
                job, prepared, 0, smooth_iterations
            )

        # --- ステージ2: パーツごとにパネル分割→平坦化→SVG -------------------
        panels_2d: list[dict] = []
        parts_result: list[dict] = []
        preview_segments: list[tuple] = []  # (sub_mesh, sub_panel_labels) プレビュー用
        global_panel_id = 0

        for part_meta in parts_meta:
            part_id = int(part_meta["part_id"])
            part_name = part_meta.get("name")
            part_label = f"部位{part_id + 1}: {part_name}" if part_name else f"部位{part_id + 1}"
            part_color = PALETTE_HEX[part_id % len(PALETTE_HEX)]
            face_idx = np.where(part_labels == part_id)[0]

            part_entry = {
                "part_id": part_id,
                "part_label": part_label,
                "color_hex": part_color,
                "stats": part_meta,
                "n_panels": 0,
                "n_attachment_openings": 0,
                "panels": [],
            }
            parts_result.append(part_entry)
            if len(face_idx) == 0:
                continue

            closed_mesh, cap_info = cap_part(prepared, face_idx)
            part_entry["n_attachment_openings"] = int(cap_info["n_boundary_loops"])

            # 頂点カラーの引き継ぎ(前処理済みメッシュ→蓋済みパーツメッシュ)
            closed_colors = None
            if use_colors and prepared_colors is not None and len(closed_mesh.vertices) > 0:
                try:
                    tree = cKDTree(prepared.vertices)
                    _, nn_idx = tree.query(closed_mesh.vertices, k=1)
                    closed_colors = prepared_colors[nn_idx]
                except Exception:
                    closed_colors = None

            sub_mesh, sub_panel_labels, rim_coords = segment_part_panels(
                closed_mesh,
                cap_info["cap_face_mask"],
                max_panels=n_panels,
                vertex_colors=closed_colors,
                use_colors=use_colors,
                seed=0,
            )
            if len(sub_mesh.faces) == 0:
                continue
            preview_segments.append((sub_mesh, sub_panel_labels))

            sub_stats = panel_stats(sub_mesh, sub_panel_labels)
            rim_tree = cKDTree(rim_coords) if len(rim_coords) > 0 else None

            for panel_no, sub_stat in enumerate(sub_stats, start=1):
                local_panel_id = sub_stat["panel_id"]
                panel_face_idx = np.where(sub_panel_labels == local_panel_id)[0]
                flat_result = flatten_panel(
                    sub_mesh, panel_face_idx, n_arap_iterations=_PATTERN_ARAP_ITERATIONS
                )
                flat_result["panel_id"] = global_panel_id
                flat_result["panel_no"] = panel_no
                flat_result["part_id"] = part_id
                flat_result["part_label"] = part_label
                flat_result["part_color_hex"] = part_color

                panel_json = {
                    "panel_id": global_panel_id,
                    "panel_no": panel_no,
                    "n_faces": sub_stat["n_faces"],
                    "area_mm2": sub_stat["area_mm2"],
                    "boundary_loops": sub_stat["boundary_loops"],
                    "disk_topology": sub_stat["disk_topology"],
                    "flatten_failed": bool(flat_result.get("flatten_failed")),
                    "has_attachment_opening": False,
                }

                if flat_result.get("flatten_failed"):
                    panel_json["flatten_failed_reason"] = flat_result.get("reason", "")
                else:
                    panel_json["distortion"] = flat_result["distortion"]
                    # 取付口の判定: パネル境界点のうち、パーツ切断面のリムに
                    # 一致する区間をマークする(SVGで太線+ラベル表示)。
                    if rim_tree is not None:
                        loop3d = flat_result["vertices_3d"][flat_result["boundary_loop_indices"]]
                        dists, _ = rim_tree.query(loop3d, k=1)
                        attachment_mask = dists <= _PATTERN_ATTACHMENT_TOLERANCE_MM
                        flat_result["attachment_mask"] = attachment_mask
                        panel_json["has_attachment_opening"] = bool(np.any(attachment_mask))

                panels_2d.append(flat_result)
                part_entry["panels"].append(panel_json)
                global_panel_id += 1

            part_entry["n_panels"] = len(part_entry["panels"])

        # --- パネル色分けプレビューGLB(パーツごとの部分メッシュを結合) ------
        preview_meshes = []
        offset = 0
        for sub_mesh, sub_panel_labels in preview_segments:
            # グローバルパネルIDで色分け(SVG・statsと同じ色順序になる)
            preview_meshes.append(build_preview_mesh(sub_mesh, sub_panel_labels + offset))
            offset += len(np.unique(sub_panel_labels))

        if preview_meshes:
            combined = trimesh.util.concatenate(preview_meshes)
            job.pattern_preview_glb_path().write_bytes(combined.export(file_type="glb"))

        model_height_mm = float(job.stats.get("bbox_mm", [0, 0, 0])[2] or job.params.get("target_height_mm", 0) or 0)
        model_name = job.original_filename or job_id

        svg_text = build_pattern_svg(
            panels_2d,
            seam_allowance_mm=seam_allowance_mm,
            model_name=model_name,
            model_height_mm=model_height_mm,
        )
        job.pattern_svg_path().write_text(svg_text, encoding="utf-8")

        n_panels_total = len(panels_2d)
        n_flatten_ok = sum(1 for p in panels_2d if not p.get("flatten_failed"))

        result = {
            "job_id": job_id,
            "n_parts": len(parts_result),
            "n_panels_max_per_part": n_panels,
            "n_panels_total": n_panels_total,
            "n_panels_flattened": n_flatten_ok,
            "use_colors": use_colors,
            "smooth_iterations": smooth_iterations,
            "seam_allowance_mm": seam_allowance_mm,
            "parts": parts_result,
        }
        job.pattern_json_path().write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    import asyncio

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _run_pattern)
    except Exception as exc:
        logger.exception("Pattern generation failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"型紙生成に失敗しました: {exc}") from exc

    return result


@app.get("/api/jobs/{job_id}/pattern.json")
async def get_job_pattern_json(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.pattern_json_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="型紙がまだ生成されていません。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}/pattern_preview.glb")
async def get_job_pattern_preview_glb(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.pattern_preview_glb_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="型紙プレビューがまだ生成されていません。")
    return FileResponse(
        path, media_type="model/gltf-binary", filename=f"{job_id}_pattern_preview.glb"
    )


@app.get("/api/jobs/{job_id}/pattern.svg")
async def get_job_pattern_svg(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.pattern_svg_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="型紙SVGがまだ生成されていません。")
    return FileResponse(path, media_type="image/svg+xml", filename=f"{job_id}_pattern.svg")


@app.get("/api/jobs/{job_id}/pattern_parts.json")
async def get_job_pattern_parts_json(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.pattern_parts_json_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="パーツ分解がまだ実行されていません。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}/pattern_parts_preview.glb")
async def get_job_pattern_parts_preview_glb(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    path = job.pattern_parts_preview_glb_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="パーツ分解プレビューがまだ生成されていません。")
    return FileResponse(
        path, media_type="model/gltf-binary", filename=f"{job_id}_pattern_parts_preview.glb"
    )


@app.post("/api/sheet/split")
async def split_sheet(image: UploadFile = File(...)):
    """キャラクターシート画像から被写体パネルを自動検出する (SPEC.md §3.8 / FR-9)。

    ジョブは作成しない同期API。数秒で結果を返す。
    """
    data = await _read_and_validate_upload(image, "image")

    from .preprocess import load_and_validate_image

    pil_image = load_and_validate_image(data, config.MAX_UPLOAD_BYTES)

    import asyncio

    loop = asyncio.get_running_loop()
    panels = await loop.run_in_executor(None, sheet.split_sheet, pil_image)
    views = sheet.suggested_views(len(panels))

    result = []
    for idx, (panel, suggested_view) in enumerate(zip(panels, views)):
        buf = io.BytesIO()
        panel.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        result.append(
            {
                "index": idx,
                "image_b64": image_b64,
                "suggested_view": suggested_view,
            }
        )

    return {"panels": result}


@app.get("/api/health")
async def health():
    gpu_info = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "vram_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                ),
            }
    except ImportError:
        pass

    from . import texture
    from . import llm_parts

    return {
        "status": "ok",
        "generator": job_manager.generator.name,
        "python_version": platform.python_version(),
        "gpu": gpu_info,
        "texgen_available": texture.is_available(),
        "llm_parts_available": llm_parts.is_available(config.IMAGE3D_LLM_ENDPOINT),
    }


# --- 静的フロントエンド配信 (SPEC.md §5 `GET /`) -----------------------------
app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
