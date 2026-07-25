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


async def _post_to_rig_service(filename: str, glb_bytes: bytes, params: Optional[str]) -> dict:
    """rig-service に GLB を送ってリグジョブを作る。

    ブラウザから直接叩かずサーバ経由にしているのは、数十MBのGLBを
    クライアントに往復させないためと、CORS設定を rig-service 側に強いないため。
    """
    import httpx

    files = {"model": (filename, glb_bytes, "model/gltf-binary")}
    data = {"params": params} if params else None
    async with httpx.AsyncClient(timeout=config.RIGSVC_TIMEOUT_SEC) as client:
        response = await client.post(
            f"{config.RIGSVC_URL}/api/rig", files=files, data=data
        )
    if response.status_code != 200:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise HTTPException(
            status_code=502,
            detail=f"リグサービスがエラーを返しました({response.status_code}): {detail}",
        )
    return response.json()


def _rig_params_with_image3d_conventions(params: Optional[str]) -> str:
    """rig-service へ渡す params に image-3d の座標系規約を明示する。

    rig-service は未知のGLBも受け取るため上方向・正面を推定できるが、推定は
    完璧ではない(実測: 正面判定は生成物77体に対して86〜91%)。**image-3d は
    自分の出力規約を確実に知っている**ので、推測させずに宣言する:

    - 上方向は Z(`server/generators/*` が全ジェネレータの出力をZ-upへ変換する)
    - 正面は -Y(同上。hunyuan3d は X軸+90°、pixal3d は X軸180°回転で揃える)

    呼び出し側が明示した値はそのまま尊重する(検証用に上書きできるように)。
    """
    data: dict = {}
    if params:
        try:
            data = json.loads(params)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"paramsのJSONが不正です: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=400, detail="paramsはJSONオブジェクトである必要があります。"
            )
    data.setdefault("up_axis", "z")
    data.setdefault("facing", "-y")
    return json.dumps(data)


@app.post("/api/jobs/{job_id}/rig")
async def send_job_to_rig_service(job_id: str, params: Optional[str] = Form(None)):
    """完了ジョブの model.glb を rig-service へ送る (別リポジトリ rig-service の計画書 §7 R4-1)。

    リグ結果は rig-service 側のジョブになるため、ここではそのジョブIDと
    プレビューURLだけを返す(image-3d はリグ結果を保持しない)。
    """
    if not config.RIGSVC_URL:
        raise HTTPException(
            status_code=503,
            detail="リグサービスのURLが未設定です(環境変数 IMAGE3D_RIGSVC_URL)。",
        )
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    if job.status != STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail=f"ジョブは未完了です(status={job.status})。")
    path = job.model_path("glb")
    if not path.exists():
        raise HTTPException(status_code=404, detail="モデルファイルが見つかりません。")

    import asyncio

    loop = asyncio.get_running_loop()
    glb_bytes = await loop.run_in_executor(None, path.read_bytes)
    # VRMのタイトルは rig-service 側でファイル名から決まるため、元の画像名を渡す
    filename = job.original_filename or f"{job_id}.glb"
    filename = f"{filename.rsplit('.', 1)[0]}.glb"

    try:
        result = await _post_to_rig_service(
            filename, glb_bytes, _rig_params_with_image3d_conventions(params)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to call rig service for job %s", job_id)
        raise HTTPException(
            status_code=502,
            detail=f"リグサービス({config.RIGSVC_URL})に接続できませんでした: {exc}",
        ) from exc

    rig_job_id = result.get("job_id")
    return {
        "rig_job_id": rig_job_id,
        "url": f"{config.RIGSVC_URL}/?job={rig_job_id}",
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    ok = job_manager.delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    return {"deleted": True}


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

    return {
        "status": "ok",
        "generator": job_manager.generator.name,
        "python_version": platform.python_version(),
        "gpu": gpu_info,
        "texgen_available": texture.is_available(),
        # 未設定(null)ならフロントは「リグ/VRM化」ボタンを出さない
        "rigsvc_url": config.RIGSVC_URL,
    }


# --- 静的フロントエンド配信 (SPEC.md §5 `GET /`) -----------------------------
class RevalidatingStaticFiles(StaticFiles):
    """静的ファイルに `Cache-Control: no-cache` を付けて必ず再検証させる。

    Starlette の StaticFiles は ETag / Last-Modified は返すが Cache-Control を
    付けない。するとブラウザは**ヒューリスティックキャッシュ**(最終更新からの
    経過時間の約10%)を使い、その間はサーバに問い合わせずに古いファイルを
    使い続ける。コードを直したのに画面が変わらない、という混乱の原因になる。

    `no-cache` は「保存してよいが使う前に必ず再検証せよ」の意味なので、
    毎回 ETag による条件付きGETが走り、変更が無ければ 304 で終わる
    (転送量はほとんど増えない)。
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


app.mount("/", RevalidatingStaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
