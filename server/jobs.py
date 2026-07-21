"""ジョブ管理: モデル定義・直列実行キュー・`data/jobs/`永続化 (SPEC.md §3.3, §3.7, §5)。

- ジョブは `data/jobs/<job_id>/` に `input.png`(前処理後入力画像)、
  `model.glb` / `model.stl` 等、`meta.json` を保存する。
- 実行は asyncio ループ + 単一スレッドワーカーで直列実行する(NFR-2)。
- status遷移: queued -> preprocessing -> generating -> postprocessing -> completed
              (どの段階でも failed に遷移しうる)
- サーバ再起動時は `data/jobs/` から履歴を再読込する。
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh
from PIL import Image

from . import colorproc, config, meshproc, preprocess, texture
from .generators.base import GenerationParams, Generator

logger = logging.getLogger(__name__)

STATUS_QUEUED = "queued"
STATUS_PREPROCESSING = "preprocessing"
STATUS_GENERATING = "generating"
STATUS_POSTPROCESSING = "postprocessing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

EXPORT_FORMATS = {"stl", "3mf", "obj", "glb"}

# SPEC.md §3.8 (FR-9): front以外に受付可能な追加ビューラベル
EXTRA_VIEW_LABELS = ("back", "left", "right")

# SPEC.md 7章: メッシュ出力は trimesh のfile_typeキーワードに合わせる
_TRIMESH_FILE_TYPE = {"stl": "stl", "3mf": "3mf", "obj": "obj", "glb": "glb"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mesh_vertex_colors(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    """メッシュが明示的な頂点カラー(ColorVisuals, kind='vertex')を持つ場合に返す。

    Pixal3D等のテクスチャ付き出力ジェネレータは、raw meshにテクスチャから
    サンプリングした頂点カラーを載せて返す (SPEC.md §3.3)。trimeshのデフォルト
    カラー(kind=None)は「色情報あり」とみなさない。
    """
    visual = getattr(mesh, "visual", None)
    if isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind == "vertex":
        return np.asarray(visual.vertex_colors)
    return None


@dataclass
class Job:
    job_id: str
    status: str = STATUS_QUEUED
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    error: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(
        default_factory=lambda: {
            "vertices": 0,
            "faces": 0,
            "watertight": False,
            "bbox_mm": [0, 0, 0],
            "volume_cm3": 0.0,
            "palette": [],
        }
    )
    generator: str = "mock"
    bg_removed: bool = False
    original_filename: Optional[str] = None
    views: list[str] = field(default_factory=lambda: ["front"])
    # SPEC.md §3.9 (FR-10): texture_mode=paint 失敗時、jobをfailedにせず警告を記録して
    # 正面/背面投影方式にフォールバックする(graceful degradation)。
    warnings: list[str] = field(default_factory=list)
    textured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dir_path(self) -> Path:
        return config.JOBS_DIR / self.job_id

    def input_image_path(self) -> Path:
        return self.dir_path() / "input.png"

    def original_image_path(self) -> Path:
        return self.dir_path() / "original.png"

    def extra_view_input_path(self, view: str) -> Path:
        return self.dir_path() / f"input_{view}.png"

    def extra_view_raw_path(self, view: str) -> Path:
        return self.dir_path() / f"_raw_upload_{view}"

    def model_path(self, fmt: str) -> Path:
        return self.dir_path() / f"model.{fmt}"

    def model_color_3mf_path(self) -> Path:
        """カラーモード時のマルチオブジェクト3MF(色ごとに分割済み)。"""
        return self.dir_path() / "model_color.3mf"

    def is_color_mode(self) -> bool:
        return self.params.get("color_mode", "none") == "color4"

    def meta_path(self) -> Path:
        return self.dir_path() / "meta.json"

    def save_meta(self) -> None:
        self.dir_path().mkdir(parents=True, exist_ok=True)
        self.meta_path().write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load_meta(cls, meta_path: Path) -> "Job":
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(**data)


class JobManager:
    """ジョブの生成・永続化・直列実行キューを管理する。"""

    def __init__(self, generator: Generator) -> None:
        self.generator = generator
        self.jobs: dict[str, Job] = {}
        self._queue: Optional["asyncio.Queue[str]"] = None
        self._worker_task: Optional[asyncio.Task] = None
        # texgen paint pipeline は shape pipeline とは別に常駐させる(遅延ロード)。
        # 生成器がmockの場合やtexgen未導入環境でも `_run_paint` 呼び出し自体は
        # 安全に失敗できるよう、生成時にのみインスタンス化する。
        self._texture_pipeline: Optional[texture.TexturePipelineWrapper] = None
        self._texture_pipeline_lock = threading.Lock()

    def _get_texture_pipeline(self) -> texture.TexturePipelineWrapper:
        if self._texture_pipeline is None:
            with self._texture_pipeline_lock:
                if self._texture_pipeline is None:
                    self._texture_pipeline = texture.TexturePipelineWrapper()
        return self._texture_pipeline

    def _run_paint(
        self, mesh: trimesh.Trimesh, image: Image.Image, job: "Job"
    ) -> Optional[trimesh.Trimesh]:
        """texture_mode=paint 時のペイント実行(同期・ワーカースレッドから呼ばれる)。

        失敗時は例外を送出せず None を返し、`job.warnings` に警告メッセージを
        記録する(graceful degradation: SPEC.md §3.9)。
        """
        try:
            pipeline = self._get_texture_pipeline()
            return pipeline.paint(mesh, image)
        except Exception as exc:
            logger.exception("texture_mode=paint failed for job %s; falling back", job.job_id)
            job.warnings.append(
                f"テクスチャ生成(paint)に失敗したため、正面/背面投影方式にフォールバックしました: {exc}"
            )
            return None

    # --- 永続化 -----------------------------------------------------------
    def load_history(self) -> None:
        """サーバ起動時に `data/jobs/` から既存ジョブを再読込する。"""
        config.ensure_dirs()
        if not config.JOBS_DIR.exists():
            return
        for job_dir in sorted(config.JOBS_DIR.iterdir()):
            meta_path = job_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                job = Job.load_meta(meta_path)
            except Exception:
                logger.exception("Failed to load job meta: %s", meta_path)
                continue
            # 再起動時に実行中だったジョブは failed 扱いにする(実行状態を失うため)
            if job.status in (STATUS_QUEUED, STATUS_PREPROCESSING, STATUS_GENERATING, STATUS_POSTPROCESSING):
                job.status = STATUS_FAILED
                job.error = "サーバ再起動のため中断されました。"
                job.updated_at = _now_iso()
                job.save_meta()
            self.jobs[job.job_id] = job

    async def start_worker(self) -> None:
        # asyncio.Queue はイベントループにバインドされるため、ワーカー起動時
        # (実行中のイベントループが確定するタイミング)に生成する。
        self._queue = asyncio.Queue()
        for job in self.jobs.values():
            if job.status == STATUS_QUEUED:
                await self._queue.put(job.job_id)
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        self._queue = None

    # --- ジョブ操作 ---------------------------------------------------------
    def list_jobs(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def create_job(
        self,
        image_bytes: bytes,
        params: GenerationParams,
        original_filename: Optional[str] = None,
        extra_images: Optional[dict[str, bytes]] = None,
    ) -> Job:
        """ジョブを作成する。

        Args:
            image_bytes: 正面(front)画像。必須。
            extra_images: 追加ビュー画像({"back": bytes, "left": bytes, ...})。
                SPEC.md §3.8 (FR-9)。指定時はジョブの `views` に
                ["front", ...指定されたビュー] を記録し、生成時に
                ジェネレータへ extra_views として渡す。
        """
        extra_images = extra_images or {}
        views = ["front"] + [v for v in EXTRA_VIEW_LABELS if v in extra_images]

        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            status=STATUS_QUEUED,
            params={
                "steps": params.steps,
                "guidance_scale": params.guidance_scale,
                "octree_resolution": params.octree_resolution,
                "seed": params.seed,
                "remove_bg": params.remove_bg,
                "target_height_mm": params.target_height_mm,
                "max_faces": params.max_faces,
                "color_mode": params.color_mode,
                "n_colors": params.n_colors,
                "texture_mode": params.texture_mode,
            },
            generator=self.generator.name,
            original_filename=original_filename,
            views=views,
        )
        job.dir_path().mkdir(parents=True, exist_ok=True)
        # アップロードされた生データを一時的に保存(前処理はワーカー内で実行)
        raw_path = job.dir_path() / "_raw_upload"
        raw_path.write_bytes(image_bytes)
        for view, data in extra_images.items():
            job.extra_view_raw_path(view).write_bytes(data)

        job.save_meta()
        self.jobs[job_id] = job
        await self._queue.put(job_id)
        return job

    def delete_job(self, job_id: str) -> bool:
        job = self.jobs.pop(job_id, None)
        if job is None:
            return False
        if job.dir_path().exists():
            shutil.rmtree(job.dir_path(), ignore_errors=True)
        return True

    # --- ワーカー -----------------------------------------------------------
    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._run_job(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unhandled error while running job %s", job_id)
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return

        loop = asyncio.get_running_loop()

        def set_status(status: str) -> None:
            job.status = status
            job.updated_at = _now_iso()
            job.save_meta()

        try:
            set_status(STATUS_PREPROCESSING)
            raw_path = job.dir_path() / "_raw_upload"
            raw_bytes = raw_path.read_bytes()

            params = GenerationParams(**job.params)

            original, processed, bg_removed = await loop.run_in_executor(
                None, preprocess.preprocess_image, raw_bytes, config.MAX_UPLOAD_BYTES, params.remove_bg
            )
            job.bg_removed = bg_removed
            processed.save(job.input_image_path())
            original.save(job.original_image_path())

            # 追加ビュー(back/left/right)の前処理。各ビューにも背景除去を適用する
            # (SPEC.md §3.8)。カラーモードではback画像があれば背面投影に利用する。
            extra_views: dict[str, Image.Image] = {}
            for view in EXTRA_VIEW_LABELS:
                if view not in job.views:
                    continue
                view_raw_path = job.extra_view_raw_path(view)
                if not view_raw_path.exists():
                    continue
                view_raw_bytes = view_raw_path.read_bytes()
                _, view_processed, _ = await loop.run_in_executor(
                    None,
                    preprocess.preprocess_image,
                    view_raw_bytes,
                    config.MAX_UPLOAD_BYTES,
                    params.remove_bg,
                )
                view_processed.save(job.extra_view_input_path(view))
                extra_views[view] = view_processed

            set_status(STATUS_GENERATING)
            raw_mesh: trimesh.Trimesh = await loop.run_in_executor(
                None, self.generator.generate, processed, params, extra_views or None
            )

            set_status(STATUS_POSTPROCESSING)
            processed_mesh, stats = await loop.run_in_executor(
                None, meshproc.process, raw_mesh, params.target_height_mm, params.max_faces
            )

            # ジェネレータ(pixal3d等)がraw meshにテクスチャ由来の頂点カラーを付与
            # している場合、meshprocの簡略化・再構築で失われるため、後処理後メッシュへ
            # 最近傍転写する(座標系差はbbox正規化で吸収)。GLBは頂点カラー付きで
            # 保存され、color_mode=color4 時はこのカラーから量子化・分割する。
            generator_vertex_colors: Optional[np.ndarray] = None
            raw_colors = _mesh_vertex_colors(raw_mesh)
            if raw_colors is not None:
                def _transfer_colors(
                    src_mesh=raw_mesh, src_colors=raw_colors, dst_mesh=processed_mesh
                ):
                    transferred = colorproc.transfer_vertex_colors_nearest(
                        src_mesh.vertices, src_colors, dst_mesh.vertices, align_bbox=True
                    )
                    dst_mesh.visual = trimesh.visual.ColorVisuals(
                        mesh=dst_mesh, vertex_colors=transferred
                    )
                    return transferred

                generator_vertex_colors = await loop.run_in_executor(None, _transfer_colors)

            # SPEC.md §3.9 (FR-10): texture_mode=paint 時、texgenで全周テクスチャを
            # 生成する。ビューア用GLBはテクスチャ付きメッシュを使う。失敗時はjobを
            # failedにせず警告を記録し、正面/背面投影方式にフォールバックする。
            textured_mesh: Optional[trimesh.Trimesh] = None
            if params.texture_mode == "paint":
                textured_mesh = await loop.run_in_executor(
                    None, self._run_paint, processed_mesh, processed, job
                )
                job.textured = textured_mesh is not None

            color_mode = params.color_mode == "color4"
            palette_stats_data: list[dict] = []
            color_submeshes: Optional[list[tuple[trimesh.Trimesh, str]]] = None

            if color_mode:
                def _apply_color(
                    mesh=processed_mesh,
                    image=processed,
                    back_image=extra_views.get("back"),
                    n_colors=params.n_colors,
                    tex_mesh=textured_mesh,
                    gen_colors=generator_vertex_colors,
                ):
                    if tex_mesh is not None:
                        # テクスチャ色をUV経由で各頂点にサンプリングし、従来の
                        # 量子化・分割ロジックに接続する(正面投影の代わり)。
                        vertex_colors = texture.sample_vertex_colors_from_texture(tex_mesh)
                    elif gen_colors is not None:
                        # ジェネレータ由来の頂点カラー(pixal3dのテクスチャサンプル
                        # を転写済み)から量子化・分割する(正面投影の代わり)。
                        vertex_colors = gen_colors
                    else:
                        vertex_colors = colorproc.project_multiview_colors(
                            mesh, image, back_image=back_image
                        )
                    palette, labels = colorproc.quantize(vertex_colors, n_colors)
                    stats_data = colorproc.palette_stats(labels, palette, mesh)
                    submeshes = colorproc.split_by_color(mesh, labels, palette)
                    # GLB用(非paint時): 元メッシュ全体にも量子化前(投影そのまま)の
                    # 頂点カラーを付与する。paint時のGLBはテクスチャ優先で頂点カラーは
                    # 付与しない(下のexportロジック参照)。gen_colors時は転写済み。
                    if tex_mesh is None and gen_colors is None:
                        mesh.visual = trimesh.visual.ColorVisuals(
                            mesh=mesh, vertex_colors=vertex_colors
                        )
                    return stats_data, submeshes

                palette_stats_data, color_submeshes = await loop.run_in_executor(None, _apply_color)

            def _export_all() -> None:
                for fmt in EXPORT_FORMATS:
                    if fmt == "glb" and textured_mesh is not None:
                        # ビューア用GLBはテクスチャ付きメッシュを出力する。
                        data = textured_mesh.export(file_type="glb")
                    elif fmt == "3mf" and color_mode and color_submeshes:
                        # カラーモード時の通常3mfは形状のみ(単色扱いで従来通り出力)
                        data = processed_mesh.export(file_type="3mf")
                    else:
                        data = processed_mesh.export(file_type=_TRIMESH_FILE_TYPE[fmt])
                    if isinstance(data, str):
                        data = data.encode("utf-8")
                    job.model_path(fmt).write_bytes(data)

                if color_mode and color_submeshes:
                    scene = trimesh.Scene()
                    for idx, (sub_mesh, hex_color) in enumerate(color_submeshes, start=1):
                        scene.add_geometry(
                            sub_mesh, node_name=f"color_{idx}", geom_name=f"color_{idx}"
                        )
                    color_data = scene.export(file_type="3mf")
                    if isinstance(color_data, str):
                        color_data = color_data.encode("utf-8")
                    job.model_color_3mf_path().write_bytes(color_data)

            await loop.run_in_executor(None, _export_all)

            stats_dict = stats.to_dict()
            stats_dict["palette"] = palette_stats_data
            job.stats = stats_dict
            # raw uploadはもう不要
            raw_path.unlink(missing_ok=True)
            for view in EXTRA_VIEW_LABELS:
                job.extra_view_raw_path(view).unlink(missing_ok=True)
            set_status(STATUS_COMPLETED)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            job.status = STATUS_FAILED
            job.error = str(exc)
            job.updated_at = _now_iso()
            job.save_meta()
