"""Hunyuan3D-2 (hy3dgen) shape pipeline ラッパ (Phase 2実装)。

DEVELOPMENT_POLICY.md / IMPLEMENTATION_PLAN.md Phase 2 の本番ジェネレータ。
`hy3dgen` は third_party/Hunyuan3D-2 を `pip install -e --no-deps` でこのvenvに
導入し(requirements-gpu.txt参照)、shapeパイプラインのみを使用する
(texgen・カスタムラスタライザのビルドは不要・行わない)。

実装方針:
  - `_load_pipeline()` で hy3dgen の shape pipeline をロードし、
    プロセスに常駐させる(NFR-3: 初回リクエスト時に1度だけロード)。
  - `generate()` で steps / guidance_scale / octree_resolution / seed を
    パイプラインに渡し、生成後に trimesh.Trimesh を返す。
  - 生成後に `torch.cuda.empty_cache()` を呼びVRAMを解放する(直列キューと合わせ
    GPUメモリを保護する。DEVELOPMENT_POLICY.md §6 リスク対策)。
  - 例外は意味のあるメッセージに変換して送出する(呼び出し側 jobs.py が
    status=failed + error に格納する)。
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Optional

import trimesh
from PIL import Image

from .. import config
from .base import GenerationParams, Generator

logger = logging.getLogger(__name__)

_IMPORT_ERROR_HINT = (
    "Hunyuan3D-2 の依存関係が見つかりません。requirements-gpu.txt を参照し、"
    "torch (cu128) と hy3dgen (Hunyuan3D-2 リポジトリ) を導入してください。"
    " 例: pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision"
    " && pip install -e third_party/Hunyuan3D-2 --no-deps"
)


class Hunyuan3DGenerator(Generator):
    """Hunyuan3D-2 shape pipeline を用いた本番用ジェネレータ。

    単一ビュー画像は `hunyuan3d-dit-v2-0`(tencent/Hunyuan3D-2)パイプラインで、
    複数ビュー画像(front + back/left/right の任意組合せ)は
    `hunyuan3d-dit-v2-mv`(tencent/Hunyuan3D-2mv)マルチビューパイプラインで
    生成する (SPEC.md §3.8 / FR-9)。両パイプラインは別インスタンスとして
    常駐させ、必要な方だけ遅延ロードする。
    """

    name = "hunyuan3d"

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None
        self._mv_pipeline: Optional[Any] = None
        self._lock = threading.Lock()
        self._mv_lock = threading.Lock()

    def _load_pipeline(self) -> Any:
        """初回呼び出し時にのみ単一ビュー用モデルをロードし、以降は常駐させる (NFR-3)。"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            # hy3dgen自体が参照するローカルキャッシュ探索先 (smart_load_model)。
            # config.py 経由で IMAGE3D_HY3DGEN_MODELS_DIR が指定されていれば反映する。
            if config.HY3DGEN_MODELS_DIR:
                os.environ.setdefault("HY3DGEN_MODELS", config.HY3DGEN_MODELS_DIR)

            try:
                import torch  # noqa: F401
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            except ImportError as exc:
                raise ImportError(_IMPORT_ERROR_HINT) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading Hunyuan3D-2 shape pipeline (%s, subfolder=%s) on %s; "
                "this may take a while on first run (downloads from HuggingFace)...",
                config.HY3DGEN_MODEL_PATH,
                config.HY3DGEN_SUBFOLDER,
                device,
            )
            try:
                pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    config.HY3DGEN_MODEL_PATH,
                    subfolder=config.HY3DGEN_SUBFOLDER,
                    device=device,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Hunyuan3D-2 パイプラインのロードに失敗しました "
                    f"(model_path={config.HY3DGEN_MODEL_PATH}, subfolder={config.HY3DGEN_SUBFOLDER}): {exc}"
                ) from exc

            if device == "cuda":
                # 注意: Hunyuan3DDiTPipeline.to() は self を返さず None を返す実装のため、
                # 戻り値を代入せずインプレースの副作用のみを利用する。
                pipeline.to(device)
            self._pipeline = pipeline
            logger.info("Hunyuan3D-2 shape pipeline loaded and resident.")
            return self._pipeline

    def _load_mv_pipeline(self) -> Any:
        """初回呼び出し時にのみマルチビュー用モデルをロードし、以降は常駐させる。

        単一ビュー用パイプラインとは別インスタンスとして共存常駐させる
        (SPEC.md §3.8)。モデル重み(数GB)は初回のみHuggingFaceからDLされる。
        """
        if self._mv_pipeline is not None:
            return self._mv_pipeline

        with self._mv_lock:
            if self._mv_pipeline is not None:
                return self._mv_pipeline

            if config.HY3DGEN_MODELS_DIR:
                os.environ.setdefault("HY3DGEN_MODELS", config.HY3DGEN_MODELS_DIR)

            try:
                import torch  # noqa: F401
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            except ImportError as exc:
                raise ImportError(_IMPORT_ERROR_HINT) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading Hunyuan3D-2 multiview shape pipeline (%s, subfolder=%s) on %s; "
                "this may take a while on first run (downloads from HuggingFace, several GB)...",
                config.HY3DGEN_MV_MODEL_PATH,
                config.HY3DGEN_MV_SUBFOLDER,
                device,
            )
            try:
                pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    config.HY3DGEN_MV_MODEL_PATH,
                    subfolder=config.HY3DGEN_MV_SUBFOLDER,
                    device=device,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Hunyuan3D-2 マルチビューパイプラインのロードに失敗しました "
                    f"(model_path={config.HY3DGEN_MV_MODEL_PATH}, "
                    f"subfolder={config.HY3DGEN_MV_SUBFOLDER}): {exc}"
                ) from exc

            if device == "cuda":
                pipeline.to(device)
            self._mv_pipeline = pipeline
            logger.info("Hunyuan3D-2 multiview shape pipeline loaded and resident.")
            return self._mv_pipeline

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        extra_views: Optional[dict[str, Image.Image]] = None,
    ) -> trimesh.Trimesh:
        use_mv = bool(extra_views)

        try:
            pipeline = self._load_mv_pipeline() if use_mv else self._load_pipeline()
        except (ImportError, RuntimeError):
            raise
        except Exception as exc:  # pragma: no cover - 想定外の初期化エラー
            raise RuntimeError(f"Hunyuan3D-2 パイプラインの初期化に失敗しました: {exc}") from exc

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(params.seed))

        if use_mv:
            # MVImageProcessorV2 は {view_tag: image} の辞書入力を期待する
            # (hy3dgen/shapegen/preprocessors.py 参照)。view_tagは
            # "front" / "left" / "back" / "right" のいずれか。
            image_input: Any = {"front": image, **extra_views}
        else:
            image_input = image

        # パイプラインはRGBA画像でも動作するが、背景除去前提の入力を想定し
        # そのまま渡す(prepare_image内で前処理される)。
        try:
            result = pipeline(
                image=image_input,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance_scale,
                octree_resolution=params.octree_resolution,
                generator=generator,
                output_type="trimesh",
            )
        except Exception as exc:
            kind = "マルチビュー" if use_mv else ""
            raise RuntimeError(f"Hunyuan3D-2{kind}での3Dメッシュ生成に失敗しました: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mesh = self._extract_mesh(result)
        if mesh is None:
            raise RuntimeError(
                "Hunyuan3D-2 がメッシュを生成できませんでした(出力が空でした)。"
                "入力画像や生成パラメータを見直してください。"
            )
        tm = self._to_trimesh(mesh)
        # hy3dgen の出力は Y-up 座標系。本アプリ (meshproc / target_height_mm) は
        # Z-up 前提のため、X軸まわり +90° 回転で Y-up → Z-up に変換する。
        # マルチビュー出力にも同様に適用する(座標系はshape/mvパイプライン間で共通)。
        tm.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        )
        return tm

    @staticmethod
    def _extract_mesh(result: Any) -> Any:
        """pipeline() の戻り値(単一Trimesh or ネストしたリスト)から先頭要素を取り出す。"""
        mesh = result
        # バッチ次元がリストで返る実装・単一メッシュで返る実装の両方に対応する。
        while isinstance(mesh, (list, tuple)):
            if len(mesh) == 0:
                return None
            mesh = mesh[0]
        return mesh

    @staticmethod
    def _to_trimesh(mesh: Any) -> trimesh.Trimesh:
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        # hy3dgenのLatent2MeshOutput等、vertices/faces属性を持つ独自型のフォールバック変換。
        vertices = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        if vertices is None or faces is None:
            vertices = getattr(mesh, "mesh_v", None)
            faces = getattr(mesh, "mesh_f", None)
        if vertices is None or faces is None:
            raise RuntimeError(
                f"Hunyuan3D-2 の出力をtrimesh.Trimeshに変換できませんでした(型: {type(mesh)!r})。"
            )
        return trimesh.Trimesh(vertices=vertices, faces=faces)
