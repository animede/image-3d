"""Pixal3D (TencentARC/Pixal3D, TRELLIS.2基盤, MITライセンス) ジェネレータ。

単一画像からPBRテクスチャ付きGLBメッシュを生成する。hunyuan3d.py と同じ
プラガブルGenerator設計 (SPEC.md §3.3) に従う:
  - 遅延import・初回リクエスト時に1度だけロードし常駐 (NFR-3)
  - 生成後に torch.cuda.empty_cache() でVRAM解放
  - 例外は意味のあるメッセージに変換して送出

前提環境 (requirements-pixal3d.txt 参照):
  - 専用venv .venv-pixal3d (Python 3.10 / torch cu128 / o_voxel / drtk / natten)
  - third_party/Pixal3D のclone (pipインストールせず sys.path 経由でimport)
  - ATTN_BACKEND=sdpa / SPARSE_ATTN_BACKEND=sdpa (flash_attn不使用。
    launch.json の image3d-server-pixal3d が設定する)

実装上の注意 (実機検証で確認した非自明な事実):
  - Pixal3D公式の背景除去モデル briaai/RMBG-2.0 はHFゲート付きリポジトリのため
    ロードしない (`_load_pipeline` は upstream の from_pretrained を rembg_model
    抜きで再実装)。本アプリの前処理 (server/preprocess.py, rembg CPU) が生成した
    RGBA画像を渡すため、pipeline.preprocess_image() の rembg 経路は通らない。
    アルファチャンネルの無い画像が来た場合は意味のあるエラーを送出する。
  - カメラFOVの自動推定 (MoGe) は導入せず、固定FOV (config.PIXAL3D_FOV) を使う。
  - 座標系: o_voxel.postprocess.to_glb の出力は 上=-Z / 正面=+Y
    (momo.png での実測により確認: 頂点法線ベースの6方向投影レンダリングで
    顔が+Y側・頭頂が-Z側にあることを確認)。本アプリの Z-up / 正面=-Y へは
    X軸まわり180°回転で変換する。
  - テクスチャの活用: to_glb の出力 (UV + PBR baseColorTexture) から
    server/texture.py の sample_vertex_colors_from_texture で頂点カラーを
    サンプリングし、ColorVisuals としてメッシュに載せて返す。jobs.py 側が
    meshproc 後のメッシュへ最近傍転写する (colorproc.transfer_vertex_colors_nearest)。
  - ラスタライザ: o_voxel.postprocess.to_glb のUVテクスチャベイクは本来
    nvdiffrast (非商用限定ライセンス) 必須だが、本モジュールは
    pixal3d_raster.py (drtk, MIT) のシムに `o_voxel.postprocess.dr` を
    差し替えることでMITライセンスのみで完結させる (select_rasterizer_module /
    _inject_rasterizer、IMAGE3D_PIXAL3D_RASTERIZER で切替可能。既定 auto は
    drtkがあればdrtkを使う)。README.md「Pixal3Dジェネレータ」節参照。
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
from typing import Any, Optional

import numpy as np
import trimesh
from PIL import Image

from .. import config
from ..texture import sample_vertex_colors_from_texture
from .base import GenerationParams, Generator

logger = logging.getLogger(__name__)

_IMPORT_ERROR_HINT = (
    "Pixal3D の依存関係が見つかりません。requirements-pixal3d.txt を参照し、"
    "専用venv (.venv-pixal3d) に torch (cu128) / o_voxel / drtk / natten を"
    "導入し、third_party/Pixal3D をcloneしてください。"
    "サーバは .venv-pixal3d/bin/uvicorn で起動する必要があります"
    "(.claude/launch.json の image3d-server-pixal3d 参照)。"
)

# inference.py (third_party/Pixal3D) と同じ DinoV3 画像条件付けモデル構成。
# upstream の IMAGE_COND_CONFIGS を転記 (リポジトリ直下の inference.py は
# モジュール名が汎用的すぎて衝突リスクがあるためimportしない)。
_IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}


def _distance_from_fov(camera_angle_x: float, image_resolution: int = 512) -> float:
    """固定FOVからカメラ距離を計算する (upstream inference.py の distance_from_fov 相当)。

    grid_point (-1, 0, 0) が画像左端に写る距離を解析的に求める。
    upstream実装のBlender座標変換・NDC計算を purely-scalar に単純化した等価式。
    """
    # upstream: grid_point=(-1,0,0) → Blender回転 [[1,0,0],[0,0,-1],[0,1,0]] で
    # (x, -z, y) = (-1, 0, 0)。mesh_scale=1, /2 → (-0.5, 0, 0)。
    xw, yw = -0.5, 0.0
    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    f_pixels = focal_length * image_resolution / 32.0
    # 画像左端 (target x=0) の NDC x
    x_ndc = 0.0 - image_resolution / 2.0
    distance = f_pixels * xw / x_ndc - yw
    return float(distance)


def _has_meaningful_alpha(image: Image.Image) -> bool:
    """背景除去済み (=部分的に透明な) RGBA画像かどうか。"""
    if image.mode != "RGBA":
        return False
    alpha = np.asarray(image.getchannel("A"))
    return bool((alpha < 255).any())


def select_rasterizer_module(choice: Optional[str] = None) -> Any:
    """`IMAGE3D_PIXAL3D_RASTERIZER` に基づき、ラスタライザモジュールを選択する。

    `o_voxel.postprocess.to_glb` は `dr.RasterizeCudaContext` /
    `dr.rasterize` / `dr.interpolate` の3 APIのみを使用する (postprocess.py
    223〜266行付近)。これらをMIT (drtk) 実装のシムに差し替えることで、
    非商用限定の nvdiffrast への依存を既定構成から除去する (README.md
    「Pixal3Dジェネレータ」節参照)。

    Args:
        choice: "auto" | "drtk" | "nvdiffrast"。Noneの場合は
            `config.PIXAL3D_RASTERIZER` を使う。

    Returns:
        `dr.RasterizeCudaContext` / `dr.rasterize` / `dr.interpolate` 互換の
        呼び出し規約を持つモジュール (`pixal3d_raster` シム、または
        `nvdiffrast.torch` そのもの)。

    Raises:
        ImportError: choice="drtk" を明示指定したがdrtkが利用不可、または
            choice="nvdiffrast" を明示指定したがnvdiffrastが利用不可の場合。
        RuntimeError: choiceが不正な値の場合。
    """
    resolved = choice if choice is not None else config.PIXAL3D_RASTERIZER
    if resolved not in ("auto", "drtk", "nvdiffrast"):
        raise RuntimeError(
            f"IMAGE3D_PIXAL3D_RASTERIZER は 'auto' | 'drtk' | 'nvdiffrast' の"
            f"いずれかを指定してください (got: {resolved!r})。"
        )

    from . import pixal3d_raster

    if resolved == "drtk":
        if not pixal3d_raster.is_available():
            raise ImportError(
                "IMAGE3D_PIXAL3D_RASTERIZER=drtk が指定されましたが drtk が"
                "importできません。requirements-pixal3d.txt のdrtkビルド手順を"
                "参照してください。"
            )
        logger.info("Pixal3D: rasterizer backend = drtk (MIT, 明示指定)")
        return pixal3d_raster

    if resolved == "nvdiffrast":
        import nvdiffrast.torch as ndr  # noqa: F401 (import可否のみ確認)

        logger.info(
            "Pixal3D: rasterizer backend = nvdiffrast (NVIDIA Source Code "
            "License, 非商用限定, 明示指定)"
        )
        return ndr

    # auto: drtkがimport可能ならdrtkを優先し、そうでなければnvdiffrastへ
    # フォールバックする(ログでどちらが選ばれたかを明示する)。
    if pixal3d_raster.is_available():
        logger.info("Pixal3D: rasterizer backend = drtk (MIT, auto解決)")
        return pixal3d_raster

    logger.warning(
        "Pixal3D: drtk (MIT) が利用できないため nvdiffrast (NVIDIA Source Code "
        "License, 非商用限定) にフォールバックします。requirements-pixal3d.txt "
        "のdrtkビルド手順を参照し、可能であればdrtkの導入を推奨します。"
    )
    import nvdiffrast.torch as ndr

    return ndr


def _inject_rasterizer(o_voxel_module: Any) -> Any:
    """`o_voxel.postprocess.dr` を選択したラスタライザモジュールに差し替える。

    to_glb呼び出し前に呼ぶ (Pixal3DGenerator._to_textured_trimesh 参照)。
    差し替え後のモジュールも返す (呼び出し側でのログ・テスト用途)。
    """
    module = select_rasterizer_module()
    o_voxel_module.postprocess.dr = module
    return module


class Pixal3DGenerator(Generator):
    """Pixal3D image-to-3D パイプラインを用いたジェネレータ。

    出力はテクスチャからサンプリングした頂点カラー付き (ColorVisuals) の
    trimesh.Trimesh。マルチビュー入力 (extra_views) は非対応。
    """

    name = "pixal3d"

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None
        self._lock = threading.Lock()

    # --- パイプラインロード -------------------------------------------------
    def _load_pipeline(self) -> Any:
        """初回呼び出し時にのみモデルをロードし、以降常駐させる (NFR-3)。"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            repo_dir = str(config.PIXAL3D_REPO_DIR)
            if not os.path.isdir(os.path.join(repo_dir, "pixal3d")):
                raise RuntimeError(
                    f"Pixal3Dリポジトリが見つかりません ({repo_dir})。"
                    "`git clone https://github.com/TencentARC/Pixal3D third_party/Pixal3D` "
                    "を実行してください。"
                )
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)

            # pixal3d のattention backend はimport時に環境変数から確定するため、
            # import前に未設定ならSDPAをデフォルトにする (flash_attn非導入環境)。
            os.environ.setdefault("ATTN_BACKEND", "sdpa")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            try:
                import torch
                from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
            except ImportError as exc:
                raise ImportError(_IMPORT_ERROR_HINT) from exc

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Pixal3D はGPU (CUDA) 必須です。GPUが利用できない環境では "
                    "IMAGE3D_GENERATOR=mock を使用してください。"
                )

            logger.info(
                "Loading Pixal3D pipeline (%s, low_vram=%s, resolution=%d); "
                "this may take a while on first run (downloads ~23GB from HuggingFace)...",
                config.PIXAL3D_MODEL_PATH,
                config.PIXAL3D_LOW_VRAM,
                config.PIXAL3D_RESOLUTION,
            )
            try:
                pipeline = self._from_pretrained_without_rembg(
                    Pixal3DImageTo3DPipeline, config.PIXAL3D_MODEL_PATH
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Pixal3D パイプラインのロードに失敗しました "
                    f"(model_path={config.PIXAL3D_MODEL_PATH}): {exc}"
                ) from exc

            try:
                self._setup_image_cond_models(pipeline)
            except Exception as exc:
                raise RuntimeError(
                    f"Pixal3D 画像条件付けモデル (DinoV3/NAF) の初期化に失敗しました: {exc}"
                ) from exc

            if config.PIXAL3D_LOW_VRAM:
                pipeline._device = torch.device("cuda")
                pipeline.low_vram = True
            else:
                pipeline.low_vram = False
                pipeline.cuda()
                for attr in (
                    "image_cond_model_ss",
                    "image_cond_model_shape_512",
                    "image_cond_model_shape_1024",
                    "image_cond_model_tex_1024",
                ):
                    getattr(pipeline, attr).cuda()

            self._pipeline = pipeline
            logger.info("Pixal3D pipeline loaded and resident.")
            return self._pipeline

    @staticmethod
    def _from_pretrained_without_rembg(pipeline_cls: Any, path: str) -> Any:
        """upstream の Pixal3DImageTo3DPipeline.from_pretrained 相当の再実装。

        唯一の差分: rembg_model (briaai/RMBG-2.0, HFゲート付き) をロードしない。
        本アプリは背景除去済みRGBA画像を渡すため rembg_model は呼ばれない
        (pixal3d_image_to_3d.py preprocess_image の has_alpha 分岐参照)。
        """
        import json

        from huggingface_hub import hf_hub_download
        from pixal3d import models as models_mod
        from pixal3d.pipelines import samplers

        config_file = "pipeline.json"
        is_local = os.path.exists(f"{path}/{config_file}")
        cfg_path = f"{path}/{config_file}" if is_local else hf_hub_download(path, config_file)
        with open(cfg_path) as f:
            args = json.load(f)["args"]

        _models = {}
        for k, v in args["models"].items():
            if hasattr(pipeline_cls, "model_names_to_load") and k not in pipeline_cls.model_names_to_load:
                continue
            try:
                _models[k] = models_mod.from_pretrained(f"{path}/{v}")
            except Exception:
                _models[k] = models_mod.from_pretrained(v)

        pipeline = pipeline_cls(_models)
        pipeline._pretrained_args = args

        for stage in ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"):
            sampler = getattr(samplers, args[stage]["name"])(**args[stage]["args"])
            setattr(pipeline, stage, sampler)
            setattr(pipeline, f"{stage}_params", args[stage]["params"])

        pipeline.shape_slat_normalization = args["shape_slat_normalization"]
        pipeline.tex_slat_normalization = args["tex_slat_normalization"]
        pipeline.image_cond_model_ss = None
        pipeline.image_cond_model_shape_512 = None
        pipeline.image_cond_model_shape_1024 = None
        pipeline.image_cond_model_tex_1024 = None
        pipeline.rembg_model = None  # 意図的に非ロード (docstring参照)
        pipeline.low_vram = args.get("low_vram", True)
        pipeline.default_pipeline_type = args.get("default_pipeline_type", "1024_cascade")
        pipeline.pbr_attr_layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }
        pipeline._device = "cpu"
        return pipeline

    @staticmethod
    def _setup_image_cond_models(pipeline: Any) -> None:
        """DinoV3画像条件付けモデル4種を構築し、NAFアップサンプラを事前ロードする。"""
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
            DinoV3ProjFeatureExtractor,
        )

        for attr, key in (
            ("image_cond_model_ss", "ss"),
            ("image_cond_model_shape_512", "shape_512"),
            ("image_cond_model_shape_1024", "shape_1024"),
            ("image_cond_model_tex_1024", "tex_1024"),
        ):
            model = DinoV3ProjFeatureExtractor(**_IMAGE_COND_CONFIGS[key])
            model.eval()
            if getattr(model, "use_naf_upsample", False):
                # NAF重み (~2.5MB) を初回に torch.hub 経由でDL (natten必須)
                model._load_naf()
            setattr(pipeline, attr, model)

    # --- 生成 ----------------------------------------------------------------
    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        extra_views: Optional[dict[str, Image.Image]] = None,
    ) -> trimesh.Trimesh:
        if extra_views:
            raise ValueError(
                "pixal3dジェネレータはマルチビュー入力に対応していません。"
                "単一画像 (front) のみで生成するか、hunyuan3dジェネレータを使用してください。"
            )

        if not _has_meaningful_alpha(image):
            raise RuntimeError(
                "Pixal3D は背景除去済み (アルファチャンネル付き) の入力画像が必要です。"
                "remove_bg=true で生成してください (Pixal3D公式の背景除去モデル "
                "briaai/RMBG-2.0 はHFゲート付きのため本アプリでは使用しません)。"
            )

        pipeline = self._load_pipeline()

        import torch

        seed = int(params.seed) if params.seed is not None else int(
            torch.randint(0, 2**31 - 1, (1,)).item()
        )

        fov = config.PIXAL3D_FOV
        camera_params = {
            "camera_angle_x": fov,
            "distance": _distance_from_fov(fov),
            "mesh_scale": 1.0,
        }
        pipeline_type = f"{config.PIXAL3D_RESOLUTION}_cascade"

        # steps は3ステージ (sparse structure / shape / texture) のサンプラに接続する。
        # guidance_scale / octree_resolution は Pixal3D のステージごとに調整済みの
        # 既定値と互換性が無いため接続しない (SPEC.md §3.3 注記)。
        sampler_override = {"steps": int(params.steps)}

        try:
            image_preprocessed = pipeline.preprocess_image(image)
            torch.manual_seed(seed)
            # upstream inference.py と同様 return_latent=True で解像度 res を受け取る
            # (to_glb の grid_size に必要)。
            mesh_list, (_shape_slat, _tex_slat, res) = pipeline.run(
                image_preprocessed,
                camera_params=camera_params,
                seed=seed,
                sparse_structure_sampler_params=dict(sampler_override),
                shape_slat_sampler_params=dict(sampler_override),
                tex_slat_sampler_params=dict(sampler_override),
                preprocess_image=False,
                return_latent=True,
                pipeline_type=pipeline_type,
                max_num_tokens=49152,
            )
        except Exception as exc:
            raise RuntimeError(f"Pixal3D での3Dメッシュ生成に失敗しました: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not mesh_list:
            raise RuntimeError(
                "Pixal3D がメッシュを生成できませんでした (出力が空でした)。"
                "入力画像や生成パラメータを見直してください。"
            )
        raw = mesh_list[0]

        try:
            textured = self._to_textured_trimesh(pipeline, raw, res)
        except Exception as exc:
            raise RuntimeError(f"Pixal3D 出力のGLB変換に失敗しました: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # テクスチャ (PBR baseColor) から頂点カラーをサンプリングし ColorVisuals で返す。
        # UV/テクスチャは meshproc の簡略化で失われるため、この時点で頂点カラー化して
        # おき、jobs.py が後処理後メッシュへ最近傍転写する (SPEC.md §3.7 / §3.9)。
        vertex_colors = sample_vertex_colors_from_texture(textured)
        mesh = trimesh.Trimesh(
            vertices=np.asarray(textured.vertices),
            faces=np.asarray(textured.faces),
            process=False,
        )
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)

        # 重要: to_glb の出力はUVアトラス境界で頂点が複製されており、そのままでは
        # 数万個の連結成分(UVアイランド単位)に分断されている。meshproc の
        # 浮遊小部品除去(1%体積未満を削除)が本体表面まで削除してしまうため、
        # 位置ベースで頂点を溶接して面の連結を復元する(実測: 929k面のメッシュで
        # 39,819成分 → 溶接後は主要1成分が全面数の98.7%)。頂点カラーは
        # trimesh が溶接後も保持する。
        mesh.merge_vertices()

        # 座標系変換: Pixal3D (o_voxel.to_glb) の出力は 上=-Z / 正面=+Y (実測)。
        # 本アプリの Z-up / 正面=-Y へ X軸まわり180°回転で変換する。
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
        return mesh

    @staticmethod
    def _to_textured_trimesh(pipeline: Any, raw: Any, grid_size: Any) -> trimesh.Trimesh:
        """MeshWithVoxel を o_voxel.postprocess.to_glb でテクスチャ付きtrimeshに変換する。"""
        import o_voxel

        _inject_rasterizer(o_voxel)

        # upstream inference.py と同じ引数 (texture_size/decimationは設定値)。
        glb = o_voxel.postprocess.to_glb(
            vertices=raw.vertices,
            faces=raw.faces,
            attr_volume=raw.attrs,
            coords=raw.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=grid_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=config.PIXAL3D_DECIMATION_TARGET,
            texture_size=config.PIXAL3D_TEXTURE_SIZE,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=False,
        )
        if not isinstance(glb, trimesh.Trimesh):
            raise RuntimeError(
                f"o_voxel.postprocess.to_glb の出力をtrimesh.Trimeshとして"
                f"認識できませんでした (型: {type(glb)!r})。"
            )
        return glb
