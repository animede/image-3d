"""TRELLIS.2-4B (microsoft/TRELLIS.2, MITライセンス) 形状エンジンジェネレータ。

単一画像から「形状 + 自前テクスチャ」を生成する。検証スパイク
(data/spikes/trellis2-hybrid-20260730, メモリ trellis2-hybrid-spike-verdict) の
採用判断に基づく実装で、狙いは **形状忠実度** (指の分離・髪の房・眼窩の造形など
Hunyuan3D-2 では出ない細部) と、既存 texrefine による参照画像の全解像度反映の
両立 (ハイブリッド構成):

    TRELLIS.2 生成 (~25s, VRAM ~3.3GB)
      → o_voxel.postprocess.to_glb(remesh=True)  (~4分, VRAM ~4.5GB,
         narrow-band Dual Contouring で実質閉曲面化 + GPU UV展開)
      → jobs.py が texgen をスキップして texrefine を直接適用
         (metadata["pretextured_mesh"] 経由。texgen の 512px 天井を通らない)

前提環境: pixal3d と同じ専用venv .venv-pixal3d (torch cu128 / o_voxel / cumesh /
flex_gemm / drtk)。third_party/TRELLIS.2 のcloneを sys.path 経由でimportする。
サーバは .venv-pixal3d/bin/uvicorn で起動する (.claude/launch.json の
image3d-server-trellis2 参照)。

実装上の注意 (スパイクで実測・確認した非自明な事実):
  - sparse attention はこの checkout では xformers / flash_attn のみ対応
    (SDPAなし)。本物のxformersは入れず、SDPAベースの互換スタブ
    (trellis2_shims/xformers) を SPARSE_ATTN_BACKEND=xformers で使う。
  - o_voxel は import 時に nvdiffrast を要求する。本物 (非商用ライセンス) は
    導入せず、drtkへ転送するフェイクパッケージ (trellis2_shims/nvdiffrast) で
    import を成立させ、さらに to_glb 実行前に pixal3d._inject_rasterizer で
    `o_voxel.postprocess.dr` を明示的にMITシムへ差し替える。
  - rembg (briaai/RMBG-2.0) はHFゲート付きのためロードしない。本アプリの
    前処理済みRGBA画像を渡す (pipeline.preprocess_image は has_alpha 分岐で
    rembg を通らない)。
  - DINOv3 (facebook/dinov3-*) はHFゲート付き。403 の場合は camenduru/ の
    ミラーへフォールバックする。
  - 座標系: to_glb の出力は Y-up / 正面+Z (pixal3d の 上=-Z/正面+Y とは
    **異なる**。スパイク bake_glb.py で実測)。本アプリの Z-up / 正面-Y へは
    X軸まわり +90°回転で変換する。
  - 色空間: to_glb が焼くテクスチャ (デコード済み base_color 属性) は
    sRGB符号化済み。ガンマ変換は不要 (色管理監査 2026-07-30)。
  - to_glb 出力はUVアトラス境界で頂点が複製されている。meshproc の浮遊小部品
    除去が本体を削らないよう、頂点カラー化した戻り値は merge_vertices で
    溶接する (pixal3d と同じ)。テクスチャ付きメッシュ (pretextured_mesh) は
    UVを保つため溶接しない。
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh
from PIL import Image

from .. import config
from ..texture import sample_vertex_colors_from_texture
from .base import GenerationParams, Generator
from .pixal3d import _inject_rasterizer

logger = logging.getLogger(__name__)

_IMPORT_ERROR_HINT = (
    "TRELLIS.2 の依存関係が見つかりません。pixal3d と共通の専用venv "
    "(.venv-pixal3d: torch cu128 / o_voxel / cumesh / flex_gemm / drtk) と "
    "third_party/TRELLIS.2 のcloneが必要です。サーバは .venv-pixal3d/bin/uvicorn "
    "で起動してください (.claude/launch.json の image3d-server-trellis2 参照)。"
)

_SHIMS_DIR = Path(__file__).resolve().parent / "trellis2_shims"

# メタデータ経由で jobs.py へテクスチャ付きメッシュを渡すキー
# (server/jobs.py の paint 分岐が参照する)。
PRETEXTURED_MESH_KEY = "pretextured_mesh"


def _has_meaningful_alpha(image: Image.Image) -> bool:
    """背景除去済み (=部分的に透明な) RGBA画像かどうか。"""
    if image.mode != "RGBA":
        return False
    alpha = np.asarray(image.getchannel("A"))
    return bool((alpha < 255).any())


def _ensure_shims() -> None:
    """本物の xformers / nvdiffrast が無い場合のみ互換スタブを sys.path に載せる。

    どちらか一方でも import に失敗したらスタブディレクトリを追加する
    (両スタブは同居しており、本物が存在する側は import 優先順で本物が勝つ:
    site-packages は sys.path 上でスタブより先には来ないため、追加は
    「見つからない場合の補完」に限定する)。
    """
    missing = False
    for module_name in ("xformers", "nvdiffrast"):
        try:
            __import__(module_name)
        except ImportError:
            missing = True
            break
        except Exception:
            # import はできたが初期化で失敗する壊れた導入は本物を尊重する
            # (スタブで隠すと原因究明が難しくなるため)。
            continue
    if missing and str(_SHIMS_DIR) not in sys.path:
        sys.path.insert(0, str(_SHIMS_DIR))
        logger.info("TRELLIS.2: 互換スタブを使用します (%s)", _SHIMS_DIR)


class Trellis2Generator(Generator):
    """TRELLIS.2-4B image-to-3D パイプラインを用いたジェネレータ。

    出力はテクスチャからサンプリングした頂点カラー付き (ColorVisuals) の
    溶接済み trimesh.Trimesh。加えて `mesh.metadata["pretextured_mesh"]` に
    UV+テクスチャ付きメッシュ (未溶接・生成スケールのまま) を格納する。
    jobs.py は texture_mode=paint 時にこれを texgen の代わりに使用する。

    生成自体は単一画像 (front) のみを使う。extra_views はエラーにせず無視する
    (back/left/right は jobs.py 側で texrefine の参照として活用されるため、
    マルチビューのジョブを弾いてはならない)。
    """

    name = "trellis2"
    # jobs.py / /api/health: texgen (hy3dgen) が無くても texture_mode=paint を
    # 提供できることを示す (テクスチャはジェネレータ自身が生成する)。
    provides_texture = True

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

            repo_dir = str(config.TRELLIS2_REPO_DIR)
            if not os.path.isdir(os.path.join(repo_dir, "trellis2")):
                raise RuntimeError(
                    f"TRELLIS.2リポジトリが見つかりません ({repo_dir})。"
                    "`git clone https://github.com/microsoft/TRELLIS.2 "
                    "third_party/TRELLIS.2` を実行してください。"
                )

            # attention backend は import 時に環境変数から確定するため、
            # import 前に設定する。sparse 側は xformers (=スタブ) が必要
            # (この checkout の sparse attention に SDPA 経路は無い)。
            os.environ.setdefault("ATTN_BACKEND", "sdpa")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
            os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            _ensure_shims()
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)

            try:
                import torch
                import trellis2.modules.image_feature_extractor as ife
                import trellis2.pipelines.rembg as trembg
                from trellis2.pipelines import Trellis2ImageTo3DPipeline
            except ImportError as exc:
                raise ImportError(_IMPORT_ERROR_HINT) from exc

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "TRELLIS.2 はGPU (CUDA) 必須です。GPUが利用できない環境では "
                    "IMAGE3D_GENERATOR=mock を使用してください。"
                )

            self._patch_rembg(trembg)
            self._patch_dinov3_fallback(ife)

            logger.info(
                "Loading TRELLIS.2 pipeline (%s); this may take a while on first "
                "run (downloads ~16GB from HuggingFace)...",
                config.TRELLIS2_MODEL_PATH,
            )
            try:
                pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                    config.TRELLIS2_MODEL_PATH
                )
                pipeline.cuda()
            except Exception as exc:
                raise RuntimeError(
                    f"TRELLIS.2 パイプラインのロードに失敗しました "
                    f"(model_path={config.TRELLIS2_MODEL_PATH}): {exc}"
                ) from exc

            self._pipeline = pipeline
            logger.info("TRELLIS.2 pipeline loaded and resident.")
            return self._pipeline

    @staticmethod
    def _patch_rembg(trembg: Any) -> None:
        """rembg (briaai/RMBG-2.0, HFゲート付き) をロードさせない。

        本アプリは背景除去済みRGBA画像を渡すため rembg 経路は通らないが、
        万一アルファ無し画像が内部まで到達した場合に意味のあるエラーを出す。
        """

        class _NoRembg:
            def __init__(self, *args, **kwargs):
                pass

            def to(self, *args, **kwargs):
                pass

            def cuda(self):
                pass

            def cpu(self):
                pass

            def __call__(self, image):
                raise RuntimeError(
                    "TRELLIS.2 の内蔵背景除去 (briaai/RMBG-2.0) は本アプリでは"
                    "使用しません。背景除去済み (アルファ付き) の画像を入力して"
                    "ください。"
                )

        trembg.BiRefNet = _NoRembg

    @staticmethod
    def _patch_dinov3_fallback(ife: Any) -> None:
        """DINOv3 (facebook/*, HFゲート付き) を camenduru/ ミラーへフォールバック。"""
        orig = ife.DINOv3ViTModel
        if getattr(orig, "_image3d_fallback_patched", False):
            return

        class _DinoRedirect:
            _image3d_fallback_patched = True

            @staticmethod
            def from_pretrained(name: str, *args: Any, **kwargs: Any) -> Any:
                try:
                    return orig.from_pretrained(name, *args, **kwargs)
                except Exception as exc:
                    if name.startswith("facebook/dinov3"):
                        alt = "camenduru/" + name.split("/", 1)[1]
                        logger.warning(
                            "DINOv3 %s のロードに失敗 (%s)。ミラー %s へ"
                            "フォールバックします。",
                            name,
                            exc,
                            alt,
                        )
                        return orig.from_pretrained(alt, *args, **kwargs)
                    raise

        ife.DINOv3ViTModel = _DinoRedirect

    # --- 生成 ----------------------------------------------------------------
    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        extra_views: Optional[dict[str, Image.Image]] = None,
    ) -> trimesh.Trimesh:
        if extra_views:
            # 生成は単一画像のみ。back/left/right は jobs.py が texrefine の
            # 参照として使うため、ここで弾かず無視する (base.py docstring 参照)。
            logger.info(
                "TRELLIS.2: 追加ビュー (%s) は形状生成には使用しません "
                "(texture_refine の参照としてのみ利用されます)。",
                ", ".join(sorted(extra_views)),
            )

        if not _has_meaningful_alpha(image):
            raise RuntimeError(
                "TRELLIS.2 は背景除去済み (アルファチャンネル付き) の入力画像が"
                "必要です。remove_bg=true で生成してください (内蔵の背景除去"
                "モデル briaai/RMBG-2.0 はHFゲート付きのため使用しません)。"
            )

        pipeline = self._load_pipeline()

        import torch

        seed = int(params.seed) if params.seed is not None else int(
            torch.randint(0, 2**31 - 1, (1,)).item()
        )
        # steps は3ステージ (sparse structure / shape SLat / texture SLat) の
        # サンプラに接続する (pixal3d と同じ方針。TRELLIS.2 の既定は12)。
        sampler_override = {"steps": int(params.steps)}

        try:
            mesh_list = pipeline.run(
                image,
                seed=seed,
                sparse_structure_sampler_params=dict(sampler_override),
                shape_slat_sampler_params=dict(sampler_override),
                tex_slat_sampler_params=dict(sampler_override),
                pipeline_type=config.TRELLIS2_PIPELINE_TYPE,
            )
        except Exception as exc:
            raise RuntimeError(f"TRELLIS.2 での3Dメッシュ生成に失敗しました: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not mesh_list:
            raise RuntimeError(
                "TRELLIS.2 がメッシュを生成できませんでした (出力が空でした)。"
                "入力画像や生成パラメータを見直してください。"
            )
        raw = mesh_list[0]

        try:
            textured = self._to_textured_trimesh(raw)
        except Exception as exc:
            raise RuntimeError(f"TRELLIS.2 出力のGLB変換に失敗しました: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 座標系変換: to_glb 出力 (Y-up / 正面+Z) → アプリ規約 (Z-up / 正面-Y)。
        textured.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        )

        # meshproc 用の戻り値: テクスチャから頂点カラー化し、UVアトラス境界の
        # 頂点複製を位置ベースで溶接する (浮遊小部品除去が本体を削らないため。
        # pixal3d の実測: 溶接で数万成分 → 主要1成分に復元)。
        vertex_colors = sample_vertex_colors_from_texture(textured)
        mesh = trimesh.Trimesh(
            vertices=np.asarray(textured.vertices),
            faces=np.asarray(textured.faces),
            process=False,
        )
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
        mesh.merge_vertices()

        # texgen を通さない paint 経路 (jobs.py) へ、UV+テクスチャ付きメッシュを
        # そのまま渡す (未溶接・未スケール。スケーリングは jobs.py 側で
        # meshproc 済みメッシュに合わせて行う)。
        mesh.metadata[PRETEXTURED_MESH_KEY] = textured
        return mesh

    @staticmethod
    def _to_textured_trimesh(raw: Any) -> trimesh.Trimesh:
        """MeshWithVoxel を to_glb(remesh=True) でテクスチャ付きtrimeshに変換する。

        remesh=True (narrow-band Dual Contouring) が必須:
        生メッシュは開放薄シェル約3万成分で、そのままUVベイクすると texrefine の
        遮蔽判定がテクセル単位で明滅し紙吹雪状ノイズになる (スパイク実測)。
        DC で実質閉曲面 (boundary_edges=3) になり、この問題が消える。
        """
        import o_voxel

        _inject_rasterizer(o_voxel)

        # ラスタライザのインデックス上限に合わせて事前に面数を抑える
        # (upstream example.py と同じ)。
        raw.simplify(16777216)

        glb = o_voxel.postprocess.to_glb(
            vertices=raw.vertices,
            faces=raw.faces,
            attr_volume=raw.attrs,
            coords=raw.coords,
            attr_layout=raw.layout,
            voxel_size=raw.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=config.TRELLIS2_DECIMATION_TARGET,
            texture_size=config.TRELLIS2_TEXTURE_SIZE,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
        )
        if not isinstance(glb, trimesh.Trimesh):
            raise RuntimeError(
                f"o_voxel.postprocess.to_glb の出力をtrimesh.Trimeshとして"
                f"認識できませんでした (型: {type(glb)!r})。"
            )
        return glb
