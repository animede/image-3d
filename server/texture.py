"""テクスチャ生成 (Hunyuan3D-2 paint pipeline) 統合 (SPEC.md §3.9 / FR-10)。

Phase 3c: texture_mode=paint 時、Hunyuan3D-2 の texgen (Hunyuan3DPaintPipeline) を
用いて全周テクスチャ付きメッシュを生成する。

構成:
    - `is_available()`: 依存import + CUDA拡張(custom_rasterizer_kernel)存在確認のみを
      行う可用性チェック(実際のモデルロードは行わない)。`/api/health` の
      `texgen_available` に使う。
    - `TexturePipelineWrapper`: paint パイプラインの遅延ロード・常駐ラッパ。
      `generators/hunyuan3d.py` と同様、初回呼び出し時にのみロードし、以降常駐する
      (NFR-3)。GPU未搭載環境では常に失敗させる(CPU実行は非対応・非現実的なため)。
    - `sample_vertex_colors_from_texture()`: テクスチャ付きメッシュ(UV + PIL Image)
      から頂点カラー(RGBA uint8)をサンプリングする純関数。GPU不要、単体テスト対象。

paint後のメッシュはビルド前のメッシュ(mm スケール、Z-up)の頂点位置をそのまま
保持する(hy3dgenの `mesh_uv_wrap` はxatlasによる頂点複製/面再構成のみ行い、
座標値・スケールは変更しない。`MeshRender` 内部のレンダリング用正規化コピーは
出力には反映されず、`save_mesh` は `mesh_uv_wrap` 後のメッシュ座標を用いる)。
そのため呼び出し側で単位変換(mm<->正規化)を行う必要はない。
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np
import trimesh
from PIL import Image

from . import config

logger = logging.getLogger(__name__)

_IMPORT_ERROR_HINT = (
    "texgen (Hunyuan3D-2 paint pipeline) の依存関係が見つかりません。README.md の"
    " texgenセットアップ手順を参照し、custom_rasterizer拡張のビルドと"
    " xatlas/pygltflib等の追加依存を導入してください。"
)

# 検証スパイクの実測: third_party/Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/unet/modules.py:456
# `self.max_num_ref_image = 5` が参照画像の埋め込みテーブルの上限を定めているため、
# それを超える枚数を渡すと class_labels の埋め込みlookupが範囲外になる。
MAX_REF_IMAGES = 5


def is_available() -> bool:
    """texgenが利用可能か(依存import + CUDA拡張の存在確認のみ、実ロードはしない)。

    `/api/health` の `texgen_available` に使う軽量チェック。GPU無し環境や
    custom_rasterizer未ビルド環境では False を返す。
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        import custom_rasterizer_kernel  # noqa: F401
        from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: F401
    except Exception:
        return False
    return True


def _patch_multiview_ref_camera_info(pipeline: Any) -> None:
    """`Multiview_Diffusion_Net.__call__` の `camera_info_ref` 固定値を実行時ラップで解消する。

    根拠(検証スパイクでのコード読解、third_party/Hunyuan3D-2/hy3dgen/texgen/ 配下):
      - `utils/multiview_utils.py:80` の `camera_info_ref = [[0]]` は「参照画像は
        常に1枚、インデックス0」というハードコード。参照画像を複数渡しても
        2枚目以降のcamera embeddingが使われず、背面ビューへ正面の配色が
        回り込む原因になっていた。
      - `hunyuanpaint/unet/modules.py:456` の `self.max_num_ref_image = 5` が
        参照側camera embeddingテーブルの上限。参照画像のインデックスは生成側
        (azim/elevから符号化される値域[0,44)、pipelines.py参照)とは別体系で、
        素の0始まりインデックス(0,1,2,...)をそのまま使うのが正しい
        (modules.py:508 の生成側 `+ max_num_ref_image` オフセットは参照側には
        加算されないため)。
      - スパイク(painted_multi.glb)で、この拡張により背面ビューに背面参照画像の
        配色(帽子の赤・服の紺)が反映されることを確認済み。VRAMピーク9.69GB、
        生成時間は1枚時と同等(常駐後~7秒)。

    vendored ファイル(multiview_utils.py)自体は書き換えず、ロード済み
    パイプラインインスタンスの `multiview_model.__call__` をこの関数でラップする。
    想定した属性構造(`pipeline.models['multiview_model']`)が無い場合は例外にせず、
    警告ログを出して従来動作(参照1枚固定)にフォールバックする
    (このリポジトリの graceful degradation の流儀に合わせる)。
    """
    try:
        from hy3dgen.texgen.utils import multiview_utils as mvu

        multiview_model = pipeline.models["multiview_model"]
        if not isinstance(multiview_model, mvu.Multiview_Diffusion_Net):
            raise TypeError(
                f"pipeline.models['multiview_model'] は "
                f"Multiview_Diffusion_Net ではありません(型: {type(multiview_model)!r})"
            )
    except Exception as exc:
        logger.warning(
            "Could not attach the multi-reference camera_info_ref patch "
            "(falling back to single reference image only): %s",
            exc,
        )
        return

    def patched_call(self, input_images, control_images, camera_info):
        from typing import List as _List

        self.seed_everything(0)

        if not isinstance(input_images, _List):
            input_images = [input_images]

        n_ref = len(input_images)

        input_images = [
            im.resize((self.view_size, self.view_size)) for im in input_images
        ]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((self.view_size, self.view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")

        import torch as _torch

        kwargs = dict(generator=_torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        camera_info_gen = [camera_info]
        # ここが本パッチの核: 参照画像の枚数ぶんの camera_info_ref を生成する
        # (固定 [[0]] ではなく [[0, 1, ...]])。
        camera_info_ref = [list(range(n_ref))]

        kwargs["width"] = self.view_size
        kwargs["height"] = self.view_size
        kwargs["num_in_batch"] = num_view
        kwargs["camera_info_gen"] = camera_info_gen
        kwargs["camera_info_ref"] = camera_info_ref
        kwargs["normal_imgs"] = normal_image
        kwargs["position_imgs"] = position_image

        mvd_image = self.pipeline(input_images, num_inference_steps=30, **kwargs).images
        return mvd_image

    # インスタンス単位でバインドし直す(クラス自体は書き換えない = 他の
    # TexturePipelineWrapper/将来のパイプラインインスタンスには影響しない)。
    import types

    multiview_model.__call__ = types.MethodType(patched_call, multiview_model)
    logger.info(
        "Patched Multiview_Diffusion_Net.__call__ on the loaded pipeline instance "
        "to support multiple reference images (camera_info_ref expansion)."
    )


class TexturePipelineWrapper:
    """Hunyuan3D-2 paint pipeline の遅延ロード・常駐ラッパ。

    `tencent/Hunyuan3D-2` の `hunyuan3d-paint-v2-0`(multiview拡散)と
    `hunyuan3d-delight-v2-0`(ライト除去)を内部で使用する。
    """

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None
        self._lock = threading.Lock()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            import os

            if config.HY3DGEN_MODELS_DIR:
                os.environ.setdefault("HY3DGEN_MODELS", config.HY3DGEN_MODELS_DIR)

            try:
                import torch  # noqa: F401
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
            except ImportError as exc:
                raise ImportError(_IMPORT_ERROR_HINT) from exc

            logger.info(
                "Loading Hunyuan3D-2 paint pipeline (%s, subfolder=%s); "
                "this may take a while on first run (downloads from HuggingFace)...",
                config.HY3DGEN_MODEL_PATH,
                config.HY3DGEN_PAINT_SUBFOLDER,
            )
            try:
                pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                    config.HY3DGEN_MODEL_PATH,
                    subfolder=config.HY3DGEN_PAINT_SUBFOLDER,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Hunyuan3D-2 paint パイプラインのロードに失敗しました: {exc}"
                ) from exc

            self._pipeline = pipeline
            _patch_multiview_ref_camera_info(pipeline)
            logger.info("Hunyuan3D-2 paint pipeline loaded and resident.")
            return self._pipeline

    def paint(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        back_image: Optional[Image.Image] = None,
    ) -> trimesh.Trimesh:
        """メッシュ(mm スケール、Z-up)+ 参照画像からテクスチャ付きメッシュを生成する。

        検証スパイク(2026-07時点)で、`Hunyuan3DPaintPipeline.__call__(mesh, image)`
        は image がリストなら複数画像を受け付け、各画像に delight を適用したうえで
        multiview拡散の参照画像として使う実装であることを確認済み。ただし
        `multiview_utils.Multiview_Diffusion_Net.__call__` 内の `camera_info_ref` が
        `[[0]]` に固定されているため(third_party/Hunyuan3D-2/hy3dgen/texgen/utils/
        multiview_utils.py:80)、参照画像を増やしても2枚目以降が正しく紐付かず、
        背面ビューへ正面色が回り込む問題があった。`_load_pipeline` 内で
        `_patch_multiview_ref_camera_info` により実行時ラップし、参照画像枚数に
        応じて camera_info_ref を拡張することでこれを解消する(vendored コード
        自体は書き換えない)。

        Args:
            mesh: 後処理済みメッシュ(meshproc.process適用後、mm スケール)。
            image: 正面画像(背景除去後推奨、RGBA)。
            back_image: 背面参照画像(背景除去後推奨、RGBA)。省略時は正面のみの
                従来動作(参照画像1枚)。

        Returns:
            UV + テクスチャ画像(TextureVisuals)付きの trimesh.Trimesh。
            頂点座標は入力meshと同じ座標系・スケールを維持する。
        """
        pipeline = self._load_pipeline()

        import torch

        images: list[Image.Image] = [image]
        if back_image is not None:
            images.append(back_image)
        if len(images) > MAX_REF_IMAGES:
            # スパイクで確認した上限(max_num_ref_image=5)を超えないようガードする。
            # 現状呼び出し元(jobs.py)はfront+backの最大2枚しか渡さないため
            # 実運用では発生しないが、将来left/right追加時の安全弁として残す。
            logger.warning(
                "Truncating %d reference images to MAX_REF_IMAGES=%d for texgen paint.",
                len(images),
                MAX_REF_IMAGES,
            )
            images = images[:MAX_REF_IMAGES]

        # paint pipeline はメッシュを破壊的に変更しうるためコピーを渡す。
        mesh_copy = mesh.copy()
        try:
            pipeline_input = images[0] if len(images) == 1 else images
            textured = pipeline(mesh_copy, pipeline_input)
        except Exception as exc:
            raise RuntimeError(f"Hunyuan3D-2 texgen でのテクスチャ生成に失敗しました: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not isinstance(textured, trimesh.Trimesh):
            raise RuntimeError(
                f"texgen の出力をtrimesh.Trimeshとして認識できませんでした(型: {type(textured)!r})。"
            )
        if not ensure_non_metallic(textured):
            # 効かないまま出すと「金属扱いで暗い」成果物になるので気づけるようにする
            logger.warning(
                "Could not force metallicFactor=0 on the painted material (%s); "
                "the model may render dark in PBR viewers.",
                type(getattr(getattr(textured, "visual", None), "material", None)).__name__,
            )
        return textured


def ensure_non_metallic(mesh: trimesh.Trimesh) -> bool:
    """テクスチャ付きメッシュの材質に `metallicFactor = 0` を明示する。

    **glTF 2.0 の `metallicFactor` 既定値は 1.0(完全な金属)**であり、
    trimesh は値が未設定だとこのフィールドを書き出さない。結果、生成した
    キャラクターが「金属」として解釈される。金属は拡散反射を持たず環境の
    映り込みしか返さないため、環境マップを持たないビューア(Godot、
    three.jsの素のセットアップ、多くのglTFビューア)では**色が暗く沈む**。

    キャラクターは布・肌・毛といった非金属なので 0.0 を明示する。

    Returns:
        値を設定したら True(既に設定済み、または材質が無ければ False)。
    """
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    if material is None:
        return False

    if not hasattr(material, "metallicFactor"):
        # texgen が返すのは `SimpleMaterial` で、metallicFactor を持たない。
        # GLB書き出し時に trimesh が PBR へ変換するものの、そのとき
        # baseColorFactor と roughnessFactor は書かれるのに metallicFactor は
        # 落ちる。先に PBR へ変換しておかないと値を指定できない。
        if not hasattr(material, "to_pbr"):
            return False
        material = material.to_pbr()
        visual.material = material

    if material.metallicFactor is not None:
        return False
    material.metallicFactor = 0.0
    return True


def sample_vertex_colors_from_texture(mesh: trimesh.Trimesh) -> np.ndarray:
    """UV+テクスチャ付きメッシュから頂点カラー(RGBA uint8)をサンプリングする。

    GPU不要の純関数(単体テスト対象)。`mesh.visual` が `TextureVisuals` で
    `uv` と `material.image` (または `image`) を持つことを前提とする。

    Args:
        mesh: `trimesh.visual.TextureVisuals` を持つメッシュ(paint後の出力)。

    Returns:
        (N, 4) uint8 頂点カラー配列(RGBA)。テクスチャが無い場合は白(255)を返す。
    """
    n = len(mesh.vertices)
    visual = mesh.visual

    uv = getattr(visual, "uv", None)
    tex_image = _extract_texture_image(visual)

    if uv is None or tex_image is None:
        return np.full((n, 4), 255, dtype=np.uint8)

    if tex_image.mode != "RGBA":
        tex_image = tex_image.convert("RGBA")

    tex = np.asarray(tex_image, dtype=np.uint8)  # (h, w, 4)
    h, w = tex.shape[0], tex.shape[1]

    uv_arr = np.asarray(uv, dtype=np.float64)
    if len(uv_arr) != n:
        # UV数が頂点数と不一致(通常は発生しない)場合は白にフォールバック
        return np.full((n, 4), 255, dtype=np.uint8)

    u = np.clip(uv_arr[:, 0], 0.0, 1.0)
    # 画像テクスチャ座標はv=0が上端、UVはv=0が下端の慣例(OpenGL式)。
    v = np.clip(1.0 - uv_arr[:, 1], 0.0, 1.0)

    px = np.clip((u * (w - 1)).round().astype(np.int64), 0, w - 1)
    py = np.clip((v * (h - 1)).round().astype(np.int64), 0, h - 1)

    colors = tex[py, px]  # (N, 4)
    return colors.astype(np.uint8)


def _extract_texture_image(visual: Any) -> Optional[Image.Image]:
    """TextureVisualsからPIL Imageを取り出す(material.image優先、無ければvisual.image)。"""
    material = getattr(visual, "material", None)
    if material is not None:
        image = getattr(material, "image", None)
        if image is not None:
            return image
        base_color_texture = getattr(material, "baseColorTexture", None)
        if base_color_texture is not None:
            return base_color_texture
    image = getattr(visual, "image", None)
    if image is not None:
        return image
    return None
