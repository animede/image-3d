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
            logger.info("Hunyuan3D-2 paint pipeline loaded and resident.")
            return self._pipeline

    def paint(self, mesh: trimesh.Trimesh, image: Image.Image) -> trimesh.Trimesh:
        """メッシュ(mm スケール、Z-up)+ 正面画像(背景除去後)からテクスチャ付き
        メッシュを生成する。

        Args:
            mesh: 後処理済みメッシュ(meshproc.process適用後、mm スケール)。
            image: 正面画像(背景除去後推奨、RGBA)。

        Returns:
            UV + テクスチャ画像(TextureVisuals)付きの trimesh.Trimesh。
            頂点座標は入力meshと同じ座標系・スケールを維持する。
        """
        pipeline = self._load_pipeline()

        import torch

        # paint pipeline はメッシュを破壊的に変更しうるためコピーを渡す。
        mesh_copy = mesh.copy()
        try:
            textured = pipeline(mesh_copy, image)
        except Exception as exc:
            raise RuntimeError(f"Hunyuan3D-2 texgen でのテクスチャ生成に失敗しました: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not isinstance(textured, trimesh.Trimesh):
            raise RuntimeError(
                f"texgen の出力をtrimesh.Trimeshとして認識できませんでした(型: {type(textured)!r})。"
            )
        return textured


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
