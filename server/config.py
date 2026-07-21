"""アプリケーション設定。

すべて環境変数 `IMAGE3D_*` で上書き可能。コードにハードコードしない方針
(DEVELOPMENT_POLICY.md §4)。
"""
from __future__ import annotations

import os
from pathlib import Path

# --- パス ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("IMAGE3D_DATA_DIR", BASE_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"
WEB_DIR = Path(os.environ.get("IMAGE3D_WEB_DIR", BASE_DIR / "web"))

# --- サーバ ---------------------------------------------------------------
HOST = os.environ.get("IMAGE3D_HOST", "127.0.0.1")
PORT = int(os.environ.get("IMAGE3D_PORT", "8100"))

# --- ジェネレータ選択 (SPEC.md §3.3) --------------------------------------
# "auto": GPU + hy3dgen が利用可能なら hunyuan3d、なければ mock に自動解決
# (server/main.py の _build_generator)。テストは conftest.py で mock を明示する。
GENERATOR = os.environ.get("IMAGE3D_GENERATOR", "auto")

# --- アップロード制限 (FR-1) ------------------------------------------------
MAX_UPLOAD_BYTES = int(os.environ.get("IMAGE3D_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}

# --- 生成パラメータのデフォルト (SPEC.md §3.3, §5) ---------------------------
DEFAULT_STEPS = int(os.environ.get("IMAGE3D_DEFAULT_STEPS", "30"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("IMAGE3D_DEFAULT_GUIDANCE_SCALE", "5.5"))
DEFAULT_OCTREE_RESOLUTION = int(os.environ.get("IMAGE3D_DEFAULT_OCTREE_RESOLUTION", "384"))
ALLOWED_OCTREE_RESOLUTIONS = {256, 384, 512}
DEFAULT_REMOVE_BG = os.environ.get("IMAGE3D_DEFAULT_REMOVE_BG", "true").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_TARGET_HEIGHT_MM = float(os.environ.get("IMAGE3D_DEFAULT_TARGET_HEIGHT_MM", "100"))
DEFAULT_MAX_FACES = int(os.environ.get("IMAGE3D_DEFAULT_MAX_FACES", "200000"))

# --- ビルドプレート目安 (FR-5) ---------------------------------------------
BUILD_PLATE_MM = float(os.environ.get("IMAGE3D_BUILD_PLATE_MM", "220"))

# --- Hunyuan3D-2 (hy3dgen) 設定 (Phase 2) ----------------------------------
# HuggingFaceリポジトリID / サブフォルダ(標準shapeモデル)。
# mini版に切り替える場合は IMAGE3D_HY3DGEN_SUBFOLDER=hunyuan3d-dit-v2-0-mini 等を指定する。
HY3DGEN_MODEL_PATH = os.environ.get("IMAGE3D_HY3DGEN_MODEL_PATH", "tencent/Hunyuan3D-2")
HY3DGEN_SUBFOLDER = os.environ.get("IMAGE3D_HY3DGEN_SUBFOLDER", "hunyuan3d-dit-v2-0")
# hy3dgen自体が参照するローカルキャッシュ探索先(未設定ならhy3dgen既定の ~/.cache/hy3dgen)。
# 設定時はプロセス起動時に環境変数 HY3DGEN_MODELS へ反映する(server/main.py)。
HY3DGEN_MODELS_DIR = os.environ.get("IMAGE3D_HY3DGEN_MODELS_DIR")

# --- Hunyuan3D-2 マルチビュー (hunyuan3d-dit-v2-mv) 設定 (Phase 3a, SPEC.md §3.8) ---
# 公式リポジトリは単一ビューモデルとは別の tencent/Hunyuan3D-2mv (subfolder
# hunyuan3d-dit-v2-mv) である(third_party/Hunyuan3D-2/README.md 参照)。
HY3DGEN_MV_MODEL_PATH = os.environ.get("IMAGE3D_HY3DGEN_MV_MODEL_PATH", "tencent/Hunyuan3D-2mv")
HY3DGEN_MV_SUBFOLDER = os.environ.get("IMAGE3D_HY3DGEN_MV_SUBFOLDER", "hunyuan3d-dit-v2-mv")

# --- Hunyuan3D-2 paint (texgen) 設定 (Phase 3c, SPEC.md §3.9 / FR-10) ------------
# paintパイプラインは同じ tencent/Hunyuan3D-2 リポジトリの
# hunyuan3d-paint-v2-0 (multiview拡散) + hunyuan3d-delight-v2-0 (ライト除去)
# サブフォルダを使用する(hy3dgen/texgen/pipelines.py 参照)。
HY3DGEN_PAINT_SUBFOLDER = os.environ.get("IMAGE3D_HY3DGEN_PAINT_SUBFOLDER", "hunyuan3d-paint-v2-0")

# --- Pixal3D 設定 (SPEC.md §3.3。requirements-pixal3d.txt / .venv-pixal3d 前提) ---
# Pixal3D本体のリポジトリclone先(pipインストールせずsys.path経由でimportする)
PIXAL3D_REPO_DIR = Path(
    os.environ.get("IMAGE3D_PIXAL3D_REPO_DIR", BASE_DIR / "third_party" / "Pixal3D")
)
# HuggingFaceリポジトリID(モデル重み ~23GB、初回実行時に自動DL)
PIXAL3D_MODEL_PATH = os.environ.get("IMAGE3D_PIXAL3D_MODEL_PATH", "TencentARC/Pixal3D")
# 低VRAMモード: モデルをCPU常駐させステージごとにGPUへロード
# (実測: 低VRAM+1024でプロセスVRAMピーク ~16GB。標準+1536は全モデル常駐で高VRAM)
PIXAL3D_LOW_VRAM = os.environ.get("IMAGE3D_PIXAL3D_LOW_VRAM", "true").lower() in (
    "1",
    "true",
    "yes",
)
# パイプライン解像度 (1024 or 1536)
PIXAL3D_RESOLUTION = int(os.environ.get("IMAGE3D_PIXAL3D_RESOLUTION", "1024"))
# カメラ水平FOV (ラジアン)。MoGe(自動FOV推定)は導入しないため固定値を使う。
# 実機検証(momo.png)では 0.6 rad (~34°) で入力画像に忠実な結果を確認した。
PIXAL3D_FOV = float(os.environ.get("IMAGE3D_PIXAL3D_FOV", "0.6"))
# GLB化(o_voxel.postprocess.to_glb)のテクスチャサイズ / デシメーション目標
PIXAL3D_TEXTURE_SIZE = int(os.environ.get("IMAGE3D_PIXAL3D_TEXTURE_SIZE", "2048"))
PIXAL3D_DECIMATION_TARGET = int(os.environ.get("IMAGE3D_PIXAL3D_DECIMATION_TARGET", "1000000"))
# UVテクスチャベイクのラスタライザ選択 ("auto" | "drtk" | "nvdiffrast")。
# auto: drtk (Meta製, MIT) がimport可能ならdrtkを使用し、そうでなければ
# nvdiffrast (NVIDIA Source Code License, 非商用限定) にフォールバックする
# (server/generators/pixal3d_raster.py 参照)。
PIXAL3D_RASTERIZER = os.environ.get("IMAGE3D_PIXAL3D_RASTERIZER", "auto")
_PIXAL3D_RASTERIZER_CHOICES = {"auto", "drtk", "nvdiffrast"}
if PIXAL3D_RASTERIZER not in _PIXAL3D_RASTERIZER_CHOICES:
    raise ValueError(
        f"IMAGE3D_PIXAL3D_RASTERIZER は {sorted(_PIXAL3D_RASTERIZER_CHOICES)} のいずれかを"
        f"指定してください (got: {PIXAL3D_RASTERIZER!r})。"
    )


# --- マルチモーダルLLMによるパーツ検出 (SPEC.md §3.12 / FR-13 第3層誘導) --------
# OpenAI互換チャットAPI(例: llama.cpp + vision対応モデル)のベースURL。
# 既定は空文字列(=無効)。公開リポジトリのためユーザー固有のIPをデフォルトに
# 設定しない(ユーザー環境では例えば IMAGE3D_LLM_ENDPOINT=http://<host>:<port> を指定)。
IMAGE3D_LLM_ENDPOINT = os.environ.get("IMAGE3D_LLM_ENDPOINT", "")
IMAGE3D_LLM_TIMEOUT = float(os.environ.get("IMAGE3D_LLM_TIMEOUT", "60"))


def ensure_dirs() -> None:
    """必要なディレクトリを作成する。"""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
