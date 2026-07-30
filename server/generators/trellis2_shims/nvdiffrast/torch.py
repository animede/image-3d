"""`nvdiffrast.torch` 互換シム — pixal3d_raster (drtkベース, MIT) へ転送する。

o_voxel.postprocess.to_glb が使う3 API (RasterizeCudaContext / rasterize /
interpolate) のみを提供する。`server` パッケージが import 可能ならそれを使い、
そうでない場合 (sys.path に repo ルートが無い実行形態) はファイルパスから
直接ロードする。
"""
import importlib.util as _ilu
from pathlib import Path as _Path

try:  # 通常経路: uvicorn が repo ルートから server.main:app を起動している
    from server.generators import pixal3d_raster as _mod
except ImportError:  # フォールバック: 自身の位置から pixal3d_raster.py を解決
    _raster_path = _Path(__file__).resolve().parents[2] / "pixal3d_raster.py"
    _spec = _ilu.spec_from_file_location("_trellis2_pixal3d_raster", _raster_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

RasterizeCudaContext = _mod.RasterizeCudaContext
rasterize = _mod.rasterize
interpolate = _mod.interpolate
is_available = _mod.is_available
