"""nvdiffrast フェイクパッケージ (TRELLIS.2 ジェネレータ用)。

o_voxel/__init__.py は import 時に postprocess を eager import し、
postprocess.py は `import nvdiffrast.torch as dr` をハードコードしている。
本パッケージは server/generators/pixal3d_raster.py (drtkベースのMITシム) に
転送することで、nvdiffrast 本体 (NVIDIA Source Code License, 非商用限定)
なしで import を成立させる。本物の nvdiffrast が import できる環境では
このディレクトリは sys.path に追加されない (trellis2.py の _ensure_shims)。
"""
from . import torch  # noqa: F401
