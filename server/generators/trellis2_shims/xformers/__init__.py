"""xformers 互換スタブ (TRELLIS.2 の SPARSE_ATTN_BACKEND=xformers 経路用)。

third_party/TRELLIS.2 の sparse attention は xformers / flash_attn のみ対応で
SDPA バックエンドが無い。本物の xformers は torch 2.11+cu128 / sm_120 向け
wheel の ABI 適合が不確実なため、必要最小限の API を PyTorch SDPA で代替する
純 torch 実装を使う (検証スパイク trellis2-hybrid-20260730 で移植。
参照実装とのブロック対角アテンション最大誤差 0.0 を確認済み)。

このディレクトリ (trellis2_shims) は本物の xformers が import できない場合に
のみ sys.path へ追加される (server/generators/trellis2.py の _ensure_shims)。
"""
from . import ops  # noqa: F401

__version__ = "0.0.0+image3d-trellis2-sdpa-stub"
