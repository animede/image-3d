"""xformers.ops 互換の最小スタブ (SDPAベース)。

TRELLIS.2 が使うのは以下の2箇所のみ (実コード確認済み):
  - trellis2/modules/sparse/attention/full_attn.py:
      mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
      out = xops.memory_efficient_attention(q, k, v, mask)   # q: [1, T, H, C]
  - trellis2/modules/sparse/attention/windowed_attn.py:
      attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)  # 自己注意
      out = xops.memory_efficient_attention(q, k, v, attn_bias=...)

意味論: パックされた可変長系列のブロック対角アテンション。
実装: 系列ごとに [B, max_len] へパディングして F.scaled_dot_product_attention を
1回で呼ぶ (Pythonループなし)。パディング行は出力から捨てる。
"""
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


class BlockDiagonalMask:
    def __init__(self, q_seqlen, kv_seqlen):
        self.q_seqlen = [int(x) for x in q_seqlen]
        self.kv_seqlen = [int(x) for x in kv_seqlen]

    @classmethod
    def from_seqlens(cls, q_seqlen: Sequence[int], kv_seqlen: Optional[Sequence[int]] = None):
        if kv_seqlen is None:
            kv_seqlen = q_seqlen
        return cls(q_seqlen, kv_seqlen)


class fmha:
    BlockDiagonalMask = BlockDiagonalMask


def _pad_packed(x: torch.Tensor, seqlens: torch.Tensor, max_len: int):
    """x: [T, H, C] packed -> ([B, max_len, H, C], batch_idx, pos_idx)"""
    device = x.device
    B = seqlens.shape[0]
    batch_idx = torch.repeat_interleave(torch.arange(B, device=device), seqlens)
    starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=seqlens.dtype), seqlens[:-1]]), 0
    )
    pos_idx = torch.arange(x.shape[0], device=device) - starts[batch_idx]
    padded = x.new_zeros(B, max_len, *x.shape[1:])
    padded[batch_idx, pos_idx] = x
    return padded, batch_idx, pos_idx


def memory_efficient_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[BlockDiagonalMask] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """q/k/v: [1, T, H, C] (packed)。戻り値も [1, Tq, H, Cv]。"""
    assert q.ndim == 4 and q.shape[0] == 1, f"expected [1, T, H, C], got {tuple(q.shape)}"
    if attn_bias is None or len(attn_bias.q_seqlen) == 1:
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
        )
        return out.transpose(1, 2)

    device = q.device
    q_lens = torch.tensor(attn_bias.q_seqlen, device=device, dtype=torch.long)
    kv_lens = torch.tensor(attn_bias.kv_seqlen, device=device, dtype=torch.long)
    assert int(q_lens.sum()) == q.shape[1], "q_seqlen sum mismatch"
    assert int(kv_lens.sum()) == k.shape[1], "kv_seqlen sum mismatch"
    max_q = int(q_lens.max())
    max_kv = int(kv_lens.max())

    qp, q_batch, q_pos = _pad_packed(q[0], q_lens, max_q)  # [B, max_q, H, C]
    kp, _, _ = _pad_packed(k[0], kv_lens, max_kv)
    vp, _, _ = _pad_packed(v[0], kv_lens, max_kv)

    kv_valid = (
        torch.arange(max_kv, device=device)[None, :] < kv_lens[:, None]
    )  # [B, max_kv]
    attn_mask = kv_valid[:, None, None, :]  # broadcast to [B, H, max_q, max_kv]

    out = F.scaled_dot_product_attention(
        qp.transpose(1, 2),
        kp.transpose(1, 2),
        vp.transpose(1, 2),
        attn_mask=attn_mask,
        scale=scale,
    )  # [B, H, max_q, Cv]
    out = out.transpose(1, 2)  # [B, max_q, H, Cv]
    packed = out[q_batch, q_pos]  # [Tq, H, Cv]
    return packed.unsqueeze(0)
