import math

import torch


def _varlen_mask(varlen, size, padding):
    positions = torch.arange(size, device=varlen.device)
    if padding == "right":
        return positions[None, :] < varlen[:, None]
    if padding == "left":
        return positions[None, :] >= size - varlen[:, None]
    raise ValueError("padding must be either 'right' or 'left'.")


def _alibi_bias(q, k, alibi_slopes):
    _, n_heads, q_len, _ = q.shape
    k_len = k.shape[-2]
    if alibi_slopes.shape != (n_heads,):
        raise ValueError(f"alibi_slopes must have shape ({n_heads},); got {tuple(alibi_slopes.shape)}.")

    if q_len == 1 and k_len > 1:
        rel_pos = torch.arange(k_len, device=q.device) - (k_len - 1)
        rel_pos = rel_pos.view(1, 1, 1, k_len)
    else:
        q_pos = torch.arange(q_len, device=q.device)
        k_pos = torch.arange(k_len, device=q.device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rel_pos = rel_pos.view(1, 1, q_len, k_len)
    return alibi_slopes.to(q.device).view(1, n_heads, 1, 1) * rel_pos


def entmax_attention(q, k, v, alpha=1.5, varlen=None, is_causal=False, padding="right", niter=2, alibi_slopes=None):
    """Dense QK attention using the public v2 Triton entmax activation."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k and v must have shape (batch, heads, seq_len, head_dim).")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape; got {tuple(k.shape)} and {tuple(v.shape)}.")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError("q, k and v must agree on batch, heads and head_dim.")

    _, _, q_len, head_dim = q.shape
    k_len = k.shape[-2]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)

    if alibi_slopes is not None:
        scores = scores + _alibi_bias(q, k, alibi_slopes)

    if is_causal:
        if q_len == k_len:
            causal = torch.tril(torch.ones(q_len, k_len, device=q.device, dtype=torch.bool))
        else:
            q_pos = torch.arange(q_len, device=q.device) + (k_len - q_len)
            k_pos = torch.arange(k_len, device=q.device)
            causal = q_pos[:, None] >= k_pos[None, :]
        scores = scores.masked_fill(~causal.view(1, 1, q_len, k_len), float("-inf"))

    output_mask = None
    if varlen is not None:
        if varlen.dim() != 1 or varlen.shape[0] != q.shape[0]:
            raise ValueError(f"varlen must be a 1-D tensor of shape ({q.shape[0]},).")
        key_mask = _varlen_mask(varlen.to(q.device), k_len, padding)
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        if q_len == k_len:
            output_mask = _varlen_mask(varlen.to(q.device), q_len, padding)[:, None, :, None]

    from .triton_entmax_v2 import triton_entmax

    probs = triton_entmax(scores.contiguous(), alpha=alpha, n_iter=niter, fast_math=False)
    out = torch.matmul(probs, v)
    if output_mask is not None:
        out = out.masked_fill(~output_mask, 0)
    return out
