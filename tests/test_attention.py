import math

import pytest
import torch
from entmax import entmax_bisect

from examples.attention import flash_entmax_attention, slow_entmax_attention


pytestmark = pytest.mark.gpu


def _varlen_mask(varlen, size, padding):
    positions = torch.arange(size, device=varlen.device)
    if padding == "right":
        return positions[None, :] < varlen[:, None]
    return positions[None, :] >= size - varlen[:, None]


def reference_attention(q, k, v, alpha=1.5, varlen=None, is_causal=False, padding="right", alibi_slopes=None):
    _, n_heads, q_len, head_dim = q.shape
    k_len = k.shape[-2]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)

    if alibi_slopes is not None:
        q_pos = torch.arange(q_len, device=q.device)
        k_pos = torch.arange(k_len, device=q.device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        scores = scores + alibi_slopes.view(1, n_heads, 1, 1) * rel_pos.view(1, 1, q_len, k_len)

    if is_causal:
        causal = torch.tril(torch.ones(q_len, k_len, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.view(1, 1, q_len, k_len), float("-inf"))

    output_mask = None
    if varlen is not None:
        key_mask = _varlen_mask(varlen, k_len, padding)
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        if q_len == k_len:
            output_mask = _varlen_mask(varlen, q_len, padding)[:, None, :, None]
            scores = scores.masked_fill(~output_mask, 0.0)

    probs = entmax_bisect(scores.float(), alpha=alpha).to(q.dtype)
    out = torch.matmul(probs, v)
    if output_mask is not None:
        out = out.masked_fill(~output_mask, 0)
    return out


def _run_attention_case(padding, is_causal):
    torch.manual_seed(42)
    q = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    k = torch.randn_like(q, requires_grad=True).contiguous()
    v = torch.randn_like(q, requires_grad=True).contiguous()
    do = torch.randn_like(q)
    varlen = torch.tensor([48], device="cuda", dtype=torch.int32)
    alibi = torch.tensor([0.1, 0.2], device="cuda", dtype=torch.float32)

    ref = reference_attention(q, k, v, varlen=varlen, is_causal=is_causal, padding=padding, alibi_slopes=alibi)
    ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref, (q, k, v), do)

    out = slow_entmax_attention(
        q,
        k,
        v,
        varlen=varlen,
        is_causal=is_causal,
        padding=padding,
        niter=10,
        alibi_slopes=alibi,
    )
    tri_dq, tri_dk, tri_dv = torch.autograd.grad(out, (q, k, v), do)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dq, ref_dq, atol=1e-3, rtol=1e-3)
    assert torch.allclose(tri_dk, ref_dk, atol=1e-3, rtol=1e-3)
    assert torch.allclose(tri_dv, ref_dv, atol=1e-3, rtol=1e-3)


def test_slow_entmax_attention_fast_forward_backward_smoke():
    _run_attention_case(padding="right", is_causal=True)


def test_flash_entmax_attention_example_smoke():
    torch.manual_seed(42)
    q = torch.randn(1, 1, 128, 32, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    k = torch.randn_like(q, requires_grad=True).contiguous()
    v = torch.randn_like(q, requires_grad=True).contiguous()
    do = torch.randn_like(q)

    ref = reference_attention(q, k, v, is_causal=True)
    ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref, (q, k, v), do)

    out = flash_entmax_attention(q, k, v, is_causal=True, niter=10)
    tri_dq, tri_dk, tri_dv = torch.autograd.grad(out, (q, k, v), do)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dq, ref_dq, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dk, ref_dk, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dv, ref_dv, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("padding", ["left", "right"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_slow_entmax_attention_forward_backward_matches_reference(padding, is_causal):
    _run_attention_case(padding=padding, is_causal=is_causal)
