import math

import pytest
import torch
from entmax import entmax_bisect

from adasplash import adasplash_v2 as sparse_attn

pytestmark = pytest.mark.gpu


def reference_attention(q, k, v):
    """Reference using entmax_bisect with alpha=1.5, causal masking."""
    B, N_H, N_CTX, H_DIM = q.shape
    scale = math.sqrt(H_DIM)

    qk = torch.matmul(q, k.transpose(-1, -2)) / scale
    causal = torch.tril(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool))
    qk = qk.masked_fill(~causal[None, None], float("-inf"))

    p = entmax_bisect(qk.float(), alpha=1.5).to(q.dtype)
    return torch.matmul(p, v)


def reference_attention_varlen(q, k, v, varlen):
    """Per-batch causal entmax-bisect on the :varlen[b] slice. Padded rows are zero."""
    B, N_H, N_CTX, H_DIM = q.shape
    out = torch.zeros_like(q)
    for b in range(B):
        L = int(varlen[b].item())
        out[b : b + 1, :, :L, :] = reference_attention(
            q[b : b + 1, :, :L, :].contiguous(),
            k[b : b + 1, :, :L, :].contiguous(),
            v[b : b + 1, :, :L, :].contiguous(),
        )
    return out


def test_v2_fast_forward_backward_smoke():
    torch.manual_seed(42)
    q = torch.randn(1, 1, 256, 32, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    k = torch.randn_like(q, requires_grad=True).contiguous()
    v = torch.randn_like(q, requires_grad=True).contiguous()
    do = torch.randn_like(q)

    ref = reference_attention(q, k, v)
    ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref, (q, k, v), do)

    out = sparse_attn(q, k, v, niter=10)
    tri_dq, tri_dk, tri_dv = torch.autograd.grad(out, (q, k, v), do)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dq, ref_dq, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dk, ref_dk, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dv, ref_dv, atol=1e-4, rtol=1e-4)


def test_v2_fast_varlen_gqa_smoke():
    torch.manual_seed(42)
    q = torch.randn(1, 2, 256, 32, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    k = torch.randn(1, 1, 256, 32, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    v = torch.randn_like(k, requires_grad=True).contiguous()
    do = torch.randn_like(q)
    varlen = torch.tensor([160], device="cuda", dtype=torch.int32)

    k_rep = k.repeat_interleave(2, dim=1).contiguous()
    v_rep = v.repeat_interleave(2, dim=1).contiguous()
    ref = reference_attention_varlen(q, k_rep, v_rep, varlen)
    ref_dq, ref_dk_rep, ref_dv_rep = torch.autograd.grad(ref, (q, k_rep, v_rep), do)
    ref_dk = ref_dk_rep.view(1, 1, 2, 256, 32).sum(dim=2)
    ref_dv = ref_dv_rep.view(1, 1, 2, 256, 32).sum(dim=2)

    out = sparse_attn(q, k, v, niter=10, varlen=varlen)
    tri_dq, tri_dk, tri_dv = torch.autograd.grad(out, (q, k, v), do)

    assert torch.allclose(out[:, :, :160], ref[:, :, :160], atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dq[:, :, :160], ref_dq[:, :, :160], atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dk[:, :, :160], ref_dk[:, :, :160], atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dv[:, :, :160], ref_dv[:, :, :160], atol=1e-4, rtol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
def test_forward_matches_reference(seq_len):
    """Forward pass of adasplash_v2 must match entmax_bisect reference in fp32."""
    torch.manual_seed(42)
    B, N_H, H_DIM = 2, 2, 64
    dtype = torch.float32

    q = torch.randn(B, N_H, seq_len, H_DIM, dtype=dtype, device="cuda").contiguous()
    k = torch.randn_like(q).contiguous()
    v = torch.randn_like(q).contiguous()

    with torch.no_grad():
        ref_out = reference_attention(q, k, v)

    tri_out = sparse_attn(q, k, v, niter=10)

    assert torch.allclose(tri_out, ref_out, atol=1e-4), (
        f"forward mismatch at seq_len={seq_len}: " f"max abs diff = {(tri_out - ref_out).abs().max().item():.3e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
def test_backward_matches_reference(seq_len):
    """Backward pass gradients must match entmax_bisect's autograd in fp32."""
    torch.manual_seed(42)
    B, N_H, H_DIM = 2, 2, 64
    dtype = torch.float32

    q = torch.randn(B, N_H, seq_len, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    k = torch.randn_like(q, requires_grad=True).contiguous()
    v = torch.randn_like(q, requires_grad=True).contiguous()
    do = torch.randn_like(q).contiguous()

    # Reference path
    ref_out = reference_attention(q, k, v)
    ref_out.backward(do)
    ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = k.grad = v.grad = None

    # Triton path
    tri_out = sparse_attn(q, k, v, niter=10)
    tri_out.backward(do)
    tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    assert torch.allclose(tri_dq, ref_dq, atol=1e-4), (
        f"dq mismatch at seq_len={seq_len}: " f"max abs diff = {(tri_dq - ref_dq).abs().max().item():.3e}"
    )
    assert torch.allclose(tri_dk, ref_dk, atol=1e-4), (
        f"dk mismatch at seq_len={seq_len}: " f"max abs diff = {(tri_dk - ref_dk).abs().max().item():.3e}"
    )
    assert torch.allclose(tri_dv, ref_dv, atol=1e-4), (
        f"dv mismatch at seq_len={seq_len}: " f"max abs diff = {(tri_dv - ref_dv).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("seq_len", [16384, 32768])
@pytest.mark.slow
def test_forward_matches_reference_long_context(seq_len):
    """Forward pass of adasplash_v2 must match entmax_bisect reference in fp32
    at long contexts (basic causal MHA, no varlen, no GQA)."""
    torch.manual_seed(42)
    B, N_H, H_DIM = 1, 1, 64
    dtype = torch.float32

    q = torch.randn(B, N_H, seq_len, H_DIM, dtype=dtype, device="cuda").contiguous()
    k = torch.randn_like(q).contiguous()
    v = torch.randn_like(q).contiguous()

    with torch.no_grad():
        ref_out = reference_attention(q, k, v)

    tri_out = sparse_attn(q, k, v, niter=10)

    assert torch.allclose(tri_out, ref_out, atol=1e-4), (
        f"forward mismatch at seq_len={seq_len}: " f"max abs diff = {(tri_out - ref_out).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("seq_len", [16384, 32768])
@pytest.mark.slow
def test_backward_matches_reference_long_context(seq_len):
    """Backward pass gradients must match entmax_bisect's autograd in fp32
    at long contexts (basic causal MHA, no varlen, no GQA)."""
    torch.manual_seed(42)
    B, N_H, H_DIM = 1, 1, 64
    dtype = torch.float32

    q = torch.randn(B, N_H, seq_len, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    k = torch.randn_like(q, requires_grad=True).contiguous()
    v = torch.randn_like(q, requires_grad=True).contiguous()
    do = torch.randn_like(q).contiguous()

    # Reference path
    ref_out = reference_attention(q, k, v)
    ref_out.backward(do)
    ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = k.grad = v.grad = None

    # Triton path
    tri_out = sparse_attn(q, k, v, niter=10)
    tri_out.backward(do)
    tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    assert torch.allclose(tri_dq, ref_dq, atol=1e-4), (
        f"dq mismatch at seq_len={seq_len}: " f"max abs diff = {(tri_dq - ref_dq).abs().max().item():.3e}"
    )
    assert torch.allclose(tri_dk, ref_dk, atol=1e-4), (
        f"dk mismatch at seq_len={seq_len}: " f"max abs diff = {(tri_dk - ref_dk).abs().max().item():.3e}"
    )
    assert torch.allclose(tri_dv, ref_dv, atol=1e-4), (
        f"dv mismatch at seq_len={seq_len}: " f"max abs diff = {(tri_dv - ref_dv).abs().max().item():.3e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_ctx,varlen_list",
    [
        (2048, [1000, 1500]),
        (4096, [3333, 2000]),
        (2048, [1024, 1024]),
        (4096, [64, 4096]),
        (1024, [65, 130]),
    ],
)
def test_forward_varlen_matches_reference(n_ctx, varlen_list):
    """Forward with per-batch varlen must match entmax_bisect on the valid rows of each batch."""
    torch.manual_seed(42)
    B = len(varlen_list)
    N_H, H_DIM = 2, 64
    dtype = torch.float32

    q = torch.randn(B, N_H, n_ctx, H_DIM, dtype=dtype, device="cuda").contiguous()
    k = torch.randn_like(q).contiguous()
    v = torch.randn_like(q).contiguous()
    varlen = torch.tensor(varlen_list, dtype=torch.int32, device="cuda")

    with torch.no_grad():
        ref_out = reference_attention_varlen(q, k, v, varlen)

    tri_out = sparse_attn(q, k, v, niter=10, varlen=varlen)

    for b, L in enumerate(varlen_list):
        diff = (tri_out[b, :, :L, :] - ref_out[b, :, :L, :]).abs().max().item()
        assert diff < 1e-4, f"forward mismatch at batch={b}, varlen={L}, n_ctx={n_ctx}: " f"max abs diff = {diff:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_ctx,varlen_list",
    [
        (2048, [1000, 1500]),
        (4096, [3333, 2000]),
        (2048, [1024, 1024]),
        (4096, [64, 4096]),
        (1024, [65, 130]),
    ],
)
def test_backward_varlen_matches_reference(n_ctx, varlen_list):
    """Backward gradients with per-batch varlen must match entmax_bisect's autograd on valid rows."""
    torch.manual_seed(42)
    B = len(varlen_list)
    N_H, H_DIM = 2, 64
    dtype = torch.float32

    q = torch.randn(B, N_H, n_ctx, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    k = torch.randn_like(q, requires_grad=True).contiguous()
    v = torch.randn_like(q, requires_grad=True).contiguous()
    do = torch.randn_like(q).contiguous()
    for b, L in enumerate(varlen_list):
        do[b, :, L:, :] = 0.0
    varlen = torch.tensor(varlen_list, dtype=torch.int32, device="cuda")

    # Reference path
    ref_out = reference_attention_varlen(q, k, v, varlen)
    ref_out.backward(do)
    ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = k.grad = v.grad = None

    # Triton path
    tri_out = sparse_attn(q, k, v, niter=10, varlen=varlen)
    tri_out.backward(do)
    tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    for b, L in enumerate(varlen_list):
        diff_dq = (tri_dq[b, :, :L, :] - ref_dq[b, :, :L, :]).abs().max().item()
        diff_dk = (tri_dk[b, :, :L, :] - ref_dk[b, :, :L, :]).abs().max().item()
        diff_dv = (tri_dv[b, :, :L, :] - ref_dv[b, :, :L, :]).abs().max().item()
        assert diff_dq < 1e-4, f"dq mismatch at batch={b}, varlen={L}: {diff_dq:.3e}"
        assert diff_dk < 1e-4, f"dk mismatch at batch={b}, varlen={L}: {diff_dk:.3e}"
        assert diff_dv < 1e-4, f"dv mismatch at batch={b}, varlen={L}: {diff_dv:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("n_kv_h", [1, 2, 4, 8])
@pytest.mark.parametrize("seq_len", [1024, 2048])
def test_forward_gqa_matches_reference(seq_len, n_kv_h):
    """Forward GQA must match entmax_bisect reference with replicated K/V."""
    torch.manual_seed(42)
    B, N_H, H_DIM = 2, 8, 64
    assert N_H % n_kv_h == 0
    group_size = N_H // n_kv_h
    dtype = torch.float32

    q = torch.randn(B, N_H, seq_len, H_DIM, dtype=dtype, device="cuda").contiguous()
    k = torch.randn(B, n_kv_h, seq_len, H_DIM, dtype=dtype, device="cuda").contiguous()
    v = torch.randn(B, n_kv_h, seq_len, H_DIM, dtype=dtype, device="cuda").contiguous()

    with torch.no_grad():
        k_rep = k.repeat_interleave(group_size, dim=1).contiguous()
        v_rep = v.repeat_interleave(group_size, dim=1).contiguous()
        ref_out = reference_attention(q, k_rep, v_rep)

    tri_out = sparse_attn(q, k, v, niter=10)

    diff = (tri_out - ref_out).abs().max().item()
    assert diff < 1e-4, f"forward GQA mismatch at seq_len={seq_len}, n_kv_h={n_kv_h}: " f"max abs diff = {diff:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("n_kv_h", [1, 2, 4, 8])
@pytest.mark.parametrize("seq_len", [1024, 2048])
def test_backward_gqa_matches_reference(seq_len, n_kv_h):
    """Backward GQA gradients must match the reference, with dk/dv reduced
    across each group of Q heads sharing a KV head."""
    torch.manual_seed(42)
    B, N_H, H_DIM = 2, 8, 64
    assert N_H % n_kv_h == 0
    group_size = N_H // n_kv_h
    dtype = torch.float32

    q = torch.randn(B, N_H, seq_len, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    k = torch.randn(B, n_kv_h, seq_len, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    v = torch.randn(B, n_kv_h, seq_len, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    do = torch.randn_like(q).contiguous()

    # Reference
    k_rep = k.repeat_interleave(group_size, dim=1)
    v_rep = v.repeat_interleave(group_size, dim=1)
    ref_out = reference_attention(q, k_rep, v_rep)
    ref_dq, ref_dk_rep, ref_dv_rep = torch.autograd.grad(ref_out, (q, k_rep, v_rep), do)
    ref_dk = ref_dk_rep.view(B, n_kv_h, group_size, seq_len, H_DIM).sum(dim=2)
    ref_dv = ref_dv_rep.view(B, n_kv_h, group_size, seq_len, H_DIM).sum(dim=2)

    # Triton path
    tri_out = sparse_attn(q, k, v, niter=10)
    tri_dq, tri_dk, tri_dv = torch.autograd.grad(tri_out, (q, k, v), do)

    for name, t, r in (("dq", tri_dq, ref_dq), ("dk", tri_dk, ref_dk), ("dv", tri_dv, ref_dv)):
        diff = (t - r).abs().max().item()
        assert diff < 1e-4, f"{name} GQA mismatch at seq_len={seq_len}, n_kv_h={n_kv_h}: " f"max abs diff = {diff:.3e}"


def _gqa_varlen_reference(q, k, v, varlen, group_size):
    """Per-batch causal entmax-bisect with K/V replicated across the group."""
    B, N_H, N_CTX, H_DIM = q.shape
    out = torch.zeros_like(q)
    for b in range(B):
        L = int(varlen[b].item())
        kb = k[b : b + 1, :, :L, :].repeat_interleave(group_size, dim=1).contiguous()
        vb = v[b : b + 1, :, :L, :].repeat_interleave(group_size, dim=1).contiguous()
        out[b : b + 1, :, :L, :] = reference_attention(
            q[b : b + 1, :, :L, :].contiguous(),
            kb,
            vb,
        )
    return out


@pytest.mark.slow
@pytest.mark.parametrize("n_kv_h", [1, 2, 4])
@pytest.mark.parametrize(
    "n_ctx,varlen_list",
    [
        (2048, [1000, 1500]),
        (1024, [65, 130]),
    ],
)
def test_gqa_varlen_matches_reference(n_ctx, varlen_list, n_kv_h):
    """Forward+backward GQA with per-batch varlen must match the replicated reference."""
    torch.manual_seed(42)
    B = len(varlen_list)
    N_H, H_DIM = 8, 64
    assert N_H % n_kv_h == 0
    group_size = N_H // n_kv_h
    dtype = torch.float32

    q = torch.randn(B, N_H, n_ctx, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    k = torch.randn(B, n_kv_h, n_ctx, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    v = torch.randn(B, n_kv_h, n_ctx, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    do = torch.randn_like(q).contiguous()
    for b, L in enumerate(varlen_list):
        do[b, :, L:, :] = 0.0
    varlen = torch.tensor(varlen_list, dtype=torch.int32, device="cuda")

    # Reference forward (varlen) — used for the forward check.
    with torch.no_grad():
        ref_out = _gqa_varlen_reference(q, k, v, varlen, group_size)

    # Triton forward
    tri_out = sparse_attn(q, k, v, niter=10, varlen=varlen)
    for b, L in enumerate(varlen_list):
        diff = (tri_out[b, :, :L, :] - ref_out[b, :, :L, :]).abs().max().item()
        assert diff < 1e-4, (
            f"forward GQA varlen mismatch at batch={b}, varlen={L}, "
            f"n_ctx={n_ctx}, n_kv_h={n_kv_h}: max abs diff = {diff:.3e}"
        )

    k_rep = k.repeat_interleave(group_size, dim=1)
    v_rep = v.repeat_interleave(group_size, dim=1)
    ref_out_full = reference_attention_varlen(q, k_rep, v_rep, varlen)
    ref_dq, ref_dk_rep, ref_dv_rep = torch.autograd.grad(ref_out_full, (q, k_rep, v_rep), do)
    ref_dk = ref_dk_rep.view(B, n_kv_h, group_size, n_ctx, H_DIM).sum(dim=2)
    ref_dv = ref_dv_rep.view(B, n_kv_h, group_size, n_ctx, H_DIM).sum(dim=2)

    # Triton backward
    tri_dq, tri_dk, tri_dv = torch.autograd.grad(tri_out, (q, k, v), do)

    for b, L in enumerate(varlen_list):
        diff_dq = (tri_dq[b, :, :L, :] - ref_dq[b, :, :L, :]).abs().max().item()
        diff_dk = (tri_dk[b, :, :L, :] - ref_dk[b, :, :L, :]).abs().max().item()
        diff_dv = (tri_dv[b, :, :L, :] - ref_dv[b, :, :L, :]).abs().max().item()
        assert diff_dq < 1e-4, f"dq GQA varlen mismatch at batch={b}, varlen={L}, n_kv_h={n_kv_h}: {diff_dq:.3e}"
        assert diff_dk < 1e-4, f"dk GQA varlen mismatch at batch={b}, varlen={L}, n_kv_h={n_kv_h}: {diff_dk:.3e}"
        assert diff_dv < 1e-4, f"dv GQA varlen mismatch at batch={b}, varlen={L}, n_kv_h={n_kv_h}: {diff_dv:.3e}"


if __name__ == "__main__":
    pytest.main(["-v", "-s", "--color=yes"])
