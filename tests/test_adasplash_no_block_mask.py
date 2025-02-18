import math

import pytest
import torch
from entmax import entmax_bisect

from adasplash import adasplash_no_block_mask as sparse_attn


def mask_from_varlen(varlen, N_CTX):
    """Create attention mask from variable length tensor"""
    mask = torch.arange(N_CTX, device="cuda")[None, :] < varlen[:, None]
    return mask.unsqueeze(1).unsqueeze(-2)


def get_causal_mask(N_CTX):
    """Generate causal mask for attention"""
    mask = torch.tril(torch.ones(N_CTX, N_CTX, device="cuda", dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(1)


def reference_attention(q, k, v, alpha=1.5, varlen=None, is_causal=False):
    """Reference implementation using entmax_bisect"""
    B, N_H, N_CTX, H_DIM = q.shape
    scale = math.sqrt(H_DIM)

    # Compute attention scores
    qk = torch.matmul(q, k.transpose(-1, -2)) / scale

    # Create seqlen mask
    seq_mask = None
    if varlen is not None:
        seq_mask = mask_from_varlen(varlen, N_CTX)

    # Create causal mask
    causal_mask = None
    if is_causal:
        causal_mask = get_causal_mask(N_CTX)

    # Create qk mask
    qk_mask = None
    if seq_mask is not None and causal_mask is not None:
        qk_mask = seq_mask & causal_mask
    elif seq_mask is not None:
        qk_mask = seq_mask
    elif causal_mask is not None:
        qk_mask = causal_mask

    if qk_mask is not None:
        qk = qk.masked_fill(~qk_mask, float("-inf"))

    # Apply mask and compute attention
    p = entmax_bisect(qk.float(), alpha=alpha).to(q.dtype)
    out = torch.matmul(p, v)

    # Mask output for consistency with the Triton implementation
    if seq_mask is not None:
        out = out.masked_fill(~seq_mask.transpose(-1, -2), 0)

    return out


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_heads", [1, 2])
@pytest.mark.parametrize("seq_len", [128, 256])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("alpha", [1.333, 1.5, 2.0])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_correctness(batch_size, n_heads, seq_len, head_dim, alpha, is_causal, dtype):
    """Test forward pass against reference implementation"""
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float16 else 1e-4

    # Generate random inputs
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device="cuda").contiguous()
    k = torch.randn_like(q).contiguous()
    v = torch.randn_like(q).contiguous()

    # Random variable lengths
    varlen = torch.randint(seq_len // 2, seq_len, (batch_size,), device="cuda")

    # Compute reference
    with torch.no_grad():
        ref_out = reference_attention(q, k, v, alpha=alpha, varlen=varlen, is_causal=is_causal)

    # Compute Triton implementation
    tri_out = sparse_attn(q, k, v, alpha=alpha, varlen=varlen, is_causal=is_causal)

    # Compare results
    assert torch.allclose(tri_out, ref_out, atol=atol)


@pytest.mark.parametrize("seq_len", [256, 512])
@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_backward_gradients(seq_len, alpha):
    """Test gradient correctness using PyTorch's autograd"""
    torch.manual_seed(42)
    dtype = torch.float32  # Use float32 for precise gradient checks
    B, N_H, H_DIM = 2, 2, 64

    # Initialize inputs with requires_grad
    q = torch.randn(B, N_H, seq_len, H_DIM, dtype=dtype, device="cuda", requires_grad=True).contiguous()
    k = torch.randn_like(q, requires_grad=True).contiguous()
    v = torch.randn_like(q, requires_grad=True).contiguous()
    do = torch.randn_like(q).contiguous()

    # Reference implementation
    ref_out = reference_attention(q, k, v, alpha=alpha)
    ref_out.backward(do)
    ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = k.grad = v.grad = None

    # Triton implementation
    tri_out = sparse_attn(q, k, v, alpha=alpha)
    tri_out.backward(do)
    tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # Compare gradients
    assert torch.allclose(tri_dq, ref_dq, atol=1e-2), "q gradients mismatch"
    assert torch.allclose(tri_dk, ref_dk, atol=1e-2), "k gradients mismatch"
    assert torch.allclose(tri_dv, ref_dv, atol=1e-2), "v gradients mismatch"


@pytest.mark.parametrize("seq_len", [512, 1024])
def test_numerical_stability(seq_len):
    """Test for numerical stability and output properties"""
    dtype = torch.float16
    B, N_H, H_DIM = 2, 2, 64
    alpha = 1.5

    # Large input values
    q = torch.randn(B, N_H, seq_len, H_DIM, dtype=dtype, device="cuda") * 100
    k = torch.randn_like(q) * 100
    v = torch.randn_like(q) * 100

    # Forward pass
    out = sparse_attn(q, k, v, alpha=alpha)

    # Check for NaNs/Infs
    assert not torch.isnan(out).any(), "NaNs in output"
    assert not torch.isinf(out).any(), "Infs in output"

    # Check output magnitude
    assert out.abs().max() < 1e3, "Output values too large"


@pytest.mark.parametrize("seq_len", [64, 128])
def test_simple_case(seq_len):
    """Test simple case"""
    # All zeros input
    q = torch.zeros(1, 1, seq_len, 64, device="cuda")
    k = torch.zeros_like(q)
    v = torch.ones_like(q)

    out = sparse_attn(q, k, v, alpha=1.5)
    ref = v.mean(-2).unsqueeze(-2)
    assert torch.allclose(out, ref, atol=1e-4), "All zeros input failed"


def test_variable_length_handling():
    """Test proper handling of variable sequence lengths"""
    B, N_H, N_CTX, H_DIM = 2, 1, 256, 64
    varlen = torch.tensor([128, 64], device="cuda")

    q = torch.randn(B, N_H, N_CTX, H_DIM, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = sparse_attn(q, k, v, varlen=varlen)

    # Check padding is zero
    for i in range(B):
        assert (out[i, :, varlen[i] :] == 0).all(), "Padding not zeroed out"


@pytest.mark.parametrize("max_seqlen", [512, 1024])
def test_large_sequences(max_seqlen):
    """Stress test with large sequence lengths"""
    dtype = torch.float16
    B, N_H, H_DIM = 1, 1, 64

    q = torch.randn(B, N_H, max_seqlen, H_DIM, dtype=dtype, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = sparse_attn(q, k, v, alpha=1.5)

    assert out.shape == (B, N_H, max_seqlen, H_DIM), "Output shape mismatch"
    assert not torch.isnan(out).any(), "NaNs in large sequence output"


if __name__ == "__main__":
    pytest.main(["-v", "-s", "--color=yes"])
