"""Compare forward+backward latency: tl.make_tensor_descriptor (adasplash.py)
vs raw pointer arithmetic (old_adasplash.py). Simplest case: non-varlen,
no GQA, power-of-two N_CTX, fp32, niter=1.

Note: both modules hard-code `DEBUG = True`, which collapses the autotune
space to a single (BLOCK_M=32, BLOCK_N=64, num_warps=2, num_stages=2) config.
For a real perf comparison, edit DEBUG=False in both files before running.
"""

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from adasplash.adasplash import sparse_attn as sparse_attn_new
from adasplash.old_adasplash import sparse_attn as sparse_attn_old


def flash_attn(q, k, v, niter=1):
    """PyTorch SDPA pinned to the FlashAttention backend; causal to match
    adasplash. `niter` is accepted for signature compatibility."""
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def bench(fn, q, k, v, do, label):
    """Run forward and backward through do_bench. Returns (fwd_ms, bwd_ms)."""
    q_in = q.detach().clone().requires_grad_(True)
    k_in = k.detach().clone().requires_grad_(True)
    v_in = v.detach().clone().requires_grad_(True)

    # Forward
    fwd_ms = triton.testing.do_bench(
        lambda: fn(q_in, k_in, v_in, niter=1),
        warmup=50,
        rep=200,
    )

    # Backward: do one forward, then time repeated backward calls on the
    # retained graph so we measure backward kernels only.
    out = fn(q_in, k_in, v_in, niter=1)

    def _bwd():
        q_in.grad = k_in.grad = v_in.grad = None
        out.backward(do, retain_graph=True)

    bwd_ms = triton.testing.do_bench(_bwd, warmup=50, rep=200)

    print(f"{label:>14s}  fwd: {fwd_ms:8.3f} ms   bwd: {bwd_ms:8.3f} ms")
    return fwd_ms, bwd_ms


def main():
    torch.manual_seed(42)
    B, N_H, N_CTX, H_DIM = 16, 16, 16384, 64
    dtype = torch.bfloat16
    device = "cuda"

    print(f"shapes: B={B} N_H={N_H} N_CTX={N_CTX} H_DIM={H_DIM} dtype={dtype}\n")

    q = torch.randn(B, N_H, N_CTX, H_DIM, dtype=dtype, device=device).contiguous()
    k = torch.randn_like(q).contiguous()
    v = torch.randn_like(q).contiguous()
    do = torch.randn_like(q).contiguous()

    fwd_new, bwd_new = bench(sparse_attn_new, q, k, v, do, "new (TMA)")
    fwd_old, bwd_old = bench(sparse_attn_old, q, k, v, do, "old (ptr)")
    fwd_fa, bwd_fa = bench(flash_attn, q, k, v, do, "flash (SDPA)")

    print(
        f"\nspeedup new/old:   "
        f"fwd {fwd_old / fwd_new:5.2f}x   bwd {bwd_old / bwd_new:5.2f}x"
    )
    print(
        f"speedup new/flash: "
        f"fwd {fwd_fa / fwd_new:5.2f}x   bwd {bwd_fa / bwd_new:5.2f}x"
    )


if __name__ == "__main__":
    main()
