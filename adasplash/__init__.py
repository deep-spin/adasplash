__version__ = "0.2.0"


def adasplash_v1(q, k, v, alpha=1.5, is_causal=False, varlen=None, niter=10):
    from .adasplash_block_mask import sparse_attn

    return sparse_attn(q, k, v, alpha=alpha, is_causal=is_causal, varlen=varlen, niter=niter)


def adasplash_v2(q, k, v, niter=1, varlen=None):
    return _adasplash_v2(q, k, v, niter=niter, varlen=varlen)


def _adasplash_v2(q, k, v, niter=1, varlen=None):
    from .adasplash_v2 import sparse_attn

    return sparse_attn(q, k, v, niter=niter, varlen=varlen)


def adasplash(q, k, v, alpha=1.5, is_causal=True, varlen=None, niter=None):
    """Compatibility dispatcher.

    AdaSplash 0.2 defaults supported causal alpha=1.5 calls to AdaSplash-2.
    Calls requesting v1-only semantics keep using the original block-mask kernel.
    """
    if alpha != 1.5 or not is_causal:
        return adasplash_v1(
            q, k, v, alpha=alpha, is_causal=is_causal, varlen=varlen, niter=10 if niter is None else niter
        )
    return _adasplash_v2(q, k, v, niter=1 if niter is None else niter, varlen=varlen)


def adasplash_no_block_mask(q, k, v, alpha=1.5, is_causal=False, varlen=None, niter=10):
    from .adasplash_no_block_mask import sparse_attn

    return sparse_attn(q, k, v, alpha=alpha, is_causal=is_causal, varlen=varlen, niter=niter)


def triton_entmax_v1(x, alpha=1.5, n_iter=10, fast_math=True):
    from .triton_entmax import triton_entmax

    return triton_entmax(x, alpha=alpha, n_iter=n_iter, fast_math=fast_math)


def triton_entmax_v2(x, alpha=1.5, n_iter=2, use_histogram=True, fast_math=False):
    return _triton_entmax_v2(x, alpha=alpha, n_iter=n_iter, use_histogram=use_histogram, fast_math=fast_math)


def _triton_entmax_v2(x, alpha=1.5, n_iter=2, use_histogram=True, fast_math=False):
    from .triton_entmax_v2 import triton_entmax

    return triton_entmax(x, alpha=alpha, n_iter=n_iter, use_histogram=use_histogram, fast_math=fast_math)


def triton_entmax(x, alpha=1.5, n_iter=2, use_histogram=True, fast_math=False):
    return _triton_entmax_v2(x, alpha=alpha, n_iter=n_iter, use_histogram=use_histogram, fast_math=fast_math)


def triton_sparsemax(x, **kwargs):
    from .triton_entmax_v2 import triton_sparsemax

    return triton_sparsemax(x, **kwargs)


def triton_entmax15(x, **kwargs):
    from .triton_entmax_v2 import triton_entmax15

    return triton_entmax15(x, **kwargs)


adasplash2 = _adasplash_v2

__all__ = [
    "__version__",
    "adasplash",
    "adasplash2",
    "adasplash_v1",
    "adasplash_v2",
    "adasplash_no_block_mask",
    "triton_entmax",
    "triton_entmax_v1",
    "triton_entmax_v2",
    "triton_sparsemax",
    "triton_entmax15",
]
