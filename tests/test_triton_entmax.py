import pytest
import torch
from entmax import entmax_bisect
from torch.autograd import gradcheck

import adasplash


pytestmark = pytest.mark.gpu


def _assert_close(actual, expected, *, atol, rtol, label):
    diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), f"{label}: max abs diff = {diff:.3e}"


def _forward_tolerances(dtype, alpha):
    if dtype == torch.float16:
        return (2e-2, 2e-2) if alpha == 1.333 else (1e-2, 1e-2)
    return (5e-4, 5e-4) if alpha == 1.333 else (1e-4, 1e-4)


def _call_entmax(impl_name, x, alpha=1.5, n_iter=10, fast_math=False, use_histogram=True):
    impl = getattr(adasplash, impl_name)
    if impl_name == "triton_entmax_v1":
        return impl(x, alpha=alpha, n_iter=n_iter, fast_math=fast_math)
    return impl(x, alpha=alpha, n_iter=n_iter, use_histogram=use_histogram, fast_math=fast_math)


@pytest.mark.parametrize("impl_name", ["triton_entmax_v1", "triton_entmax_v2", "triton_entmax"])
@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_entmax_fast_forward_backward_smoke(impl_name, alpha):
    torch.manual_seed(42)
    x = torch.randn(2, 16, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    do = torch.randn_like(x)

    ref = entmax_bisect(x, alpha=alpha)
    ref_dx = torch.autograd.grad(ref, x, do, retain_graph=False)[0]

    out = _call_entmax(impl_name, x, alpha=alpha)
    tri_dx = torch.autograd.grad(out, x, do, retain_graph=False)[0]

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tri_dx, ref_dx, atol=1e-2, rtol=1e-2)


@pytest.mark.slow
@pytest.mark.parametrize("impl_name", ["triton_entmax_v1", "triton_entmax_v2", "triton_entmax"])
@pytest.mark.parametrize("shape", [(2,), (2, 4), (2, 4, 8), (2, 4, 8, 16)])
@pytest.mark.parametrize("alpha", [1.333, 1.5, 2.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("fast_math", [False, True])
def test_entmax_forward_matches_reference(impl_name, shape, alpha, dtype, fast_math):
    torch.manual_seed(42)
    atol, rtol = _forward_tolerances(dtype, alpha)

    x = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    with torch.no_grad():
        ref = entmax_bisect(x.float(), alpha=alpha).to(dtype)

    out = _call_entmax(impl_name, x, alpha=alpha, fast_math=fast_math)

    _assert_close(out, ref, atol=atol, rtol=rtol, label=f"{impl_name} forward alpha={alpha}")
    assert torch.allclose(out.sum(-1), torch.ones_like(out.sum(-1)), atol=1e-2)


@pytest.mark.slow
@pytest.mark.parametrize("impl_name", ["triton_entmax_v1", "triton_entmax_v2", "triton_entmax"])
@pytest.mark.parametrize("shape", [(2, 4), (2, 4, 8)])
@pytest.mark.parametrize("alpha", [1.333, 1.5, 2.0])
def test_entmax_backward_matches_reference(impl_name, shape, alpha):
    torch.manual_seed(42)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    do = torch.randn_like(x)

    ref = entmax_bisect(x, alpha=alpha)
    ref_dx = torch.autograd.grad(ref, x, do, retain_graph=False)[0]

    out = _call_entmax(impl_name, x, alpha=alpha)
    tri_dx = torch.autograd.grad(out, x, do, retain_graph=False)[0]

    atol = 5e-2 if alpha == 1.333 else 1e-2
    _assert_close(tri_dx, ref_dx, atol=atol, rtol=atol, label=f"{impl_name} backward alpha={alpha}")


@pytest.mark.slow
@pytest.mark.parametrize("impl_name", ["triton_entmax_v1", "triton_entmax_v2"])
def test_entmax_gradcheck_small(impl_name):
    pytest.skip("Triton entmax kernels do not currently compile the fp64 path required by torch.gradcheck.")
    torch.manual_seed(42)
    impl = getattr(adasplash, impl_name)
    x = torch.randn(2, 4, device="cuda", dtype=torch.float64, requires_grad=True).contiguous()

    if impl_name == "triton_entmax_v1":
        fn = lambda z: impl(z, alpha=1.5, n_iter=10, fast_math=False)
    else:
        fn = lambda z: impl(z, alpha=1.5, n_iter=10, use_histogram=True, fast_math=False)
    assert gradcheck(fn, x, atol=1e-2, eps=1e-4)


@pytest.mark.slow
def test_v2_histogram_toggle_and_convenience_aliases():
    torch.manual_seed(42)
    x = torch.randn(4, 16, device="cuda", dtype=torch.float32).contiguous()

    out_hist = adasplash.triton_entmax_v2(x, alpha=1.5, use_histogram=True)
    out_no_hist = adasplash.triton_entmax_v2(x, alpha=1.5, use_histogram=False, n_iter=10)
    ref_15 = entmax_bisect(x, alpha=1.5)
    ref_sparse = entmax_bisect(x, alpha=2.0)

    assert torch.allclose(out_hist, ref_15, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_no_hist, ref_15, atol=1e-4, rtol=1e-4)
    assert torch.allclose(adasplash.triton_entmax15(x), ref_15, atol=1e-4, rtol=1e-4)
    assert torch.allclose(adasplash.triton_sparsemax(x), ref_sparse, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_entmax_numerical_stability():
    x = (torch.randn(2, 32, device="cuda") * 100).contiguous().requires_grad_(True)
    y = adasplash.triton_entmax(x, alpha=1.5, n_iter=10)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert (y >= 0).all()
    y.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
