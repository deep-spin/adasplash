import pytest
import torch
from entmax import entmax_bisect  # Requires entmax package installation
from torch.autograd import gradcheck

from adasplash.triton_entmax import triton_entmax


@pytest.mark.parametrize("shape", [(2,), (2, 4), (2, 4, 8), (2, 4, 8, 16)])  # 1D  # 2D  # 3D  # 4D
@pytest.mark.parametrize("alpha", [1.1, 1.333, 1.5, 1.666, 2.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("fast_math", [False, True])
def test_forward_correctness(shape, alpha, dtype, fast_math):
    """Test forward pass against reference entmax implementation"""
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float16 else 1e-4

    # Generate test data
    x = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    x_ref = x.detach().clone().contiguous().requires_grad_(False)

    # Compute reference
    with torch.no_grad():
        ref = entmax_bisect(x_ref, alpha=alpha)

    # Compute Triton implementation
    triton_out = triton_entmax(x, alpha=alpha, fast_math=fast_math)

    # Compare results
    assert torch.allclose(triton_out.cpu(), ref.cpu(), atol=atol)


@pytest.mark.parametrize("shape", [(2,), (2, 4), (2, 4, 8)])
@pytest.mark.parametrize("alpha", [1.1, 1.333, 1.5, 1.666, 2.0])
def test_backward_gradients(shape, alpha):
    """Test backward pass using PyTorch's gradcheck"""
    torch.manual_seed(42)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32, requires_grad=True)

    # Use double precision for gradcheck
    assert gradcheck(lambda x: triton_entmax(x, alpha, 10, True), x, atol=1e-2, eps=1e-4), "Gradcheck failed"


@pytest.mark.parametrize("alpha", [1.1, 1.333, 1.5, 1.666, 2.0])
def test_numerical_stability(alpha):
    """Test for NaNs, infinities, and extreme values"""
    shapes = [(2,), (2, 5), (2, 4, 8)]
    for shape in shapes:
        x = torch.randn(*shape, device="cuda").contiguous() * 100  # Large values
        x.requires_grad = True

        # Forward pass
        y = triton_entmax(x, alpha=alpha)
        assert not torch.isnan(y).any(), "NaNs in output"
        assert not torch.isinf(y).any(), "Infs in output"
        assert (y >= 0).all(), "Negative values in output"
        assert torch.allclose(y.sum(-1), torch.ones(y.shape[:-1], device="cuda"), atol=1e-2), "Output doesn't sum to 1"

        # Backward pass
        loss = y.sum()
        loss.backward()
        assert not torch.isnan(x.grad).any(), "NaNs in gradients"


@pytest.mark.parametrize("alpha", [1.333, 1.5, 1.666, 2.0])
def test_edge_cases(alpha):
    """Test edge cases like all zeros, large inputs, etc."""
    # All zeros input
    x = torch.zeros(64, device="cuda")
    y = triton_entmax(x, alpha=alpha)
    y_ref = torch.ones_like(y) / y.shape[-1]
    assert torch.allclose(y, y_ref, atol=1e-2), "All zeros input failed"

    # All equal large values
    x = torch.full((64,), 1e3, device="cuda")
    y = triton_entmax(x, alpha=alpha)
    y_ref = torch.ones_like(y) / y.shape[-1]
    assert torch.allclose(y, y_ref, atol=1e-2), "Large equal values failed"

    # One-hot input
    x = torch.zeros(64, device="cuda")
    x[0] = 100
    y = triton_entmax(x, alpha=alpha)
    y_ref = torch.eye(y.shape[-1], device="cuda")[0]
    assert torch.allclose(y, y_ref, atol=1e-2), "One-hot input failed"


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_different_dimensions(dim):
    """Test that kernel works across different dimensions"""
    shape = [32, 32, 32]
    x = torch.randn(*shape, device="cuda")

    # Reference implementation along specified dimension
    ref = entmax_bisect(x.cpu().float(), alpha=1.5, dim=dim)

    # Triton implementation
    if dim == 0:
        x = x.permute((1, 2, 0)).contiguous()
        triton_out = triton_entmax(x, alpha=1.5)
        triton_out = triton_out.permute((2, 0, 1))

    elif dim == 1:
        x = x.permute((0, 2, 1)).contiguous()
        triton_out = triton_entmax(x, alpha=1.5)
        triton_out = triton_out.permute((0, 2, 1))

    else:
        triton_out = triton_entmax(x, alpha=1.5)

    # Compare results
    assert torch.allclose(triton_out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-5), f"Dimension {dim} test failed"


@pytest.mark.parametrize("fast_math", [False, True])
def test_fast_math_consistency(fast_math):
    """Test that fast_math flag produces consistent results"""
    x = torch.randn(4, 8, device="cuda")
    y1 = triton_entmax(x, fast_math=fast_math)
    y2 = triton_entmax(x, fast_math=fast_math)  # Second run

    # Check consistency between runs
    assert torch.allclose(y1, y2, atol=1e-5), "Fast math results inconsistent between runs"


def test_autograd_function():
    """Test integration with PyTorch autograd"""
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    y = triton_entmax(x, alpha=1.5)

    # Test backward
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Gradients not computed"
    assert not torch.isnan(x.grad).any(), "NaNs in gradients"
    assert x.grad.abs().max() < 1e3, "Exploding gradients"


if __name__ == "__main__":
    pytest.main(["-v", "-s", "--color=yes"])
