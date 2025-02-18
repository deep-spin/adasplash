import torch

import triton
import triton.language as tl
from triton.language.extra.libdevice import fast_powf


def get_configs():
    """Generate Triton configurations for autotuning.

    Returns:
        list: List of Triton Config objects with varying block sizes and warp counts.
    """
    return [
        triton.Config({"BLOCK_N": bs}, num_warps=nw) for bs in [32, 64, 128, 256, 512, 1024] for nw in [1, 2, 4, 8, 16]
    ]


@triton.jit
def _masked_pow(x, x_mask, coeff, FAST_MATH: tl.constexpr):
    """Compute masked power function using either fast approximation or precise method.

    Args:
        x: Input tensor
        x_mask: Boolean mask for valid elements
        coeff: Exponent coefficient
        FAST_MATH: Whether to use fast math approximations

    Returns:
        tl.tensor: Element-wise x^coeff where mask is True, 0 otherwise
    """
    if FAST_MATH:
        return tl.where(x_mask, fast_powf(x, coeff), 0)
    else:
        return tl.where(x_mask, tl.exp2(x * coeff), 0)


@triton.autotune(configs=get_configs(), key=["SIZE_N", "FAST_MATH", "N_ITER"])
@triton.jit
def _fwd_entmax(
    X,
    Y,
    ALPHA,
    N_ITER: tl.constexpr,
    SIZE_N: tl.constexpr,
    FAST_MATH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Triton kernel for forward pass of entmax transformation.

    Implements the entmax transformation using an iterative Newton method with Halley updates.

    Args:
        X: Input tensor pointer
        Y: Output tensor pointer
        ALPHA: Entmax alpha parameter (> 1)
        N_ITER: Number of optimization iterations
        SIZE_N: Size of the last dimension
        FAST_MATH: Enable fast math optimizations
        BLOCK_N: Block size for processing
    """
    # Constants initialization
    EPS: tl.constexpr = 1e-6  # Small epsilon for numerical stability
    _scalar = ALPHA - 1
    coeff_0 = 1 / (ALPHA - 1)  # First order coefficient
    coeff_1 = coeff_0 - 1  # Second order coefficient
    coeff_2 = coeff_1 - 1  # Third order coefficient

    # Program ID and pointer setup
    off_m = tl.program_id(0) * SIZE_N
    X += off_m
    Y += off_m

    # Initial max value computation for tau approximation
    max_val = float("-inf")
    for curr_n in range(0, SIZE_N, BLOCK_N):
        curr_offsets = curr_n + tl.arange(0, BLOCK_N)
        load_mask = curr_offsets < SIZE_N
        x = tl.load(X + curr_offsets, mask=load_mask, other=float("-inf"))
        max_val = tl.maximum(max_val, tl.max(x))

    # Initialize tau bounds and values
    max_val *= _scalar
    t = max_val - 0.5  # Initial tau estimate
    t_hi = max_val  # Upper bound
    t_lo = max_val - 1  # Lower bound

    # Halley's method iterations
    for _ in range(N_ITER):
        acc_0 = tl.zeros((BLOCK_N,), dtype=tl.float32)  # Function accumulator
        acc_1 = tl.zeros((BLOCK_N,), dtype=tl.float32)  # First derivative accumulator
        acc_2 = tl.zeros((BLOCK_N,), dtype=tl.float32)  # Second derivative accumulator

        # Accumulate statistics across the tensor
        for curr_n in range(0, SIZE_N, BLOCK_N):
            curr_offsets = curr_n + tl.arange(0, BLOCK_N)
            load_mask = curr_offsets < SIZE_N
            x = tl.load(X + curr_offsets, mask=load_mask, other=float("-inf")) * _scalar

            x_mask = (x > t) & load_mask
            x_act = x - t
            if not FAST_MATH:
                x_act = tl.log2(tl.maximum(x_act, EPS))

            acc_0 += _masked_pow(x_act, x_mask, coeff_0, FAST_MATH)
            acc_1 += _masked_pow(x_act, x_mask, coeff_1, FAST_MATH)
            acc_2 += _masked_pow(x_act, x_mask, coeff_2, FAST_MATH)

        # Compute function value and derivatives
        ff = tl.sum(acc_0) - 1.0  # Function value
        df = -coeff_0 * tl.sum(acc_1)  # First derivative
        ddf = coeff_0 * coeff_1 * tl.sum(acc_2)  # Second derivative

        # Halley's method update
        new_t = t - (2 * ff * df) / (2 * df * df - ff * ddf)

        # Update tau bounds
        t_lo = tl.where(ff > 0, t, t_lo)
        t_hi = tl.where(ff < 0, t, t_hi)

        # Maintain numerical stability
        is_good = (new_t > t_lo - EPS) & (new_t < t_hi + EPS)
        t = tl.where(is_good, new_t, 0.5 * (t_lo + t_hi))

    # Final output computation
    for curr_n in range(0, SIZE_N, BLOCK_N):
        curr_offsets = curr_n + tl.arange(0, BLOCK_N)
        load_mask = curr_offsets < SIZE_N
        x = tl.load(X + curr_offsets, mask=load_mask, other=float("-inf")) * _scalar

        x_mask = (x > t) & load_mask
        x_act = x - t
        if not FAST_MATH:
            x_act = tl.log2(tl.maximum(x_act, EPS))

        y = _masked_pow(x_act, x_mask, coeff_0, FAST_MATH)
        tl.store(Y + curr_offsets, y, mask=load_mask)


@triton.autotune(configs=get_configs(), key=["SIZE_N", "FAST_MATH"])
@triton.jit
def _bwd_entmax(
    Y,
    DY,
    DX,
    ALPHA,
    SIZE_N: tl.constexpr,
    FAST_MATH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Triton kernel for backward pass of entmax transformation.

    Computes gradients using the chain rule and precomputed outputs from the forward pass.

    Args:
        Y: Output tensor from forward pass
        DY: Gradient of output
        DX: Gradient of input (to be computed)
        ALPHA: Entmax alpha parameter (> 1)
        SIZE_N: Size of the last dimension
        FAST_MATH: Enable fast math optimizations
        BLOCK_N: Block size for processing
    """
    # Backward pass constants
    coeff_f = 2 - ALPHA  # Coefficient for gradient calculation
    EPS: tl.constexpr = 1e-6

    # Program ID and pointer setup
    off_m = tl.program_id(0) * SIZE_N
    Y += off_m
    DY += off_m
    DX += off_m

    # First pass: compute scalar term for gradient calculation
    uDy_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    y_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for curr_n in range(0, SIZE_N, BLOCK_N):
        curr_offsets = curr_n + tl.arange(0, BLOCK_N)
        load_mask = curr_offsets < SIZE_N

        y = tl.load(Y + curr_offsets, mask=load_mask, other=0)
        dy = tl.load(DY + curr_offsets, mask=load_mask, other=0)

        u_mask = (y > 0) & load_mask
        if not FAST_MATH:
            y = tl.log2(tl.maximum(y, EPS))

        u = _masked_pow(y, u_mask, coeff_f, FAST_MATH)

        uDy_sum += u * dy
        y_sum += u

    # Compute normalization scalar
    scalar = tl.sum(uDy_sum) / tl.sum(y_sum)

    # Second pass: compute and store final gradients
    for curr_n in range(0, SIZE_N, BLOCK_N):
        curr_offsets = curr_n + tl.arange(0, BLOCK_N)
        load_mask = curr_offsets < SIZE_N

        y = tl.load(Y + curr_offsets, mask=load_mask, other=0)
        dy = tl.load(DY + curr_offsets, mask=load_mask, other=0)

        u_mask = (y > 0) & load_mask
        if not FAST_MATH:
            y = tl.log2(tl.maximum(y, EPS))

        u = _masked_pow(y, u_mask, coeff_f, FAST_MATH)
        grad = u * dy - scalar * u
        tl.store(DX + curr_offsets, grad, mask=load_mask)


class _entmax_triton(torch.autograd.Function):
    """PyTorch autograd Function wrapper for entmax transformation."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        alpha: float = 1.5,
        n_iter: int = 10,
        fast_math: bool = False,
    ):
        """Forward pass of entmax transformation.

        Args:
            x: Input tensor
            alpha: Entmax alpha parameter (> 1)
            n_iter: Number of optimization iterations
            fast_math: Use fast math approximations

        Returns:
            torch.Tensor: entmax transformed output
        """

        # Input validation and preparation
        SIZE_N = x.shape[-1]
        assert x.is_contiguous(), "x must be contiguous"

        ## -- constants --
        # BLOCK_N = 512
        # Either uncomment the above line or use autotune to find the best BLOCK_N!

        # Output initialization
        y = torch.zeros_like(x).contiguous()

        # Define grid size for Triton kernel launch
        grid = (x.shape[:-1].numel(), 1, 1)

        # Kernel launch
        _fwd_entmax[grid](
            x,
            y,
            ##
            alpha,
            n_iter,
            SIZE_N,
            fast_math,
        )

        # Save context for backward pass
        ctx.save_for_backward(y)
        ctx.alpha = alpha
        ctx.fast_math = fast_math

        return y

    @staticmethod
    def backward(ctx, dy):
        """Backward pass of entmax transformation.

        Args:
            ctx: Context object with saved forward pass information
            dy: Gradient of the output

        Returns:
            tuple: Gradient of the input and None for non-tensor arguments
        """

        # Retrieve saved tensors and parameters
        y = ctx.saved_tensors[0]
        SIZE_N = y.shape[-1]

        # Make sure gradient is contiguous
        if not dy.is_contiguous():
            dy = dy.contiguous()

        # Gradient initialization
        dx = torch.zeros_like(dy).contiguous()

        # Define grid size for Triton kernel launch
        grid = (y.shape[:-1].numel(), 1, 1)

        # Kernel launch
        _bwd_entmax[grid](
            y,
            dy,
            dx,
            ctx.alpha,
            SIZE_N,
            ctx.fast_math,
        )

        return dx, None, None, None


def triton_entmax(x: torch.Tensor, alpha: float = 1.5, n_iter: int = 10, fast_math: bool = True) -> torch.Tensor:
    """Entmax activation function with Triton acceleration.

    This function applies the Entmax transformation along the last dimension of the input tensor.

    Args:
        x: Input tensor
        alpha: Entmax alpha parameter (> 1)
        n_iter: Number of optimization iterations
        fast_math: Use fast math approximations for better performance

    Returns:
        torch.Tensor: entmax transformed output with the same shape as `x`

    Example:
        >>> x = torch.randn(128, 256).cuda()
        >>> y = triton_entmax(x, alpha=1.5)
        >>> torch.allclose(y.sum(-1), torch.ones(128).cuda())
    """
    return _entmax_triton.apply(x, alpha, n_iter, fast_math)
