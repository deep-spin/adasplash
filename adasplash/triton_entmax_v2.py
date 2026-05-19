import os

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import fast_powf, float2int_rd, isinf


def _autotune(*args, **kwargs):
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return lambda fn: fn
    return triton.autotune(*args, **kwargs)


def get_configs():
    """Generate Triton configurations for autotuning."""
    if os.environ.get("ADASPLASH_TEST_FAST_AUTOTUNE", "0") == "1":
        return [triton.Config({"BLOCK_N": 32}, num_warps=2)]
    return [triton.Config({"BLOCK_N": bs}, num_warps=nw) for bs in [32, 64, 128, 256, 512] for nw in [2, 4, 8]]


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
    EPS: tl.constexpr = 1e-6
    if FAST_MATH:
        return tl.where(x_mask, fast_powf(x, coeff), 0)
    else:
        return tl.where(x_mask, tl.exp2(tl.log2(tl.maximum(x, EPS)) * coeff), 0)


@triton.jit
def hybrid_update(
    t,
    t_lo,
    t_hi,
    t_prev,
    f_prev,  # for secant
    acc_0,
    acc_1,
    acc_2,
    coeff_0,
    coeff_1,
    alpha: tl.constexpr,
    allow_secant: tl.constexpr,
):
    """Hybrid tau update with Halley, Newton, secant, then bisection fallback."""
    EPS: tl.constexpr = 1e-6
    EPS_DENOM: tl.constexpr = 1e-12

    # Function value
    ff = acc_0 - 1.0
    # First derivative (needed for both Halley and Newton)
    df = -coeff_0 * acc_1
    # Second derivative (needed for Halley)
    ddf = coeff_0 * coeff_1 * acc_2

    # Update bounds
    t_lo = tl.where((ff > 0), t, t_lo)
    t_hi = tl.where((ff < 0), t, t_hi)

    # Try halley first
    halley_num = 2 * ff * df
    halley_denom = 2 * df * df - ff * ddf
    t_halley = t - halley_num / halley_denom
    halley_in_bounds = (t_halley > t_lo - EPS) & (t_halley < t_hi + EPS)
    halley_finite = ~isinf(t_halley)
    halley_denom_ok = tl.abs(halley_denom) > EPS_DENOM
    halley_valid = halley_in_bounds & halley_finite & halley_denom_ok & (alpha <= 1.5)

    t_candidate = tl.where(halley_valid, t_halley, 0.0)
    has_candidate = halley_valid

    # Try Newton.
    newton_denom = df
    t_newton = t - ff / newton_denom
    newton_in_bounds = (t_newton > t_lo - EPS) & (t_newton < t_hi + EPS)
    newton_finite = ~isinf(t_newton)
    newton_denom_ok = tl.abs(newton_denom) > EPS_DENOM
    newton_valid = newton_in_bounds & newton_finite & newton_denom_ok & (alpha <= 2.0)

    newton_take = newton_valid & ~has_candidate
    t_candidate = tl.where(newton_take, t_newton, t_candidate)
    has_candidate = has_candidate | newton_take

    if allow_secant:
        secant_num = ff * (t - t_prev)
        secant_denom = ff - f_prev
        t_secant = t - secant_num / secant_denom
        secant_in_bounds = (t_secant > t_lo - EPS) & (t_secant < t_hi + EPS)
        secant_finite = ~isinf(t_secant)
        secant_denom_ok = tl.abs(secant_denom) > EPS_DENOM
        secant_valid = secant_in_bounds & secant_finite & secant_denom_ok

        secant_take = secant_valid & ~has_candidate
        t_candidate = tl.where(secant_take, t_secant, t_candidate)
        has_candidate = has_candidate | secant_take

    t_bisect = 0.5 * (t_lo + t_hi)
    return tl.where(has_candidate, t_candidate, t_bisect), t_lo, t_hi


@triton.jit
def hist_init_tau_alpha2(hist, t_init, BINS: tl.constexpr, BITS_PER_BIN: tl.constexpr):
    """Initialize tau for alpha=2.0"""
    BIN_MASK: tl.constexpr = (1 << BITS_PER_BIN) - 1  # 0xFF
    sum_z = 1.0
    sum_k = 1.0
    for sj in range(BINS - 1, -1, -1):
        c_bin = (hist >> (sj * BITS_PER_BIN)) & BIN_MASK
        c_bin = tl.sum(c_bin)
        if sj == BINS - 1:
            c_bin -= 1
        c_tau = sj / BINS
        new_z = sum_z + c_bin * c_tau
        new_k = sum_k + c_bin
        flag_good = (new_z - new_k * c_tau) < 1.0
        sum_z = tl.where(flag_good, new_z, sum_z)
        sum_k = tl.where(flag_good, new_k, sum_k)
    t = t_init + (sum_z - 1.0) / sum_k
    return t


@triton.jit
def hist_init_tau_alpha15(hist, t_init, BINS: tl.constexpr, BITS_PER_BIN: tl.constexpr):
    """Initialize tau for alpha=1.5"""
    BIN_MASK: tl.constexpr = (1 << BITS_PER_BIN) - 1  # 0xFF
    sum_z = 1.0
    sum_k = 1.0
    sum_zsquared = 1.0
    for sj in range(BINS - 1, -1, -1):
        c_bin = (hist >> (sj * BITS_PER_BIN)) & BIN_MASK
        c_bin = tl.sum(c_bin)
        if sj == BINS - 1:
            c_bin -= 1
        c_tau = sj / BINS
        new_z = sum_z + c_bin * c_tau
        new_k = sum_k + c_bin
        new_z_squared = sum_zsquared + c_bin * (c_tau * c_tau)
        flag_good = new_k * c_tau * c_tau - 2 * new_z * c_tau + new_z_squared < 1.0
        sum_zsquared = tl.where(flag_good, new_z_squared, sum_zsquared)
        sum_z = tl.where(flag_good, new_z, sum_z)
        sum_k = tl.where(flag_good, new_k, sum_k)
    t = t_init + (sum_z - tl.sqrt(sum_z * sum_z - sum_k * (sum_zsquared - 1))) / sum_k
    return t


@triton.jit
def hist_init_tau_generic(hist, t_init, alpha: tl.constexpr, BINS: tl.constexpr, BITS_PER_BIN: tl.constexpr):
    """Initialize tau for generic alpha"""
    return t_init  # return bisect lower bound for now


@triton.jit
def evaluate_f_hist(tau, t_shift, hist, BINS: tl.constexpr, BITS_PER_BIN: tl.constexpr):
    """
    Evaluate f(tau) ≈ sum_j count_j * [bin_center_j - tau]_+^2 - 1

    Args:
        tau: scalar tau value (BLOCK_M,) shaped
        t_shift: the shift applied (max - 1.0) (BLOCK_M,)
        hist: packed histogram (BLOCK_M, BLOCK_N) in uint64
        BINS: number of bins
        BITS_PER_BIN: bits per bin (typically 8)

    Returns:
        f_values: (BLOCK_M,) the function values
    """
    BIN_MASK: tl.constexpr = (1 << BITS_PER_BIN) - 1

    # Initialize accumulator
    f_acc = 0.0

    # Loop over bins and accumulate
    for bin_idx in range(BINS):
        # Extract count for this bin
        c_bin = (hist >> (bin_idx * BITS_PER_BIN)).to(tl.int32) & BIN_MASK
        c_bin = tl.sum(c_bin).to(tl.float32)

        # Bin center in projected space [0, 1]
        bin_center_proj = (bin_idx + 0.5) / BINS

        # Bin center in original qk space
        bin_center_qk = t_shift + bin_center_proj

        # Compute [bin_center - tau]_+^2
        diff = bin_center_qk - tau
        diff_pos = tl.where(diff > 0, diff, 0.0)

        # Accumulate: count * diff^2
        f_acc += c_bin * diff_pos * diff_pos

    # Subtract 1 to get f(tau) = sum(...) - 1
    return f_acc - 1.0


@triton.jit
def hist_init_tau_muller(t_lo, t_hi, t_shift, hist, BINS: tl.constexpr, BITS_PER_BIN: tl.constexpr):
    """
    Compute quadratic initialization from histogram.

    Given:
        - Histogram counts
        - Bounds t_lo, t_hi

    Procedure:
        1. Define three points: a = t_lo, b = (t_lo + t_hi) / 2, c = t_hi
        2. Evaluate f_hist at these three points
        3. Fit quadratic through (a, fa), (b, fb), (c, fc)
        4. Find root of quadratic in [a, c]

    Args:
        t_lo: lower bound for tau (BLOCK_M,)
        t_hi: upper bound for tau (BLOCK_M,)
        t_shift: the shift value (max - 1.0) (BLOCK_M,)
        hist: packed histogram (BLOCK_M, BLOCK_N) in uint64
        BINS: number of bins
        BITS_PER_BIN: bits per bin

    Returns:
        tau0: initialized tau value (BLOCK_M,)
    """
    EPS: tl.constexpr = 1e-10

    # Three evaluation points
    a = t_lo
    c = t_hi
    b = 0.5 * (a + c)

    # Evaluate f at these points
    fa = evaluate_f_hist(a, t_shift, hist, BINS, BITS_PER_BIN)
    fb = evaluate_f_hist(b, t_shift, hist, BINS, BITS_PER_BIN)
    fc = evaluate_f_hist(c, t_shift, hist, BINS, BITS_PER_BIN)

    # Fit quadratic f(x) = A*x^2 + B*x + C through three points
    # Using Lagrange interpolation formulas

    # Denominator for Lagrange formulas
    denom = (a - b) * (a - c) * (b - c)

    # Check if points are too close (degenerate case)
    is_degenerate = tl.abs(denom) < EPS

    # Compute quadratic coefficients
    # A = [fa*(b-c) + fb*(c-a) + fc*(a-b)] / denom
    A = (fa * (b - c) + fb * (c - a) + fc * (a - b)) / denom

    # B = [fa*(c^2-b^2) + fb*(a^2-c^2) + fc*(b^2-a^2)] / denom
    B = (fa * (c * c - b * b) + fb * (a * a - c * c) + fc * (b * b - a * a)) / denom

    # C = [fa*b*c*(b-c) + fb*a*c*(c-a) + fc*a*b*(a-b)] / denom
    C = (fa * b * c * (b - c) + fb * a * c * (c - a) + fc * a * b * (a - b)) / denom

    # Now solve A*x^2 + B*x + C = 0

    # Check if quadratic (A != 0) or linear
    is_linear = tl.abs(A) < EPS

    # Linear case: x = -C/B
    x_linear = tl.where(tl.abs(B) > EPS, -C / B, 0.5 * (a + c))

    # Quadratic case: x = (-B ± sqrt(B^2 - 4AC)) / (2A)
    discriminant = B * B - 4 * A * C

    # If discriminant < 0, fall back to secant on (a, c)
    has_real_roots = discriminant >= 0

    sqrt_disc = tl.sqrt(tl.maximum(discriminant, 0.0))

    # Two potential roots
    root1 = (-B + sqrt_disc) / (2 * A)
    root2 = (-B - sqrt_disc) / (2 * A)

    # Check which roots are in [a, c]
    root1_valid = (root1 >= a) & (root1 <= c)
    root2_valid = (root2 >= a) & (root2 <= c)

    # Prefer root closer to c (right endpoint)
    dist1 = tl.abs(c - root1)
    dist2 = tl.abs(c - root2)

    # Select best root
    # If both valid, pick closest to c
    # If only one valid, pick that one
    # If neither valid, fall back to secant
    x_quad = tl.where(
        root1_valid & root2_valid,
        tl.where(dist1 <= dist2, root1, root2),
        tl.where(root1_valid, root1, tl.where(root2_valid, root2, 0.5 * (a + c))),
    )

    # If no real roots, use secant on (a, c)
    # Secant: x = c - fc * (c - a) / (fc - fa)
    secant_denom = fc - fa
    x_secant = tl.where(
        tl.abs(secant_denom) > EPS,
        c - fc * (c - a) / secant_denom,
        0.5 * (a + c),
    )

    x_quad = tl.where(has_real_roots, x_quad, x_secant)

    # Choose between linear and quadratic
    x_result = tl.where(is_linear, x_linear, x_quad)

    # Handle degenerate case (points too close)
    x_result = tl.where(is_degenerate, 0.5 * (a + c), x_result)

    # Clamp to bounds for safety
    tau0 = tl.minimum(tl.maximum(x_result, a), c)

    return tau0


@triton.jit
def hist_init_tau_secant(t_lo, t_hi, t_shift, hist, BINS: tl.constexpr, BITS_PER_BIN: tl.constexpr):
    """
    Do secant on endpoints after histogram.

    Args:
        t_lo: lower bound (BLOCK_M,)
        t_hi: upper bound (BLOCK_M,)
        t_shift: shift value (BLOCK_M,)
        hist: packed histogram (BLOCK_M, BLOCK_N)
        BINS: number of bins
        BITS_PER_BIN: bits per bin

    Returns:
        tau0: initialized tau (BLOCK_M,)
    """
    EPS: tl.constexpr = 1e-10

    # Evaluate at endpoints
    fa = evaluate_f_hist(t_lo, t_shift, hist, BINS, BITS_PER_BIN)
    fc = evaluate_f_hist(t_hi, t_shift, hist, BINS, BITS_PER_BIN)

    # Secant formula: x = c - fc * (c - a) / (fc - fa)
    denom = fc - fa
    tau0 = tl.where(
        tl.abs(denom) > EPS,
        t_hi - fc * (t_hi - t_lo) / denom,
        0.5 * (t_lo + t_hi),
    )

    # Clamp to bounds
    tau0 = tl.minimum(tl.maximum(tau0, t_lo), t_hi)

    return tau0


@_autotune(configs=get_configs(), key=["N_ITER", "USE_HISTOGRAM"])
@triton.jit
def _fwd_entmax(
    X,
    Y,
    ALPHA: tl.constexpr,
    N_ITER: tl.constexpr,
    SIZE_N: tl.constexpr,
    USE_HISTOGRAM: tl.constexpr,
    FAST_MATH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Improved forward pass of entmax with hybrid updates and optional histogram init.

    Args:
        X: Input tensor pointer
        Y: Output tensor pointer
        ALPHA: Entmax alpha parameter (> 1)
        N_ITER: Number of optimization iterations
        SIZE_N: Size of the last dimension
        USE_HISTOGRAM: Whether to use histogram-based initialization
        BLOCK_N: Block size for processing
    """
    # Constants
    EPS: tl.constexpr = 1e-6
    BINS: tl.constexpr = 4
    BITS_PER_BIN: tl.constexpr = 16
    BIN_MASK: tl.constexpr = (1 << BITS_PER_BIN) - 1

    _scalar = ALPHA - 1
    coeff_0 = 1 / (ALPHA - 1)
    coeff_1 = coeff_0 - 1
    coeff_2 = coeff_1 - 1

    # Program ID and pointer setup
    off_m = tl.program_id(0) * SIZE_N
    X += off_m
    Y += off_m

    # ============================================================
    # STAGE 1: Find maximum value
    # ============================================================
    max_val = float("-inf")
    for curr_n in range(0, SIZE_N, BLOCK_N):
        curr_offsets = curr_n + tl.arange(0, BLOCK_N)
        load_mask = curr_offsets < SIZE_N
        x = tl.load(X + curr_offsets, mask=load_mask, other=float("-inf"))
        max_val = tl.maximum(max_val, tl.max(x))

    max_val *= _scalar

    # ============================================================
    # STAGE 2: Initialize tau (with optional histogram)
    # ============================================================
    if USE_HISTOGRAM and ALPHA <= 2.0:
        # Histogram-based initialization (only for alpha <= 2.0)
        t_init = max_val - 1.0

        # Build histogram
        hist = tl.zeros((BLOCK_N,), dtype=tl.uint64)
        for curr_n in range(0, SIZE_N, BLOCK_N):
            curr_offsets = curr_n + tl.arange(0, BLOCK_N)
            load_mask = curr_offsets < SIZE_N
            x = tl.load(X + curr_offsets, mask=load_mask, other=float("-inf")) * _scalar

            proj_x = x - t_init
            bin = float2int_rd(proj_x * BINS)
            bin = tl.minimum(tl.maximum(bin, 0), BINS - 1)  # [0, BINS-1]
            shift = bin * BITS_PER_BIN
            inc = (proj_x >= 0).to(tl.uint64) << shift
            hist += inc

        # Estimate tau from histogram
        if ALPHA == 1.5:
            t_init = hist_init_tau_alpha15(hist, t_init, BINS, BITS_PER_BIN)
        elif ALPHA == 2.0:
            t_init = hist_init_tau_alpha2(hist, t_init, BINS, BITS_PER_BIN)
        else:
            t_init = hist_init_tau_generic(hist, t_init, ALPHA, BINS, BITS_PER_BIN)

        t_lo = t_init
        t_hi = t_init + 1.0 / BINS
        t_shift = max_val - 1

        # t = 0.5 * (t_lo + t_hi)
        t = hist_init_tau_muller(t_lo, t_hi, t_shift, hist, BINS, BITS_PER_BIN)
        # t = hist_init_tau_secant(t_lo, t_hi, t_shift, hist, BINS, BITS_PER_BIN)

    else:
        # Simple initialization
        t_hi = max_val
        t_lo = max_val - 1.0
        t = 0.5 * (t_lo + t_hi)

    # ============================================================
    # STAGE 3: Refinement with hybrid updates
    # ============================================================
    tau_prev = t
    f_prev = 0.0

    for iter_idx in range(N_ITER):
        acc_0 = tl.zeros((BLOCK_N,), dtype=tl.float32)  # Function accumulator
        acc_1 = tl.zeros((BLOCK_N,), dtype=tl.float32)  # First derivative accumulator
        acc_2 = tl.zeros((BLOCK_N,), dtype=tl.float32)  # Second derivative accumulator

        # Accumulate statistics
        for curr_n in range(0, SIZE_N, BLOCK_N):
            curr_offsets = curr_n + tl.arange(0, BLOCK_N)
            load_mask = curr_offsets < SIZE_N
            x = tl.load(X + curr_offsets, mask=load_mask, other=float("-inf")) * _scalar

            x_mask = (x > t) & load_mask
            x_act = tl.maximum(x - t, 0.0)

            # Handle different alpha cases efficiently
            if ALPHA == 2.0:
                # Sparsemax: [x-tau]_+
                acc_0 += tl.where(x_mask, x_act, 0.0)
                acc_1 += x_mask.to(tl.float32)
            elif ALPHA == 1.5:
                # Entmax 1.5: [x-tau]_+^2
                acc_0 += tl.where(x_mask, x_act * x_act, 0.0)
                acc_1 += tl.where(x_mask, x_act, 0.0)
                acc_2 += x_mask.to(tl.float32)
            else:
                # Generic: [x-tau]_+^(1/(alpha-1))
                acc_0 += _masked_pow(x_act, x_mask, coeff_0, FAST_MATH)
                acc_1 += _masked_pow(x_act, x_mask, coeff_1, FAST_MATH)
                acc_2 += _masked_pow(x_act, x_mask, coeff_2, FAST_MATH)

        # Hybrid update
        s_acc_0 = tl.sum(acc_0)
        s_acc_1 = tl.sum(acc_1)
        s_acc_2 = tl.sum(acc_2)

        t_new, t_lo, t_hi = hybrid_update(
            t,
            t_lo,
            t_hi,
            tau_prev,
            f_prev,
            s_acc_0,
            s_acc_1,
            s_acc_2,
            coeff_0,
            coeff_1,
            ALPHA,
            iter_idx > 0,
        )

        # Update tracking
        f_current = s_acc_0 - 1.0
        tau_prev = t
        f_prev = f_current
        t = t_new

    # ============================================================
    # STAGE 4: Compute final output
    # ============================================================
    for curr_n in range(0, SIZE_N, BLOCK_N):
        curr_offsets = curr_n + tl.arange(0, BLOCK_N)
        load_mask = curr_offsets < SIZE_N
        x = tl.load(X + curr_offsets, mask=load_mask, other=float("-inf")) * _scalar

        x_mask = (x > t) & load_mask
        x_act = tl.maximum(x - t, 0.0)

        # Compute output based on alpha
        if ALPHA == 2.0:
            y = tl.where(x_mask, x_act, 0.0)
        elif ALPHA == 1.5:
            y = tl.where(x_mask, x_act * x_act, 0.0)
        else:
            y = _masked_pow(x_act, x_mask, coeff_0, FAST_MATH)

        tl.store(Y + curr_offsets, y, mask=load_mask)


@_autotune(configs=get_configs(), key=["SIZE_N", "FAST_MATH"])
@triton.jit
def _bwd_entmax(
    Y,
    DY,
    DX,
    ALPHA: tl.constexpr,
    SIZE_N: tl.constexpr,
    FAST_MATH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Improved backward pass with cleaner special-case handling.

    Args:
        Y: Output tensor from forward pass
        DY: Gradient of output
        DX: Gradient of input (to be computed)
        ALPHA: Entmax alpha parameter (> 1)
        SIZE_N: Size of the last dimension
        BLOCK_N: Block size for processing
    """
    EPS: tl.constexpr = 1e-6
    coeff_f = 2 - ALPHA

    # Program ID and pointer setup
    off_m = tl.program_id(0) * SIZE_N
    Y += off_m
    DY += off_m
    DX += off_m

    # First pass: compute scalar term
    uDy_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    y_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for curr_n in range(0, SIZE_N, BLOCK_N):
        curr_offsets = curr_n + tl.arange(0, BLOCK_N)
        load_mask = curr_offsets < SIZE_N

        y = tl.load(Y + curr_offsets, mask=load_mask, other=0.0)
        dy = tl.load(DY + curr_offsets, mask=load_mask, other=0.0)

        u_mask = (y > 0) & load_mask

        # Handle different alpha cases
        if ALPHA == 2.0:
            # Sparsemax: u = 1
            u = tl.where(u_mask, 1.0, 0.0)
        elif ALPHA == 1.5:
            # Entmax 1.5: u = y^0.5
            u = tl.where(u_mask, tl.sqrt(y), 0.0)
        else:
            # Generic: u = y^(2-alpha)
            u = _masked_pow(y, u_mask, coeff_f, FAST_MATH)

        uDy_sum += tl.sum(u * dy)
        y_sum += tl.sum(u)

    scalar = uDy_sum / y_sum

    # Second pass: compute and store gradients
    for curr_n in range(0, SIZE_N, BLOCK_N):
        curr_offsets = curr_n + tl.arange(0, BLOCK_N)
        load_mask = curr_offsets < SIZE_N

        y = tl.load(Y + curr_offsets, mask=load_mask, other=0.0)
        dy = tl.load(DY + curr_offsets, mask=load_mask, other=0.0)

        u_mask = (y > 0) & load_mask

        # Handle different alpha cases
        if ALPHA == 2.0:
            u = tl.where(u_mask, 1.0, 0.0)
        elif ALPHA == 1.5:
            u = tl.where(u_mask, tl.sqrt(y), 0.0)
        else:
            u = _masked_pow(y, u_mask, coeff_f, FAST_MATH)

        grad = u * dy - scalar * u
        tl.store(DX + curr_offsets, grad, mask=load_mask)


class _entmax_triton(torch.autograd.Function):
    """Improved PyTorch autograd wrapper for entmax."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        alpha: float = 1.5,
        n_iter: int = 10,
        use_histogram: bool = True,
        fast_math: bool = False,
    ):
        """Forward pass with optional histogram initialization.

        Args:
            x: Input tensor
            alpha: Entmax alpha parameter (> 1)
            n_iter: Number of optimization iterations
            use_histogram: Use histogram-based initialization
            fast_math: Use fast math approximations

        Returns:
            torch.Tensor: entmax transformed output
        """
        SIZE_N = x.shape[-1]
        assert x.is_contiguous(), "x must be contiguous"

        y = torch.zeros_like(x).contiguous()
        grid = (x.numel() // SIZE_N, 1, 1)

        _fwd_entmax[grid](
            x,
            y,
            alpha,
            n_iter,
            SIZE_N,
            use_histogram,
            fast_math,
        )

        ctx.save_for_backward(y)
        ctx.alpha = alpha
        ctx.fast_math = fast_math

        return y

    @staticmethod
    def backward(ctx, dy):
        """Backward pass."""
        y = ctx.saved_tensors[0]
        SIZE_N = y.shape[-1]

        if not dy.is_contiguous():
            dy = dy.contiguous()

        dx = torch.zeros_like(dy).contiguous()
        grid = (y.numel() // SIZE_N, 1, 1)

        _bwd_entmax[grid](
            y,
            dy,
            dx,
            ctx.alpha,
            SIZE_N,
            ctx.fast_math,
        )

        return dx, None, None, None, None


def triton_entmax(
    x: torch.Tensor,
    alpha: float = 1.5,
    n_iter: int = 2,
    use_histogram: bool = True,
    fast_math: bool = False,
) -> torch.Tensor:
    """Improved entmax with hybrid updates and optional histogram initialization.

    Args:
        x: Input tensor
        alpha: Entmax alpha parameter (> 1)
        n_iter: Number of optimization iterations
        use_histogram: Use histogram-based initialization (recommended for alpha <= 2.0)
        fast_math: Use fast math approximations

    Returns:
        torch.Tensor: entmax transformed output

    Example:
        >>> x = torch.randn(128, 256).cuda()
        >>> y = triton_entmax(x, alpha=1.5, n_iter=2, use_histogram=True)
    """
    return _entmax_triton.apply(x, alpha, n_iter, use_histogram, fast_math)


# ============================================================
# Convenience functions for specific alpha values
# ============================================================


def triton_sparsemax(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Sparsemax (entmax with alpha=2.0)."""
    return triton_entmax(x, alpha=2.0, **kwargs)


def triton_entmax15(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Entmax 1.5 with histogram initialization."""
    return triton_entmax(x, alpha=1.5, **kwargs)
