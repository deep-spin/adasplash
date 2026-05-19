from math import sqrt
import os

import torch

import triton
import triton.language as tl
from triton.language.extra.libdevice import float2int_rd, popc

DEBUG = os.environ.get("ADASPLASH_TEST_FAST_AUTOTUNE", "0") == "1"


def _autotune(*args, **kwargs):
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return lambda fn: fn
    return triton.autotune(*args, **kwargs)


def get_config():
    global DEBUG
    block_pairs = [
        (128, 128),
        (128, 64),
        (128, 32),
        (64, 64),
        (64, 32),
    ]
    all_nw = [2, 4]
    all_ns = [2, 3, 4]
    if DEBUG:
        block_pairs = [(32, 64)]
        all_nw = [2]
        all_ns = [2]
    return [
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
            },
            num_warps=nw,
            num_stages=ns,
        )
        for (BM, BN) in block_pairs
        for nw in all_nw
        for ns in all_ns
    ]


def get_config_v2():
    global DEBUG
    if DEBUG:
        all_nw = [2]
        all_ns = [2]
    else:
        all_nw = [2, 4]
        all_ns = [2, 3, 4]
    return [
        triton.Config(
            {},
            num_warps=nw,
            num_stages=ns,
        )
        for nw in all_nw
        for ns in all_ns
    ]


@triton.jit
def halley_bisect_update(t, t_lo, t_hi, acc_0, acc_1, acc_2):
    EPS: tl.constexpr = 1e-5

    ## -- function eval --
    ff = acc_0 - 1.0
    ## -- first derivative --
    df = -2 * acc_1
    ## -- second derivative --
    ddf = 2 * acc_2

    ## -- update bounds --
    t_lo = tl.where((ff > 0), t, t_lo)
    t_hi = tl.where((ff < 0), t, t_hi)

    ## -- halley's update --
    new_t = t - (ff * df) / (df * df - 0.5 * ff * ddf)

    ## -- is halley's inside the bounds? --
    is_good = (new_t > t_lo - EPS) & (new_t < t_hi + EPS)
    t = tl.where(is_good, new_t, 0.5 * (t_lo + t_hi))

    return t, t_lo, t_hi


@triton.jit
def get_qk(
    c_block,
    q,
    k,
    seqlen,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    SMALL_NUMBER: tl.constexpr = -10000.0
    start_m = tl.program_id(0)
    qk = tl.dot(q, tl.trans(k), input_precision="ieee")

    if IS_CAUSAL or IS_VARLEN:
        starting_col = c_block * BLOCK_N
        cols = starting_col + tl.arange(0, BLOCK_N)
        mask = tl.full((BLOCK_M, BLOCK_N), 1, dtype=tl.int1)
        if IS_CAUSAL:
            rows = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            mask = mask & (rows[:, None] >= cols[None, :])
        if IS_VARLEN:
            mask = mask & (cols[None, :] < seqlen)
        qk = tl.where(mask, qk, SMALL_NUMBER)

    return qk


@triton.jit
def get_qk_t(
    c_block,
    q,
    k,
    seqlen,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    SMALL_NUMBER: tl.constexpr = -10000.0
    start_n = tl.program_id(0)
    qk = tl.dot(q, k, input_precision="ieee")

    if IS_CAUSAL or IS_VARLEN:
        starting_col = start_n * BLOCK_N
        cols = starting_col + tl.arange(0, BLOCK_N)
        mask = tl.full((BLOCK_M, BLOCK_N), 1, dtype=tl.int1)
        if IS_CAUSAL:
            rows = c_block * BLOCK_M + tl.arange(0, BLOCK_M)
            mask = mask & (rows[:, None] >= cols[None, :])
        if IS_VARLEN:
            mask = mask & (cols[None, :] < seqlen)
        qk = tl.where(mask, qk, SMALL_NUMBER)

    return qk


@_autotune(
    configs=get_config(),
    restore_value=["TAUS"],
    key=["N_CTX", "H_DIM"],
)
@triton.jit
def _get_tau(
    Q,
    K,
    TAUS,
    VARLEN,
    ##
    sm_scale,
    ##
    N_H: tl.constexpr,
    N_KV_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    H_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    stride_qh,
    stride_kvh,
    stride_th,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    kv_jump: tl.constexpr = BLOCK_N * H_DIM

    ## -- some coefficients --
    _scalar = 0.5 * sm_scale

    ## -- offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    q_offset = off_hz * stride_qh

    ## -- GQA: K shares one head across GROUP_SIZE consecutive Q heads --
    off_kv_h = off_h // GROUP_SIZE
    kv_offset = (off_z * N_KV_H + off_kv_h) * stride_kvh

    ## -- per-batch seqlen (and early-return for fully-OOB M-blocks) --
    seqlen = N_CTX
    if IS_VARLEN:
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)
        if start_m * BLOCK_M >= seqlen:
            return

    ## -- update pointer offsets --
    Q += q_offset + start_m * BLOCK_M * H_DIM
    K += kv_offset
    TAUS += off_hz * stride_th + start_m * BLOCK_M

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, H_DIM)

    ## -- ptrs --
    q_ptrs = Q + offsets_m[:, None] * H_DIM + offsets_k
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k
    t_ptrs = TAUS + offsets_m

    ## -- now let's load q --
    if IS_VARLEN:
        q_mask = offsets_m < seqlen - start_m * BLOCK_M
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0) * _scalar
    else:
        q = tl.load(q_ptrs) * _scalar
    q = q.to(input_dtype)

    masked_nblocks: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N)
    if IS_VARLEN:
        up_to_seqlen = tl.minimum((start_m + 1) * BLOCK_M, seqlen)
    else:
        up_to_seqlen = (start_m + 1) * BLOCK_M
    total_blocks = tl.cdiv(up_to_seqlen, BLOCK_N)
    end_masked = tl.minimum(masked_nblocks, total_blocks)

    ## ------------------------------------------------------------------
    ## 1.  First pass: get max.
    ## ------------------------------------------------------------------

    mvals = tl.full((BLOCK_M,), value=-1.0e3, dtype=tl.float32)
    for n_blk in range(0, end_masked):
        c_block = (total_blocks - 1) - n_blk

        if IS_VARLEN:
            k_mask = (c_block * BLOCK_N + offsets_n) < seqlen
            k = tl.load(k_ptrs + c_block * kv_jump, mask=k_mask[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs + c_block * kv_jump)
        qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, True, IS_VARLEN)

        ## -- update now mvals --
        mvals = tl.maximum(mvals, tl.max(qk, axis=1))

    for n_blk in range(masked_nblocks, total_blocks):
        c_block = (total_blocks - 1) - n_blk

        if IS_VARLEN:
            k_mask = (c_block * BLOCK_N + offsets_n) < seqlen
            k = tl.load(k_ptrs + c_block * kv_jump, mask=k_mask[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs + c_block * kv_jump)
        qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, False, IS_VARLEN)
        ## -- update now mvals --
        mvals = tl.maximum(mvals, tl.max(qk, axis=1))

    ## -- store tau and also mask for over seqlen entries --
    t = mvals - 1.0
    if IS_VARLEN:
        tl.store(t_ptrs, t, mask=q_mask)
    else:
        tl.store(t_ptrs, t)


def _select_bins(n_ctx: int, block_n: int) -> int:
    """Largest BINS in {8, 4, 2} such that the per-bin uint64 slot
    (64 // BINS bits wide) can hold cdiv(n_ctx, block_n) counts.

    The histogram in _get_tau_v2 packs BINS counters into a uint64, so each
    slot is 64 // BINS bits wide and saturates at (1 << bits) - 1. Per cell
    the count grows by at most 1 per K-block, bounded by cdiv(n_ctx, block_n).
    More bins = finer bracket for the subsequent Halley refinement, so we
    pick the largest BINS that still avoids overflow.
    """
    max_count = (n_ctx + block_n - 1) // block_n
    for bins in (8, 4, 2):
        bits = 64 // bins
        if max_count <= (1 << bits) - 1:
            return bins

    raise ValueError(f"N_CTX={n_ctx} too large for uint64 packed histogram")


@_autotune(
    configs=get_config_v2(),
    restore_value=["TAUS"],
    key=["N_CTX", "H_DIM"],
)
@triton.jit
def _get_tau_v2(
    Q,
    K,
    TAUS,
    VARLEN,
    ##
    sm_scale,
    ##
    N_H: tl.constexpr,
    N_KV_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    H_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    stride_qh,
    stride_kvh,
    stride_th,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BINS: tl.constexpr,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    kv_jump: tl.constexpr = BLOCK_N * H_DIM

    BITS_PER_BIN: tl.constexpr = 64 // BINS
    BIN_MASK: tl.constexpr = (1 << BITS_PER_BIN) - 1

    ## -- some coefficients --
    _scalar = 0.5 * sm_scale

    ## -- offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    q_offset = off_hz * stride_qh

    ## -- GQA: K shares one head across GROUP_SIZE consecutive Q heads --
    off_kv_h = off_h // GROUP_SIZE
    kv_offset = (off_z * N_KV_H + off_kv_h) * stride_kvh

    ## -- per-batch seqlen (and early-return for fully-OOB M-blocks) --
    seqlen = N_CTX
    if IS_VARLEN:
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)
        if start_m * BLOCK_M >= seqlen:
            return

    ## -- update pointer offsets --
    Q += q_offset + start_m * BLOCK_M * H_DIM
    K += kv_offset
    TAUS += off_hz * stride_th + start_m * BLOCK_M

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, H_DIM)

    ## -- ptrs --
    q_ptrs = Q + offsets_m[:, None] * H_DIM + offsets_k
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k
    t_ptrs = TAUS + offsets_m

    ## -- now let's load q --
    if IS_VARLEN:
        q_mask = offsets_m < seqlen - start_m * BLOCK_M
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0) * _scalar
    else:
        q = tl.load(q_ptrs) * _scalar
    q = q.to(input_dtype)

    # masked_nblocks: tl.constexpr = BLOCK_M // BLOCK_N
    # total_blocks = ((start_m + 1) * BLOCK_M) // BLOCK_N

    masked_nblocks: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N)
    if IS_VARLEN:
        up_to_seqlen = tl.minimum((start_m + 1) * BLOCK_M, seqlen)
    else:
        up_to_seqlen = (start_m + 1) * BLOCK_M
    total_blocks = tl.cdiv(up_to_seqlen, BLOCK_N)

    if IS_VARLEN:
        t = tl.load(t_ptrs, mask=q_mask, other=0.0)
    else:
        t = tl.load(t_ptrs)

    ## ------------------------------------------------------------------
    ## 2.  Get histogram.
    ## ------------------------------------------------------------------
    hist = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.uint64)
    for n_blk in range(0, masked_nblocks):
        c_block = (total_blocks - 1) - n_blk

        if IS_VARLEN:
            k_mask = (c_block * BLOCK_N + offsets_n) < seqlen
            k = tl.load(k_ptrs + c_block * kv_jump, mask=k_mask[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs + c_block * kv_jump)
        qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, True, IS_VARLEN)

        ## -- project our x --
        proj_x = qk - t[:, None]

        ## -- transform qk into uint64 bins --
        bin = float2int_rd(proj_x * BINS)
        bin = tl.minimum(bin, BINS - 1)
        bin = (bin >= 0).to(tl.uint64) << (bin * BITS_PER_BIN)

        ## -- agregate bins --
        hist += bin

    for n_blk in range(masked_nblocks, total_blocks):
        c_block = (total_blocks - 1) - n_blk

        if IS_VARLEN:
            k_mask = (c_block * BLOCK_N + offsets_n) < seqlen
            k = tl.load(k_ptrs + c_block * kv_jump, mask=k_mask[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs + c_block * kv_jump)
        qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, False, IS_VARLEN)

        ## -- project our x --
        proj_x = qk - t[:, None]

        ## -- transform qk into uint64 bins --
        bin = float2int_rd(proj_x * BINS)
        bin = tl.minimum(bin, BINS - 1)
        bin = (bin >= 0).to(tl.uint64) << (bin * BITS_PER_BIN)

        ## -- agregate bins --
        hist += bin

    ## -- get tau and its bounds --
    sum_zsquared = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_z = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_n = tl.zeros((BLOCK_M,), dtype=tl.float32)

    sum_zsquared += 1.0
    sum_z += 1.0
    sum_n += 1.0

    for sj in range(BINS - 1, -1, -1):
        c_bin = (hist >> (sj * BITS_PER_BIN)) & BIN_MASK
        c_bin = tl.sum(c_bin, axis=1)

        if sj == BINS - 1:
            c_bin -= 1

        c_tau = sj / BINS

        ## -- for alpha = 1.5 we need three accs --
        new_z_squared = sum_zsquared + c_bin * (c_tau * c_tau)
        new_z = sum_z + c_bin * c_tau
        new_n = sum_n + c_bin

        flag_good = new_n * c_tau * c_tau - 2 * new_z * c_tau + new_z_squared < 1.0
        sum_zsquared = tl.where(flag_good, new_z_squared, sum_zsquared)
        sum_z = tl.where(flag_good, new_z, sum_z)
        sum_n = tl.where(flag_good, new_n, sum_n)

    t += (sum_z - tl.sqrt(sum_z * sum_z - sum_n * (sum_zsquared - 1.0))) / sum_n

    ## -- store tau and also mask for over seqlen entries --
    if IS_VARLEN:
        tl.store(t_ptrs, t, mask=q_mask)
    else:
        tl.store(t_ptrs, t)


@_autotune(
    configs=get_config_v2(),
    restore_value=["TAUS"],
    key=["N_CTX", "H_DIM"],
)
@triton.jit
def _get_tau_v3(
    Q,
    K,
    TAUS,
    BMASK,
    VARLEN,
    ##
    sm_scale,
    NITER: tl.constexpr,
    ##
    N_H: tl.constexpr,
    N_KV_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    H_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    N_INT32s: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    stride_qh,
    stride_kvh,
    stride_th,
    stride_bmh,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BINS: tl.constexpr,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    kv_jump: tl.constexpr = BLOCK_N * H_DIM

    ## -- some coefficients --
    _scalar = 0.5 * sm_scale

    ## -- offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    q_offset = off_hz * stride_qh

    ## -- GQA: K shares one head across GROUP_SIZE consecutive Q heads --
    off_kv_h = off_h // GROUP_SIZE
    kv_offset = (off_z * N_KV_H + off_kv_h) * stride_kvh

    ## -- per-batch seqlen (and early-return for fully-OOB M-blocks) --
    seqlen = N_CTX
    if IS_VARLEN:
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)
        if start_m * BLOCK_M >= seqlen:
            return

    ## -- update pointer offsets --
    Q += q_offset + start_m * BLOCK_M * H_DIM
    K += kv_offset
    TAUS += off_hz * stride_th + start_m * BLOCK_M
    BMASK += off_hz * stride_bmh + start_m * N_INT32s

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, H_DIM)

    ## -- ptrs --
    q_ptrs = Q + offsets_m[:, None] * H_DIM + offsets_k
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k
    t_ptrs = TAUS + offsets_m

    ## -- now let's load q --
    if IS_VARLEN:
        q_mask = offsets_m < seqlen - start_m * BLOCK_M
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0) * _scalar
    else:
        q = tl.load(q_ptrs) * _scalar
    q = q.to(input_dtype)

    masked_nblocks: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N)
    if IS_VARLEN:
        up_to_seqlen = tl.minimum((start_m + 1) * BLOCK_M, seqlen)
    else:
        up_to_seqlen = (start_m + 1) * BLOCK_M
    total_blocks = tl.cdiv(up_to_seqlen, BLOCK_N)

    if IS_VARLEN:
        t = tl.load(t_ptrs, mask=q_mask, other=0.0)
    else:
        t = tl.load(t_ptrs)
    t_lo = t
    t_hi = t + 1.0 / BINS
    t = 0.5 * (t_lo + t_hi)

    for _ in tl.static_range(NITER):
        acc_0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n_blk in range(0, masked_nblocks):
            c_block = (total_blocks - 1) - n_blk

            if IS_VARLEN:
                k_mask = (c_block * BLOCK_N + offsets_n) < seqlen
                k = tl.load(k_ptrs + c_block * kv_jump, mask=k_mask[:, None], other=0.0)
            else:
                k = tl.load(k_ptrs + c_block * kv_jump)
            qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, True, IS_VARLEN)

            qk_mask = qk > t[:, None]

            qk_mask = qk_mask.to(tl.float32)
            qk_proj = (qk - t[:, None]) * qk_mask

            ## -- Acc for f, f', f'' --
            acc_0 += tl.sum(qk_proj * qk_proj, axis=1)
            acc_1 += tl.sum(qk_proj, axis=1)
            acc_2 += tl.sum(qk_mask, axis=1)

        for n_blk in range(masked_nblocks, total_blocks):
            c_block = (total_blocks - 1) - n_blk

            if IS_VARLEN:
                k_mask = (c_block * BLOCK_N + offsets_n) < seqlen
                k = tl.load(k_ptrs + c_block * kv_jump, mask=k_mask[:, None], other=0.0)
            else:
                k = tl.load(k_ptrs + c_block * kv_jump)
            qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, False, IS_VARLEN)

            qk_mask = qk > t[:, None]

            qk_mask = qk_mask.to(tl.float32)
            qk_proj = (qk - t[:, None]) * qk_mask

            ## -- Acc for f, f', f'' --
            acc_0 += tl.sum(qk_proj * qk_proj, axis=1)
            acc_1 += tl.sum(qk_proj, axis=1)
            acc_2 += tl.sum(qk_mask, axis=1)

        t, t_lo, t_hi = halley_bisect_update(t, t_lo, t_hi, acc_0, acc_1, acc_2)  # fmt: skip

    ## ------------------------------------------------------------------
    ## 3.  Second pass: last tau update and save mask.
    ## ------------------------------------------------------------------

    ## -- accumulate --
    acc_0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    bmask = 0
    for n_blk in range(0, masked_nblocks):
        c_block = (total_blocks - 1) - n_blk

        if IS_VARLEN:
            k_mask = (c_block * BLOCK_N + offsets_n) < seqlen
            k = tl.load(k_ptrs + c_block * kv_jump, mask=k_mask[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs + c_block * kv_jump)
        qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, True, IS_VARLEN)

        qk_mask = qk > t[:, None]

        qk_mask = qk_mask.to(tl.float32)
        qk_proj = (qk - t[:, None]) * qk_mask

        ## -- Acc for f, f', f'' --
        acc_0 += tl.sum(qk_proj * qk_proj, axis=1)
        acc_1 += tl.sum(qk_proj, axis=1)
        acc_2 += tl.sum(qk_mask, axis=1)

        ## -- Update bmask --
        bmask <<= 1
        bmask += tl.sum(qk_mask) > 0.5
        if (c_block % 32) == 0:
            tl.store(BMASK + c_block // 32, bmask)
            bmask = 0

    for n_blk in range(masked_nblocks, total_blocks):
        c_block = (total_blocks - 1) - n_blk

        if IS_VARLEN:
            k_mask = (c_block * BLOCK_N + offsets_n) < seqlen
            k = tl.load(k_ptrs + c_block * kv_jump, mask=k_mask[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs + c_block * kv_jump)
        qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, False, IS_VARLEN)

        qk_mask = qk > t[:, None]

        qk_mask = qk_mask.to(tl.float32)
        qk_proj = (qk - t[:, None]) * qk_mask

        ## -- Acc for f, f', f'' --
        acc_0 += tl.sum(qk_proj * qk_proj, axis=1)
        acc_1 += tl.sum(qk_proj, axis=1)
        acc_2 += tl.sum(qk_mask, axis=1)

        ## -- Update bmask --
        bmask <<= 1
        bmask += tl.sum(qk_mask) > 0.5
        if (c_block % 32) == 0:
            tl.store(BMASK + c_block // 32, bmask)
            bmask = 0
    t, t_lo, t_hi = halley_bisect_update(t, t_lo, t_hi, acc_0, acc_1, acc_2)  # fmt: skip

    ## -- store tau and also mask for over seqlen entries --
    if IS_VARLEN:
        tl.store(t_ptrs, t, mask=q_mask)
    else:
        tl.store(t_ptrs, t)


@triton.jit
def get_next_block(bmask, n_blk):
    return tl.inline_asm_elementwise(
        "fns.b32 $0, $1, 0, $2;",
        "=r,r,r",
        [bmask, n_blk + 1],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@_autotune(
    configs=get_config_v2(),
    key=["N_CTX", "H_DIM"],
)
@triton.jit
def _get_output(
    Q,
    K,
    V,
    OUT,
    OUT2,
    TAUS,
    BMASK,
    VARLEN,
    ##
    sm_scale,
    NEED_BACKWARD: tl.constexpr,
    ##
    N_H: tl.constexpr,
    N_KV_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    H_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    N_INT32s: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    stride_qh,
    stride_kvh,
    stride_th,
    stride_bmh,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    kv_jump: tl.constexpr = BLOCK_N * H_DIM

    ## -- some coefficients --
    _scalar = 0.5 * sm_scale

    ## -- offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    q_offset = off_hz * stride_qh

    ## -- GQA: K/V share one head across GROUP_SIZE consecutive Q heads --
    off_kv_h = off_h // GROUP_SIZE
    kv_offset = (off_z * N_KV_H + off_kv_h) * stride_kvh

    ## -- per-batch seqlen (and early-return for fully-OOB M-blocks) --
    seqlen = N_CTX
    if IS_VARLEN:
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)
        if start_m * BLOCK_M >= seqlen:
            return

    ## -- update pointer offsets --
    Q += q_offset + start_m * BLOCK_M * H_DIM
    K += kv_offset
    V += kv_offset
    OUT += q_offset + start_m * BLOCK_M * H_DIM
    OUT2 += q_offset + start_m * BLOCK_M * H_DIM
    TAUS += off_hz * stride_th + start_m * BLOCK_M
    BMASK += off_hz * stride_bmh + start_m * N_INT32s

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, H_DIM)

    ## -- ptrs --
    q_ptrs = Q + offsets_m[:, None] * H_DIM + offsets_k
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k
    v_ptrs = V + offsets_n[:, None] * H_DIM + offsets_k
    t_ptrs = TAUS + offsets_m

    ## -- now let's load q --
    if IS_VARLEN:
        q_mask = offsets_m < seqlen - start_m * BLOCK_M
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0) * _scalar
    else:
        q = tl.load(q_ptrs) * _scalar
    q = q.to(input_dtype)

    if IS_VARLEN:
        t = tl.load(t_ptrs, mask=q_mask, other=0.0)
    else:
        t = tl.load(t_ptrs)

    ## -- accumulate --
    out = tl.zeros((BLOCK_M, H_DIM), dtype=tl.float32)
    if NEED_BACKWARD:
        out2 = tl.zeros([BLOCK_M, H_DIM], dtype=tl.float32)
        supp_size = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_int in tl.static_range(N_INT32s):
        bmask = tl.load(BMASK + n_int).cast(tl.int32, bitcast=True)
        total_blocks = popc(bmask)
        base_block = 32 * n_int
        for n_blk in range(0, total_blocks):
            c_block = base_block + get_next_block(bmask, n_blk)

            if IS_VARLEN:
                kv_mask = (c_block * BLOCK_N + offsets_n) < seqlen
                k = tl.load(k_ptrs + c_block * kv_jump, mask=kv_mask[:, None], other=0.0)
            else:
                k = tl.load(k_ptrs + c_block * kv_jump)
            qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, True, IS_VARLEN)

            qk_proj = tl.maximum(qk - t[:, None], 0)

            ## -- Acc for f, f', f'' --
            if IS_VARLEN:
                v = tl.load(v_ptrs + c_block * kv_jump, mask=kv_mask[:, None], other=0.0)
            else:
                v = tl.load(v_ptrs + c_block * kv_jump)

            if NEED_BACKWARD:
                supp_size += tl.sum(qk_proj, axis=1)
                out2 += tl.dot(qk_proj.to(input_dtype), v, input_precision="ieee")

            out += tl.dot((qk_proj * qk_proj).to(input_dtype), v, input_precision="ieee")
    ## -- save output --
    out_ptrs = OUT + offsets_m[:, None] * H_DIM + offsets_k
    if IS_VARLEN:
        tl.store(out_ptrs, out, mask=q_mask[:, None])
    else:
        tl.store(out_ptrs, out)
    if NEED_BACKWARD:
        out2_ptrs = OUT2 + offsets_m[:, None] * H_DIM + offsets_k
        if IS_VARLEN:
            supp_size = tl.where(q_mask, supp_size, 1.0)
        out2 /= supp_size[:, None]
        if IS_VARLEN:
            tl.store(out2_ptrs, out2, mask=q_mask[:, None])
        else:
            tl.store(out2_ptrs, out2)


@triton.jit
def _bwd_preprocess(
    OUT,
    DO,
    DELTA,
    VARLEN,
    ##
    stride_oh,
    stride_dh,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    BLOCK_M: tl.constexpr,
):
    ## -- grid and offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    qvk_offset = off_hz * stride_oh

    ## -- per-batch seqlen (and early-return for fully-OOB M-blocks) --
    seqlen = N_CTX
    if IS_VARLEN:
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)
        if start_m * BLOCK_M >= seqlen:
            return

    ## -- update offsets --
    DO += qvk_offset + start_m * BLOCK_M * H_DIM
    OUT += qvk_offset + start_m * BLOCK_M * H_DIM
    DELTA += off_hz * stride_dh + start_m * BLOCK_M

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, H_DIM)

    ## -- ptrs --
    do_ptrs = DO + offsets_m[:, None] * H_DIM + offsets_k
    out_ptrs = OUT + offsets_m[:, None] * H_DIM + offsets_k

    if IS_VARLEN:
        o_mask = offsets_m < seqlen - start_m * BLOCK_M
        o = tl.load(out_ptrs, mask=o_mask[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=o_mask[:, None], other=0.0)
    else:
        o = tl.load(out_ptrs)
        do = tl.load(do_ptrs)

    ## -- calculate (o * do).sum()
    delta = tl.sum(o * do, axis=1)

    ## -- save delta --
    delta_ptrs = DELTA + offsets_m
    if IS_VARLEN:
        tl.store(delta_ptrs, delta, mask=o_mask)
    else:
        tl.store(delta_ptrs, delta)


@triton.jit
def transpose_bmask(
    BMASK,
    BMASK_T,
    ##
    N_H: tl.constexpr,
    total_blocks: tl.constexpr,
    number_of_ints32: tl.constexpr,
    ##
    stride_bmh,
):
    off_int = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    off_hz = off_z * N_H + off_h

    ## -- put it on the right (batch, head)
    BMASK += off_hz * stride_bmh
    BMASK_T += off_hz * stride_bmh

    ## -- put it on the right "column"/keys that this block will be responsible --
    BMASK += off_int
    BMASK_T += off_int * 32 * number_of_ints32

    c_bmask = tl.zeros((32,), dtype=tl.int32)
    bits = tl.arange(0, 32)
    total_blocks = max(total_blocks, 32)
    for row in range(total_blocks):
        idx_inside_int = row % 32
        bmask = tl.load(BMASK + row * number_of_ints32)

        bmask_t = bmask >> bits
        bmask_t &= 1
        bmask_t <<= idx_inside_int
        c_bmask += bmask_t

        if row != 0 and (row + 1) % 32 == 0:
            tl.store(BMASK_T + bits * number_of_ints32 + row // 32, c_bmask)
            c_bmask = tl.zeros((32,), dtype=tl.int32)


@_autotune(
    configs=get_config_v2(),
    key=["N_CTX", "H_DIM"],
)
@triton.jit
def _bwd_q_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    TAUS,
    BMASK,
    D,
    VARLEN,
    ##
    sm_scale,
    ##
    stride_qh,
    stride_kvh,
    stride_th,
    stride_bmh,
    stride_dh,
    ##
    H_DIM: tl.constexpr,
    N_H: tl.constexpr,
    N_KV_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    N_CTX: tl.constexpr,
    N_INT32s: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    kv_jump: tl.constexpr = BLOCK_N * H_DIM

    _scalar = 0.5 * sm_scale

    ## -- grid and offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    q_offset = off_hz * stride_qh

    ## -- GQA: K/V share one head across GROUP_SIZE consecutive Q heads --
    off_kv_h = off_h // GROUP_SIZE
    kv_offset = (off_z * N_KV_H + off_kv_h) * stride_kvh

    ## -- per-batch seqlen (and early-return for fully-OOB M-blocks) --
    seqlen = N_CTX
    if IS_VARLEN:
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)
        if start_m * BLOCK_M >= seqlen:
            return

    ## -- update offsets --
    Q += q_offset + start_m * BLOCK_M * H_DIM
    K += kv_offset
    V += kv_offset
    DO += q_offset + start_m * BLOCK_M * H_DIM
    DQ += q_offset + start_m * BLOCK_M * H_DIM

    D += off_hz * stride_dh + start_m * BLOCK_M
    TAUS += off_hz * stride_th + start_m * BLOCK_M
    BMASK += off_hz * stride_bmh + start_m * N_INT32s

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, H_DIM)

    ## -- ptrs --
    q_ptrs = Q + offsets_m[:, None] * H_DIM + offsets_k
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k
    v_ptrs = V + offsets_n[:, None] * H_DIM + offsets_k

    dq_ptrs = DQ + offsets_m[:, None] * H_DIM + offsets_k
    do_ptrs = DO + offsets_m[:, None] * H_DIM + offsets_k

    d_ptrs = D + offsets_m
    t_ptrs = TAUS + offsets_m

    ## -- now let's load q --
    if IS_VARLEN:
        q_mask = offsets_m < seqlen - start_m * BLOCK_M
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0) * _scalar
        do = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0).to(input_dtype)
        delta = tl.load(d_ptrs, mask=q_mask, other=0.0)
        t = tl.load(t_ptrs, mask=q_mask, other=0.0)
    else:
        q = tl.load(q_ptrs) * _scalar
        do = tl.load(do_ptrs).to(input_dtype)
        delta = tl.load(d_ptrs)
        t = tl.load(t_ptrs)
    q = q.to(input_dtype)

    ## -- to accumulate dq --
    dq = tl.zeros([BLOCK_M, H_DIM], dtype=tl.float32)
    for n_int in tl.static_range(N_INT32s):
        bmask = tl.load(BMASK + n_int).cast(tl.int32, bitcast=True)
        total_blocks = popc(bmask)
        base_block = 32 * n_int

        for n_blk in range(0, total_blocks):
            c_block = base_block + get_next_block(bmask, n_blk)

            if IS_VARLEN:
                kv_mask = (c_block * BLOCK_N + offsets_n) < seqlen
                v = tl.load(v_ptrs + c_block * kv_jump, mask=kv_mask[:, None], other=0.0).to(input_dtype)
            else:
                v = tl.load(v_ptrs + c_block * kv_jump).to(input_dtype)

            ## -- compute dp and ds --
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")

            if IS_VARLEN:
                k = tl.load(k_ptrs + c_block * kv_jump, mask=kv_mask[:, None], other=0.0)
            else:
                k = tl.load(k_ptrs + c_block * kv_jump)
            qk = get_qk(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, True, IS_VARLEN)

            ## -- calculate u, it's all we need --
            u = tl.maximum(qk - t[:, None], 0.0)
            ds = u * (dp - delta[:, None])

            ## -- compute dq --
            dq += tl.dot(ds.to(input_dtype), k, input_precision="ieee")

    dq *= sm_scale
    if IS_VARLEN:
        tl.store(dq_ptrs, dq.to(input_dtype), mask=q_mask[:, None])
    else:
        tl.store(dq_ptrs, dq.to(input_dtype))


@_autotune(
    configs=get_config_v2(),
    key=["N_CTX", "H_DIM"],
)
@triton.jit
def _bwd_kv_kernel(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    TAUS,
    BMASK,
    D,
    VARLEN,
    ##
    sm_scale,
    ##
    stride_qh,
    stride_kvh,
    stride_th,
    stride_bmh,
    stride_dh,
    ##
    H_DIM: tl.constexpr,
    N_H: tl.constexpr,
    N_KV_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    N_CTX: tl.constexpr,
    N_INT32s: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    q_jump: tl.constexpr = BLOCK_M * H_DIM

    _scalar = 0.5 * sm_scale

    ## -- grid and offsets: one program owns one (KV block, KV head, batch) --
    start_n = tl.program_id(0)
    off_kv_h = tl.program_id(1)
    off_z = tl.program_id(2)

    ## -- per-batch seqlen (and early-return for fully-OOB KV-blocks) --
    seqlen = N_CTX
    if IS_VARLEN:
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)
        if start_n * BLOCK_N >= seqlen:
            return

    ## -- KV-side base pointers: constant across the per-Q-head group loop --
    kv_offset = (off_z * N_KV_H + off_kv_h) * stride_kvh
    K += kv_offset + start_n * BLOCK_N * H_DIM
    V += kv_offset + start_n * BLOCK_N * H_DIM
    DK += kv_offset + start_n * BLOCK_N * H_DIM
    DV += kv_offset + start_n * BLOCK_N * H_DIM

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, H_DIM)

    ## -- KV-side ptrs --
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k
    v_ptrs = V + offsets_n[:, None] * H_DIM + offsets_k

    ## -- load K/V once; reused across all GROUP_SIZE Q heads in this group --
    if IS_VARLEN:
        kv_mask = offsets_n < seqlen - start_n * BLOCK_N
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
    else:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
    v = tl.trans(v.to(input_dtype))
    k *= _scalar
    k = tl.trans(k.to(input_dtype))

    ## -- dk and dv in SRAM (accumulate across the group) --
    dk = tl.zeros([BLOCK_N, H_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, H_DIM], dtype=tl.float32)

    for off_h_in_group in tl.static_range(GROUP_SIZE):
        off_h = off_kv_h * GROUP_SIZE + off_h_in_group
        off_hz = off_z * N_H + off_h
        q_offset = off_hz * stride_qh

        q_ptrs = Q + q_offset + offsets_m[:, None] * H_DIM + offsets_k
        do_ptrs = DO + q_offset + offsets_m[:, None] * H_DIM + offsets_k
        d_ptrs = D + off_hz * stride_dh + offsets_m
        t_ptrs = TAUS + off_hz * stride_th + offsets_m
        bmask_base = BMASK + off_hz * stride_bmh + start_n * N_INT32s

        for n_int in tl.static_range(N_INT32s):
            bmask = tl.load(bmask_base + n_int).cast(tl.int32, bitcast=True)
            total_blocks = popc(bmask)
            base_block = 32 * n_int
            for n_blk in range(0, total_blocks):
                c_block = base_block + get_next_block(bmask, n_blk)

                if IS_VARLEN:
                    q_mask = (c_block * BLOCK_M + offsets_m) < seqlen
                    q = tl.load(q_ptrs + c_block * q_jump, mask=q_mask[:, None], other=0.0).to(input_dtype)
                    t = tl.load(t_ptrs + c_block * BLOCK_M, mask=q_mask, other=0.0)
                else:
                    q = tl.load(q_ptrs + c_block * q_jump).to(input_dtype)
                    t = tl.load(t_ptrs + c_block * BLOCK_M)

                ## -- calculate scores --
                qk = get_qk_t(c_block, q, k, seqlen, BLOCK_M, BLOCK_N, True, IS_VARLEN)

                ## -- activation scores --
                qk_act = tl.maximum(qk - t[:, None], 0.0)

                if IS_VARLEN:
                    do = tl.load(do_ptrs + c_block * q_jump, mask=q_mask[:, None], other=0.0).to(input_dtype)
                else:
                    do = tl.load(do_ptrs + c_block * q_jump).to(input_dtype)

                ## -- compute dv --
                dv += tl.dot(tl.trans((qk_act * qk_act).to(input_dtype)), do, input_precision="ieee")

                ## -- load delta --
                if IS_VARLEN:
                    delta = tl.load(d_ptrs + c_block * BLOCK_M, mask=q_mask, other=0.0)
                else:
                    delta = tl.load(d_ptrs + c_block * BLOCK_M)

                ## -- compute dp --
                dp = tl.dot(do, v, input_precision="ieee")

                ## -- compute ds --
                ds = qk_act * (dp - delta[:, None])
                ds = ds.to(input_dtype)

                ## -- compute dk --
                dk += tl.dot(tl.trans(ds), q, input_precision="ieee")

    dk *= sm_scale

    ## -- dk and dv pointer --
    dv_ptrs = DV + offsets_n[:, None] * H_DIM + offsets_k
    dk_ptrs = DK + offsets_n[:, None] * H_DIM + offsets_k
    if IS_VARLEN:
        tl.store(dk_ptrs, dk.to(input_dtype), mask=kv_mask[:, None])
        tl.store(dv_ptrs, dv.to(input_dtype), mask=kv_mask[:, None])
    else:
        tl.store(dk_ptrs, dk.to(input_dtype))
        tl.store(dv_ptrs, dv.to(input_dtype))


def ASSERT_CONTIGUOUS(*inputs, msg="Inputs are not contiguous."):
    assert all(t.is_contiguous() for t in inputs), msg


class _sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, niter=10, varlen=None):
        # shape constraints
        B, N_H, N_CTX, H_DIM = q.shape
        BMASK_TILE = 64

        assert H_DIM in {16, 32, 64, 128, 256}

        ## -- GQA: K/V may have fewer heads than Q (N_H = GROUP_SIZE * N_KV_H) --
        assert k.shape == v.shape, f"K and V must share shape; got k.shape={tuple(k.shape)} v.shape={tuple(v.shape)}"
        N_KV_H = k.shape[1]
        assert N_KV_H >= 1 and N_H % N_KV_H == 0, f"N_H ({N_H}) must be a positive multiple of N_KV_H ({N_KV_H})."
        assert k.shape[0] == B and k.shape[2] == N_CTX and k.shape[3] == H_DIM, (
            f"K/V must have shape (B={B}, N_KV_H, N_CTX={N_CTX}, H_DIM={H_DIM}); " f"got {tuple(k.shape)}."
        )
        GROUP_SIZE = N_H // N_KV_H

        ## -- constants and flags --
        device = q.device
        sm_scale = 1 / sqrt(H_DIM)
        IS_VARLEN = varlen is not None

        ## -- length constraints --
        assert N_CTX >= BMASK_TILE, f"N_CTX={N_CTX} must be at least BMASK_TILE={BMASK_TILE}."
        if not IS_VARLEN:
            assert (N_CTX & (N_CTX - 1)) == 0, "If varlen is None, N_CTX must be a power of two."
        else:
            assert (
                varlen.dim() == 1 and varlen.shape[0] == B
            ), f"varlen must be a 1-D tensor of shape (B={B},); got {tuple(varlen.shape)}."
            assert varlen.dtype in (torch.int32, torch.int64), f"varlen must be int32/int64; got {varlen.dtype}."
            assert varlen.device == device, "varlen must be on the same device as q."
            assert varlen.is_contiguous(), "varlen must be contiguous."

        ASSERT_CONTIGUOUS(q, k, v, msg="Q, K and/or V are not contiguous.")

        ## -- tensors --
        taus = torch.empty((B, N_H, N_CTX), device=device, dtype=torch.float32)  # fmt: skip

        ######################################################
        #   First pass: This just get the maximum row-wise   #
        #          with the goal of initializing tau         #
        ######################################################

        ## -- grid: get_tau --
        grid_tau = lambda META: (triton.cdiv(N_CTX, META["BLOCK_M"]), N_H, B)
        _get_tau[grid_tau](
            q,
            k,
            taus,
            varlen,
            ##
            sm_scale,
            ##
            N_H,
            N_KV_H,
            GROUP_SIZE,
            H_DIM,
            N_CTX,
            IS_VARLEN,
            ##
            q.stride(1),
            k.stride(1),
            taus.stride(1),
        )

        ######################################################
        #           Second pass: Build the histogram         #
        ######################################################

        BLOCK_M, BLOCK_N = 64, 64
        BINS = _select_bins(N_CTX, BLOCK_N)
        grid_tau_v2 = (triton.cdiv(N_CTX, BLOCK_M), N_H, B)
        _get_tau_v2[grid_tau_v2](
            q,
            k,
            taus,
            varlen,
            ##
            sm_scale,
            ##
            N_H,
            N_KV_H,
            GROUP_SIZE,
            H_DIM,
            N_CTX,
            IS_VARLEN,
            ##
            q.stride(1),
            k.stride(1),
            taus.stride(1),
            ##
            BLOCK_M,
            BLOCK_N,
            BINS,
        )

        ######################################################
        #             Third pass: Build the bmask            #
        #              and do one iteration of HB            #
        ######################################################

        BLOCK_M, BLOCK_N = BMASK_TILE, BMASK_TILE
        number_of_mblocks = triton.cdiv(N_CTX, BLOCK_M)
        number_of_ints32 = triton.cdiv(triton.cdiv(N_CTX, BLOCK_N), 32)
        # Round mblocks up to a multiple of 32 so transpose_bmask's per-32-row
        # store guard fires for every group (no kernel change needed).
        mblocks_padded = max(32, ((number_of_mblocks + 31) // 32) * 32)
        bmask = torch.zeros((B, N_H, mblocks_padded, number_of_ints32), device=device, dtype=torch.int32)  # fmt: skip

        grid_tau_v3 = (triton.cdiv(N_CTX, BLOCK_M), N_H, B)
        _get_tau_v3[grid_tau_v3](
            q,
            k,
            taus,
            bmask,
            varlen,
            ##
            sm_scale,
            niter,
            ##
            N_H,
            N_KV_H,
            GROUP_SIZE,
            H_DIM,
            N_CTX,
            number_of_ints32,
            IS_VARLEN,
            ##
            q.stride(1),
            k.stride(1),
            taus.stride(1),
            bmask.stride(1),
            ##
            BLOCK_M,
            BLOCK_N,
            BINS,
        )

        ######################################################
        #             Fourth pass: Get output                #
        ######################################################

        if IS_VARLEN:
            out = torch.zeros_like(q)
            out2 = torch.zeros_like(q)
        else:
            out = torch.empty_like(q)
            out2 = torch.empty_like(q)
        _get_output[grid_tau_v3](
            q,
            k,
            v,
            out,
            out2,
            taus,
            bmask,
            varlen,
            ##
            sm_scale,
            True,
            ##
            N_H,
            N_KV_H,
            GROUP_SIZE,
            H_DIM,
            N_CTX,
            number_of_ints32,
            IS_VARLEN,
            ##
            q.stride(1),
            k.stride(1),
            taus.stride(1),
            bmask.stride(1),
            ##
            BLOCK_M,
            BLOCK_N,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            out2,
            taus,
            bmask,
        )
        ctx.sm_scale = sm_scale
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.varlen = varlen
        ctx.IS_VARLEN = IS_VARLEN
        ctx.mblocks_padded = mblocks_padded
        ctx.N_KV_H = N_KV_H
        ctx.GROUP_SIZE = GROUP_SIZE

        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, taus, bmask = ctx.saved_tensors

        B, N_H, N_CTX, H_DIM = q.shape
        device = q.device
        sm_scale = ctx.sm_scale
        varlen = ctx.varlen
        IS_VARLEN = ctx.IS_VARLEN
        mblocks_padded = ctx.mblocks_padded
        N_KV_H = ctx.N_KV_H
        GROUP_SIZE = ctx.GROUP_SIZE

        do = do.contiguous()

        ## -- grid: preprocess --
        PRE_BLOCK = 128
        pre_grid = (triton.cdiv(N_CTX, PRE_BLOCK), N_H, B)

        delta = torch.zeros((B, N_H, N_CTX), device=device, dtype=torch.float32)  # fmt: skip

        _bwd_preprocess[pre_grid](
            o,
            do,
            delta,
            varlen,
            ##
            o.stride(1),
            delta.stride(1),
            ##
            N_H,
            H_DIM,
            N_CTX,
            IS_VARLEN,
            ##
            PRE_BLOCK,
            ##
            num_warps=16,
            num_stages=1,
        )
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        if IS_VARLEN:
            dq = torch.zeros_like(q).contiguous()
        else:
            dq = torch.empty_like(q).contiguous()
        grid_dq = (triton.cdiv(N_CTX, BLOCK_M), N_H, B)
        number_of_ints32 = triton.cdiv(triton.cdiv(N_CTX, BLOCK_N), 32)

        _bwd_q_kernel[grid_dq](
            q,
            k,
            v,
            do,
            dq,
            taus,
            bmask,
            delta,
            varlen,
            ##
            sm_scale,
            ##
            q.stride(1),
            k.stride(1),
            taus.stride(1),
            bmask.stride(1),
            delta.stride(1),
            ##
            H_DIM,
            N_H,
            N_KV_H,
            GROUP_SIZE,
            N_CTX,
            number_of_ints32,
            IS_VARLEN,
            ##
            BLOCK_M,
            BLOCK_N,
        )

        bmask_kv = torch.zeros_like(bmask)
        grid_transpose = (number_of_ints32, N_H, B)

        transpose_bmask[grid_transpose](bmask, bmask_kv, N_H, mblocks_padded, number_of_ints32, bmask.stride(1))

        if IS_VARLEN:
            dk = torch.zeros_like(k).contiguous()
            dv = torch.zeros_like(v).contiguous()
        else:
            dk = torch.empty_like(k).contiguous()
            dv = torch.empty_like(v).contiguous()
        grid_dkdv = (triton.cdiv(N_CTX, BLOCK_N), N_KV_H, B)
        _bwd_kv_kernel[grid_dkdv](
            q,
            k,
            v,
            do,
            dk,
            dv,
            taus,
            bmask_kv,
            delta,
            varlen,
            ##
            sm_scale,
            ##
            q.stride(1),
            k.stride(1),
            taus.stride(1),
            bmask_kv.stride(1),
            delta.stride(1),
            ##
            H_DIM,
            N_H,
            N_KV_H,
            GROUP_SIZE,
            N_CTX,
            number_of_ints32,
            IS_VARLEN,
            ##
            BLOCK_M,
            BLOCK_N,
        )

        return dq, dk, dv, None, None


def sparse_attn(q, k, v, niter=1, varlen=None):
    return _sparse_attention.apply(q, k, v, niter, varlen)


def _install_callable_module():
    import inspect
    import sys
    import types

    class _CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return sparse_attn(*args, **kwargs)

    module = sys.modules[__name__]
    module.__class__ = _CallableModule
    module.__signature__ = inspect.signature(sparse_attn)


_install_callable_module()
