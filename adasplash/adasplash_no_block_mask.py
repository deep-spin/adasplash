from math import sqrt

import torch

import triton
import triton.language as tl


@triton.jit
def halley_bisect_update(t, t_lo, t_hi, acc_0, acc_1, acc_2, coeff_0, coeff_1):
    EPS: tl.constexpr = 1e-6

    ## -- function eval --
    ff = tl.sum(acc_0, axis=1) - 1.0
    ## -- first derivative --
    df = -coeff_0 * tl.sum(acc_1, axis=1)
    ## -- second derivative --
    ddf = coeff_0 * coeff_1 * tl.sum(acc_2, axis=1)

    ## -- update bounds --
    t_lo = tl.where((ff > 0), t, t_lo)
    t_hi = tl.where((ff < 0), t, t_hi)

    ## -- halley's update --
    new_t = t - (2 * ff * df) / (2 * df * df - ff * ddf)

    ## -- is halley's inside the bounds? --
    is_good = (new_t > t_lo - EPS) & (new_t < t_hi + EPS)
    t = tl.where(is_good, new_t, 0.5 * (t_lo + t_hi))

    return t, t_lo, t_hi


@triton.jit
def _get_tau(
    Q,
    K,
    TAUS,
    VARLEN,
    ##
    alpha,
    sm_scale,
    NITER,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    N_CTX,
    ##
    stride_qh,
    stride_th,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    kv_jump: tl.constexpr = BLOCK_N * H_DIM

    ## -- some coefficients --
    _scalar = (alpha - 1) * sm_scale
    coeff_0 = 1 / (alpha - 1)
    coeff_1 = 1 / (alpha - 1) - 1
    coeff_2 = 1 / (alpha - 1) - 2

    ## -- offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    qvk_offset = off_hz * stride_qh

    ## -- update pointer offsets --
    Q += qvk_offset + start_m * BLOCK_M * H_DIM
    K += qvk_offset
    TAUS += off_hz * stride_th + start_m * BLOCK_M

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, H_DIM)

    ## -- ptrs --
    q_ptrs = Q + offsets_m[:, None] * H_DIM + offsets_k
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k
    t_ptrs = TAUS + offsets_m

    seqlen = N_CTX
    if IS_VARLEN:
        ## -- let's check what is the seqlen of the current batch --
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)

        ## -- in case we're already beyond the length, let's return, nothing to be done here --
        if start_m * BLOCK_M >= seqlen:
            return

    up_to_seqlen = seqlen
    if IS_CAUSAL:
        ## -- in case it's causal, we only want to do it up to the diagonal --
        up_to_seqlen = tl.minimum((start_m + 1) * BLOCK_M, seqlen)

    ## -- now let's load q --
    if IS_VARLEN:
        q_mask = offsets_m < seqlen - start_m * BLOCK_M
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0) * _scalar
    else:
        q = tl.load(q_ptrs) * _scalar
    q = q.to(input_dtype)

    ## -- how many blocks of k do we need to go (encoder:full, decoder:till the diagonal basically) --
    valid_nblocks = tl.cdiv(up_to_seqlen, BLOCK_N)
    mvals = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)

    if IS_CAUSAL:
        ## -- get the idxs of q --
        q_idxs = offsets_m + start_m * BLOCK_M

    ## -- let's fill mvals --
    for c_block in range(0, valid_nblocks):

        if IS_VARLEN or IS_CAUSAL:
            ## -- get the idxs of k --
            k_idxs = c_block * BLOCK_N + offsets_n

        if IS_CAUSAL:
            ## -- build causal mask --
            causal_mask = q_idxs[:, None] >= k_idxs[None, :]

        ## -- load k --
        if IS_VARLEN:
            k_mask = k_idxs < seqlen
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0).to(input_dtype)
        else:
            k = tl.load(k_ptrs).to(input_dtype)

        ## -- compute mvals --
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")

        ## -- if it's causal we need to hide everything where q_idx < k_idx
        if IS_CAUSAL:
            qk = tl.where(causal_mask, qk, float("-inf"))

        ## -- if it's varlen we need to hide everything in the block that does not "exist" --
        if IS_VARLEN:
            qk = tl.where(k_mask, qk, float("-inf"))

        ## -- get max row-wise --
        c_mvals = tl.max(qk, axis=1)

        ## -- update mvals --
        mvals = tl.maximum(mvals, c_mvals)

        ## -- increment pointers --
        k_ptrs += BLOCK_N * H_DIM

    if not IS_CAUSAL:
        q_idxs = seqlen

    ## -- get tau and its bounds --
    # t_hi = max(s) - n^(1-α)
    # t_lo = max(s) - 1
    t_hi = mvals - tl.exp2((1 - alpha) * tl.log2(1.0 + q_idxs))
    t_lo = mvals - 1
    t = 0.5 * (t_lo + t_hi)

    for _ in range(NITER):
        ## -- reset ptr --
        k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k

        ## -- accumulate --
        acc_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for c_block in range(0, valid_nblocks):

            if IS_VARLEN or IS_CAUSAL:
                ## -- get the idxs of k --
                k_idxs = c_block * BLOCK_N + offsets_n

            if IS_CAUSAL:
                ## -- build causal mask --
                causal_mask = q_idxs[:, None] >= k_idxs[None, :]

            if IS_VARLEN:
                ## -- load k --
                k_mask = k_idxs < seqlen
                k = tl.load(k_ptrs, mask=k_mask[:, None], other=0).to(input_dtype)
            else:
                k = tl.load(k_ptrs)

            ## -- calculate scores --
            qk = tl.dot(q, tl.trans(k), input_precision="ieee")

            qk_mask = qk > t[:, None]
            if IS_CAUSAL:
                qk_mask &= causal_mask
            if IS_VARLEN and not IS_CAUSAL:
                qk_mask &= k_mask

            qk_log = tl.log2(qk - t[:, None])

            ## -- Acc for f, f', f'' --
            acc_0 += tl.where(qk_mask, tl.exp2(qk_log * coeff_0), 0)
            acc_1 += tl.where(qk_mask, tl.exp2(qk_log * coeff_1), 0)
            acc_2 += tl.where(qk_mask, tl.exp2(qk_log * coeff_2), 0)

            ## -- increment pointers --
            k_ptrs += kv_jump

        t, t_lo, t_hi = halley_bisect_update(t, t_lo, t_hi, acc_0, acc_1, acc_2, coeff_0, coeff_1)  # fmt: skip

    ## -- store tau and also mask for over seqlen entries --
    if IS_VARLEN:
        tl.store(t_ptrs, t, mask=q_mask)
    else:
        tl.store(t_ptrs, t)


@triton.jit
def _get_output(
    Q,
    K,
    V,
    OUT,
    OUT2,
    TAUS,
    VARLEN,
    ##
    alpha,
    sm_scale,
    NEED_BACKWARD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    N_CTX,
    ##
    stride_qh,
    stride_th,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    kv_jump: tl.constexpr = BLOCK_N * H_DIM

    _scalar = (alpha - 1) * sm_scale
    coeff_0 = 1 / (alpha - 1)
    coeff_f = (2 - alpha) / (alpha - 1)

    ## -- grid and offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    qvk_offset = off_hz * stride_qh

    ## -- update offsets --
    Q += qvk_offset + start_m * BLOCK_M * H_DIM
    K += qvk_offset
    V += qvk_offset
    OUT += qvk_offset + start_m * BLOCK_M * H_DIM

    if NEED_BACKWARD:
        OUT2 += qvk_offset + start_m * BLOCK_M * H_DIM

    TAUS += off_hz * stride_th + start_m * BLOCK_M

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, H_DIM)

    ## -- ptrs --
    q_ptrs = Q + offsets_m[:, None] * H_DIM + offsets_k
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k
    v_ptrs = V + offsets_n[:, None] * H_DIM + offsets_k
    t_ptrs = TAUS + offsets_m

    seqlen = N_CTX
    if IS_VARLEN:
        ## -- let's check what is the seqlen of the current batch --
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)

        ## -- in case we're already beyond the length, let's return, nothing to be done here --
        if start_m * BLOCK_M >= seqlen:
            return

    up_to_seqlen = seqlen
    if IS_CAUSAL:
        ## -- in case it's causal, we only want to do it up to the diagonal --
        up_to_seqlen = tl.minimum((start_m + 1) * BLOCK_M, seqlen)

    if IS_VARLEN:
        ## -- now let's load q --
        q_mask = offsets_m < seqlen - start_m * BLOCK_M
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0) * _scalar

        ## -- load tau calculated from previous kernel --
        t = tl.load(t_ptrs, q_mask, other=0)
    else:
        q = tl.load(q_ptrs) * _scalar
        t = tl.load(t_ptrs)
    q = q.to(input_dtype)

    ## -- how many blocks of k do we need to go (encoder:full, decoder:till the diagonal basically) --
    valid_nblocks = tl.cdiv(up_to_seqlen, BLOCK_N)

    if IS_CAUSAL:
        ## -- get the idxs of q --
        q_idxs = offsets_m + start_m * BLOCK_M

    ## -- compute output --
    acc = tl.zeros([BLOCK_M, H_DIM], dtype=tl.float32)
    if NEED_BACKWARD:
        acc2 = tl.zeros([BLOCK_M, H_DIM], dtype=tl.float32)
        supp_size = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for c_block in range(0, valid_nblocks):

        if IS_VARLEN or IS_CAUSAL:
            ## -- get the idxs of k --
            k_idxs = c_block * BLOCK_N + offsets_n

        if IS_CAUSAL:

            ## -- build causal_mask --
            causal_mask = q_idxs[:, None] >= k_idxs[None, :]

        ## -- load k --
        if IS_VARLEN:
            kv_mask = k_idxs < seqlen
            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0).to(input_dtype)
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0).to(input_dtype)
        else:
            k = tl.load(k_ptrs).to(input_dtype)
            v = tl.load(v_ptrs).to(input_dtype)

        ## -- compute scores --
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")

        qk_mask = qk > t[:, None]
        if IS_CAUSAL:
            qk_mask &= causal_mask
        if IS_VARLEN and not IS_CAUSAL:
            qk_mask &= kv_mask

        # -- calculate entmax(qk) --
        qk_log = tl.log2(qk - t[:, None])
        qk_act = tl.where(qk_mask, tl.exp2(qk_log * coeff_0), 0)

        ## -- load v --
        acc += tl.dot(qk_act.to(input_dtype), v, input_precision="ieee")

        if NEED_BACKWARD:
            u_i = tl.where(qk_mask, tl.exp2(qk_log * coeff_f), 0)
            acc2 += tl.dot(u_i.to(input_dtype), v, input_precision="ieee")
            supp_size += tl.sum(u_i, axis=1)

        ## -- increment pointers --
        k_ptrs += kv_jump
        v_ptrs += kv_jump

    out_ptrs = OUT + offsets_m[:, None] * H_DIM + offsets_k
    if NEED_BACKWARD:
        out2_ptrs = OUT2 + offsets_m[:, None] * H_DIM + offsets_k
    if IS_VARLEN:
        ## -- save main output --
        tl.store(out_ptrs, acc, mask=q_mask[:, None])

        ## -- save output for backward --
        if NEED_BACKWARD:
            acc2 /= supp_size[:, None]
            out2_ptrs = OUT2 + offsets_m[:, None] * H_DIM + offsets_k
            tl.store(out2_ptrs, acc2, mask=q_mask[:, None])
    else:
        ## -- save main output --
        tl.store(out_ptrs, acc)

        ## -- save output for backward --
        if NEED_BACKWARD:
            acc2 /= supp_size[:, None]
            tl.store(out2_ptrs, acc2)


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
    IS_VARLEN: tl.constexpr,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    N_CTX,
    ##
    BLOCK_M: tl.constexpr,
):
    ## -- grid and offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    qvk_offset = off_hz * stride_oh

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

    ## -- get the sequence length of this batch --
    seqlen = N_CTX
    if IS_VARLEN:
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)
        if start_m * BLOCK_M >= seqlen:
            return

    ## -- we don't need to load padded tokens --
    if IS_VARLEN:
        o_mask = offsets_m < seqlen - start_m * BLOCK_M
        o = tl.load(out_ptrs, mask=o_mask[:, None], other=0)
        do = tl.load(do_ptrs, mask=o_mask[:, None], other=0)
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
def _bwd_kv_kernel(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    TAUS,
    VARLEN,
    D,
    ##
    alpha,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    stride_qh,
    stride_th,
    stride_dh,
    ##
    H_DIM: tl.constexpr,
    N_H: tl.constexpr,
    N_CTX,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    q_jump: tl.constexpr = BLOCK_M * H_DIM

    _scalar = (alpha - 1) * sm_scale
    coeff_0 = 1 / (alpha - 1)
    coeff_f = (2 - alpha) / (alpha - 1)

    ## -- grid and offsets --
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    qkv_offset = off_hz * stride_qh

    ## -- update offsets --
    Q += qkv_offset
    K += qkv_offset + start_n * BLOCK_N * H_DIM
    V += qkv_offset + start_n * BLOCK_N * H_DIM
    DO += qkv_offset
    DK += qkv_offset + start_n * BLOCK_N * H_DIM
    DV += qkv_offset + start_n * BLOCK_N * H_DIM

    D += off_hz * stride_dh
    TAUS += off_hz * stride_th

    ## -- create local offsets --
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, H_DIM)

    ## -- ptrs --
    q_ptrs = Q + offsets_m[:, None] * H_DIM + offsets_k
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k
    v_ptrs = V + offsets_n[:, None] * H_DIM + offsets_k
    do_ptrs = DO + offsets_m[:, None] * H_DIM + offsets_k

    d_ptrs = D + offsets_m
    t_ptrs = TAUS + offsets_m

    seqlen = N_CTX
    if IS_VARLEN:
        ## -- let's check what is the seqlen of the current batch --
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)

        ## -- in case we're already beyond the length, let's return, nothing to be done here --
        if start_n * BLOCK_N >= seqlen:
            return

    start_block = 0
    valid_mblocks = tl.cdiv(seqlen, BLOCK_M)
    if IS_CAUSAL:
        ## -- in the case of dkdv we start at the diagonal and go to the end --
        start_block = (start_n * BLOCK_N) // BLOCK_M

    ## -- load k and v --
    if IS_VARLEN:
        kv_mask = offsets_n < seqlen - start_n * BLOCK_N
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0)
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0)
    else:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
    k *= _scalar
    v = tl.trans(v.to(input_dtype))
    k = tl.trans(k.to(input_dtype))

    ## -- dk and dv in SRAM --
    dk = tl.zeros([BLOCK_N, H_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, H_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        ## -- jump everything to the beginning of the diagonal --
        q_ptrs += start_block * q_jump
        do_ptrs += start_block * q_jump
        t_ptrs += start_block * BLOCK_M
        d_ptrs += start_block * BLOCK_M

    if IS_CAUSAL:
        ## -- create kv_idxs --
        kv_idxs = offsets_n + start_n * BLOCK_N

    for c_block in range(start_block, valid_mblocks):

        if IS_VARLEN or IS_CAUSAL:
            ## -- get the idxs of q --
            q_idxs = c_block * BLOCK_M + offsets_m

        ## -- load q --
        if IS_VARLEN:
            q_mask = q_idxs < seqlen
            q = tl.load(q_ptrs, mask=q_mask[:, None], other=0).to(input_dtype)
        else:
            q = tl.load(q_ptrs).to(input_dtype)

        ## -- load tau beforehand --
        if IS_VARLEN:
            t = tl.load(t_ptrs, mask=q_mask, other=0)
        else:
            t = tl.load(t_ptrs)

        ## -- calculate scores --
        qk = tl.dot(q, k, input_precision="ieee")

        if IS_CAUSAL:
            ## -- build causal mask --
            causal_mask = q_idxs[:, None] >= kv_idxs[None, :]

        ## -- get qk_mask, especial caution here --
        qk_mask = qk > t[:, None]
        if IS_CAUSAL:
            qk_mask &= causal_mask
        if IS_VARLEN:
            qk_mask &= q_mask[:, None]

        ## -- activation scores --
        qk_log = tl.log2(qk - t[:, None])
        qk_act = tl.where(qk_mask, tl.exp2(qk_log * coeff_0), 0).to(input_dtype)

        ## -- load do --
        if IS_VARLEN:
            do = tl.load(do_ptrs, mask=q_mask[:, None], other=0).to(input_dtype)
        else:
            do = tl.load(do_ptrs).to(input_dtype)

        ## -- compute dv --
        dv += tl.dot(tl.trans(qk_act), do, input_precision="ieee")

        ## -- load delta --
        if IS_VARLEN:
            delta = tl.load(d_ptrs, mask=q_mask, other=0)
        else:
            delta = tl.load(d_ptrs)

        ## -- compute dp --
        dp = tl.dot(do, v, input_precision="ieee")

        ## -- calculate u --
        u = tl.where(qk_mask, tl.exp2(qk_log * coeff_f), 0)

        ## -- compute ds --
        ds = u * (dp - delta[:, None])
        ds = ds.to(input_dtype)

        ## -- compute dk --
        dk += tl.dot(tl.trans(ds), q, input_precision="ieee")

        ## -- increment pointers --
        q_ptrs += q_jump
        do_ptrs += q_jump
        t_ptrs += BLOCK_M
        d_ptrs += BLOCK_M

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


@triton.jit
def _bwd_q_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    TAUS,
    VARLEN,
    D,
    ##
    alpha,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    stride_qh,
    stride_th,
    stride_dh,
    ##
    H_DIM: tl.constexpr,
    N_H: tl.constexpr,
    N_CTX,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    kv_jump: tl.constexpr = BLOCK_N * H_DIM

    _scalar = (alpha - 1) * sm_scale
    coeff_f = (2 - alpha) / (alpha - 1)

    ## -- grid and offsets --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * N_H + off_h
    qkv_offset = off_hz * stride_qh

    ## -- update offsets --
    Q += qkv_offset + start_m * BLOCK_M * H_DIM
    K += qkv_offset
    V += qkv_offset
    DO += qkv_offset + start_m * BLOCK_M * H_DIM
    DQ += qkv_offset + start_m * BLOCK_M * H_DIM

    D += off_hz * stride_dh + start_m * BLOCK_M
    TAUS += off_hz * stride_th + start_m * BLOCK_M

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

    seqlen = N_CTX
    if IS_VARLEN:
        ## -- let's check what is the seqlen of the current batch --
        seqlen = tl.load(VARLEN + off_z).to(tl.int32)

        ## -- in case we're already beyond the length, let's return, nothing to be done here --
        if start_m * BLOCK_M >= seqlen:
            return

    up_to_seqlen = seqlen
    if IS_CAUSAL:
        ## -- in case it's causal, we only want to do it up to the diagonal --
        up_to_seqlen = tl.minimum((start_m + 1) * BLOCK_M, seqlen)

    if IS_VARLEN:
        ## -- now let's load q --
        q_mask = offsets_m < seqlen - start_m * BLOCK_M
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0) * _scalar
        q = q.to(input_dtype)

        ## -- load do --
        do = tl.load(do_ptrs, mask=q_mask[:, None], other=0).to(input_dtype)

        ## -- load delta and tau --
        delta = tl.load(d_ptrs, q_mask, other=0)
        t = tl.load(t_ptrs, q_mask, other=0)
    else:
        ## -- now let's load q --
        q = tl.load(q_ptrs) * _scalar
        q = q.to(input_dtype)

        ## -- load do --
        do = tl.load(do_ptrs).to(input_dtype)

        ## -- load delta and tau --
        delta = tl.load(d_ptrs)
        t = tl.load(t_ptrs)

    ## -- how many blocks of k do we need to go (till the diagonal basically) --
    valid_nblocks = tl.cdiv(up_to_seqlen, BLOCK_N)

    if IS_CAUSAL:
        ## -- get the idxs of q --
        q_idxs = offsets_m + start_m * BLOCK_M

    ## -- to accumulate dq --
    dq = tl.zeros([BLOCK_M, H_DIM], dtype=tl.float32)

    for c_block in range(0, valid_nblocks):

        if IS_VARLEN or IS_CAUSAL:
            ## -- get the idxs of k --
            k_idxs = c_block * BLOCK_N + offsets_n

        if IS_CAUSAL:
            ## -- build causal_mask --
            causal_mask = q_idxs[:, None] >= k_idxs[None, :]

        ## -- load k --
        if IS_VARLEN:
            kv_mask = k_idxs < seqlen
            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0).to(input_dtype)
        else:
            k = tl.load(k_ptrs).to(input_dtype)

        ## -- compute scores --
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")

        qk_mask = qk > t[:, None]
        if IS_CAUSAL:
            qk_mask &= causal_mask
        if IS_VARLEN and not IS_CAUSAL:
            qk_mask &= kv_mask

        ## -- load v now --
        if IS_VARLEN:
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0).to(input_dtype)
        else:
            v = tl.load(v_ptrs).to(input_dtype)

        ## -- calculate u, it's all we need --
        qk_log = tl.log2(qk - t[:, None])
        u = tl.where(qk_mask, tl.exp2(qk_log * coeff_f), 0).to(input_dtype)

        ## -- compute dp and ds --
        dp = tl.dot(do.to(input_dtype), tl.trans(v), input_precision="ieee")

        ds = u * (dp - delta[:, None])

        ## -- compute dq --
        dq += tl.dot(ds.to(input_dtype), k, input_precision="ieee")

        ## -- increment ptrs --
        k_ptrs += kv_jump
        v_ptrs += kv_jump

    dq *= sm_scale
    if IS_VARLEN:
        tl.store(dq_ptrs, dq.to(input_dtype), mask=q_mask[:, None])
    else:
        tl.store(dq_ptrs, dq.to(input_dtype))


def ASSERT_CONTIGUOUS(*inputs, msg="Inputs are not contiguous."):
    assert all(t.is_contiguous() for t in inputs), msg


def ASSERT_VARLEN(varlen, N_CTX):
    if varlen is None:
        assert N_CTX.bit_count() == 1, "If varlen is not used, the context length must be a power of two."  # fmt: skip
    else:
        assert varlen.dim() == 1, "varlen must be a one-dimensional tensor."


@torch.compiler.disable
def compute_varlen_max(varlen):
    return int(varlen.max().item())


class _sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, alpha=1.5, is_causal=False, varlen=None, niter=10):
        # shape constraints
        B, N_H, N_CTX, H_DIM = q.shape
        assert H_DIM in {16, 32, 64, 128, 256}

        ## -- constants and flags --
        device = q.device
        sm_scale = 1 / sqrt(H_DIM)
        IS_CAUSAL = is_causal
        IS_VARLEN = varlen is not None
        NEED_BACKWARD = q.requires_grad

        ASSERT_VARLEN(varlen, N_CTX)
        ASSERT_CONTIGUOUS(q, k, v, msg="Q, K and/or V are not contiguous.")

        MAX_CTX = N_CTX
        if IS_VARLEN:
            MAX_CTX = compute_varlen_max(varlen)

        ## -- tensors --
        taus = torch.zeros((B, N_H, MAX_CTX), device=device, dtype=torch.float32).contiguous()  # fmt: skip

        ## -- grid: get_tau --
        BLOCK_M = 64
        BLOCK_N = 64

        mblocks = triton.cdiv(MAX_CTX, BLOCK_M)
        grid_tau = (mblocks, N_H, B)

        _get_tau[grid_tau](
            q,
            k,
            taus,
            varlen,
            ##
            alpha,
            sm_scale,
            niter,
            IS_CAUSAL,
            IS_VARLEN,
            ##
            BLOCK_M,
            BLOCK_N,
            ##
            N_H,
            H_DIM,
            MAX_CTX,
            ##
            q.stride(1),
            taus.stride(1),
            ##
            num_warps=4,
            num_stages=3,
        )
        ###################################################

        ## -- grid: get_output --
        BLOCK_M = 64
        BLOCK_N = 32

        mblocks = triton.cdiv(MAX_CTX, BLOCK_M)
        grid_out = (mblocks, N_H, B)

        out = torch.zeros_like(q).contiguous()
        out2 = torch.zeros_like(q).contiguous() if NEED_BACKWARD else None

        _get_output[grid_out](
            q,
            k,
            v,
            out,
            out2,
            taus,
            varlen,
            ##
            alpha,
            sm_scale,
            NEED_BACKWARD,
            IS_CAUSAL,
            IS_VARLEN,
            ##
            BLOCK_M,
            BLOCK_N,
            ##
            N_H,
            H_DIM,
            MAX_CTX,
            ##
            q.stride(1),
            taus.stride(1),
            ##
            num_warps=2,
            num_stages=3,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            out2,
            taus,
            varlen,
        )
        ctx.sm_scale = sm_scale
        ctx.alpha = alpha
        ctx.IS_CAUSAL = IS_CAUSAL
        ctx.IS_VARLEN = IS_VARLEN
        ctx.MAX_CTX = MAX_CTX

        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, taus, varlen = ctx.saved_tensors

        ## -- constants and flags--
        B, N_H, _, H_DIM = q.shape
        alpha = ctx.alpha
        device = q.device
        sm_scale = ctx.sm_scale
        IS_CAUSAL = ctx.IS_CAUSAL
        IS_VARLEN = ctx.IS_VARLEN
        MAX_CTX = ctx.MAX_CTX

        # ASSERT_CONTIGUOUS(do, msg="Output gradient needs to be contiguous.")
        if not do.is_contiguous():
            do = do.contiguous()

        ## -- grid: preprocess --
        PRE_BLOCK = 128
        pre_mblocks = triton.cdiv(MAX_CTX, PRE_BLOCK)
        pre_grid = (pre_mblocks, N_H, B)

        delta = torch.zeros((B, N_H, MAX_CTX), device=device, dtype=torch.float32).contiguous()  # fmt: skip

        _bwd_preprocess[pre_grid](
            o,
            do,
            delta,
            varlen,
            ##
            o.stride(1),
            delta.stride(1),
            ##
            IS_VARLEN,
            ##
            N_H,
            H_DIM,
            MAX_CTX,
            ##
            PRE_BLOCK,
            ##
            num_warps=16,
            num_stages=1,
        )

        ## -- grid: dkdv --

        BLOCK_M = 32
        BLOCK_N = 32

        nblocks = triton.cdiv(MAX_CTX, BLOCK_N)
        grid_kv = (nblocks, N_H, B)

        ## -- initializing dk and dv --
        dk = torch.zeros_like(k).contiguous()
        dv = torch.zeros_like(v).contiguous()

        _bwd_kv_kernel[grid_kv](
            q,
            k,
            v,
            do,
            dk,
            dv,
            taus,
            varlen,
            delta,
            ##
            alpha,
            sm_scale,
            IS_CAUSAL,
            IS_VARLEN,
            ##
            q.stride(1),
            taus.stride(1),
            delta.stride(1),
            ##
            H_DIM,
            N_H,
            MAX_CTX,
            ##
            BLOCK_M,
            BLOCK_N,
            ##
            num_warps=2,
            num_stages=2,
        )

        ## -- grid: dq --

        BLOCK_M = 32
        BLOCK_N = 32

        mblocks = triton.cdiv(MAX_CTX, BLOCK_M)
        grid_q = (mblocks, N_H, B)

        ## -- initializing dq --
        dq = torch.zeros_like(q).contiguous()

        _bwd_q_kernel[grid_q](
            q,
            k,
            v,
            do,
            dq,
            taus,
            varlen,
            delta,
            ##
            alpha,
            sm_scale,
            IS_CAUSAL,
            IS_VARLEN,
            ##
            q.stride(1),
            taus.stride(1),
            delta.stride(1),
            ##
            H_DIM,
            N_H,
            MAX_CTX,
            ##
            BLOCK_M,
            BLOCK_N,
            ##
            num_warps=2,
            num_stages=2,
        )

        return dq, dk, dv, None, None, None, None


def sparse_attn(q, k, v, alpha=1.5, is_causal=False, varlen=None, niter=10):
    return _sparse_attention.apply(q, k, v, alpha, is_causal, varlen, niter)
