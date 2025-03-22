from math import sqrt

import torch

import triton
import triton.language as tl


@triton.jit
def _or_combine(a, b):
    return a | b


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
    BMASK,
    ##
    alpha,
    sm_scale,
    NITER,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    BLOCK_M: tl.constexpr,
    BLOCK_M_TINY: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    N_CTX,
    ##
    stride_qh,
    stride_th,
    stride_bmh,
    stride_bmm,
):
    ## -- constants --
    input_dtype = Q.dtype.element_ty
    kv_jump: tl.constexpr = BLOCK_N * H_DIM
    TINY_FACTOR: tl.constexpr = BLOCK_M // BLOCK_M_TINY

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

    ## -- alright, after some iterations we save the mask --
    BMASK += off_hz * stride_bmh + TINY_FACTOR * start_m
    bmask_ptrs = BMASK + tl.arange(0, TINY_FACTOR)

    ## -- we need to reset k_ptrs --
    k_ptrs = K + offsets_n[:, None] * H_DIM + offsets_k

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

        ## We now need to reduce the mask
        ## 0 -> means that entry < tau
        ## 1 -> means that entry > tau

        ## -- first we will reduce row_wise, so (BLOCK_M, BLOCK_N) -> (BLOCK_M) --
        qk_mask = tl.reduce(qk_mask, combine_fn=_or_combine, axis=1)

        ## Since we want to have tiny blocks (higher sparsity) and here we are using
        ## a big BLOCK_M, we need to split our mask into BLOCK_M / BLOCK_M_TINY

        ## -- first we need to reshape --
        qk_mask = tl.reshape(qk_mask, (TINY_FACTOR, BLOCK_M_TINY), can_reorder=False)

        ## -- now we do a final reduce --
        qk_mask = tl.reduce(qk_mask, combine_fn=_or_combine, axis=1)

        ## At the end we essentially get a vector of size BLOCK_M / BLOCK_M_TINY
        ## which will tells us whether the blocks do or don't have a non-zero entry.
        ## However, because we do not want to do a uncoalesced stores, we require the
        ## BMASK to be (nblocks, tiny_mblocks) instead of the more intuitive (tiny_mblocks, nblocks)

        ## -- now save the mask --
        tl.store(bmask_ptrs, 1, mask=qk_mask)

        ## Quick note, we are saving as above because the following seems to not work:
        ## tl.store(bmask_ptrs, qk_mask)
        ## I suspect that this does not work because how torch.bool are encoded in PyTorch
        ## PyTorch's bool are actually 1 byte (8bits), and maybe Triton's int1 is actually
        ## 1 bit, so the storing is not "aligned".

        ## -- increments pts --
        k_ptrs += kv_jump
        bmask_ptrs += stride_bmm


@torch.compile
def compute_bidxs_and_cubcounts(
    bmask: torch.Tensor,
    B: int,
    N_H: int,
    mblocks: int,
    nblocks: int,
    NEED_BACKWARD: bool = True,
    device: str = "cuda",
):
    kv_bidxs = None
    kv_cubcount = None

    if NEED_BACKWARD:
        kv_cubcount = torch.zeros((B * N_H * nblocks + 1,), device=device, dtype=torch.int32)  # fmt: skip
        kv_bidxs = bmask.nonzero(as_tuple=True)[3].to(torch.int16)
        torch.cumsum(bmask.sum(dim=-1).flatten(), dim=0, out=kv_cubcount[1:])

    q_cubcount = torch.zeros((B * N_H * mblocks + 1,), device=device, dtype=torch.int32)
    bmask_q = bmask.permute(0, 1, 3, 2)
    q_bidxs = bmask_q.nonzero(as_tuple=True)[3].to(torch.int16)
    torch.cumsum(bmask_q.sum(dim=-1).flatten(), dim=0, out=q_cubcount[1:])

    return kv_bidxs, kv_cubcount, q_bidxs, q_cubcount


@triton.jit
def _get_output(
    Q,
    K,
    V,
    OUT,
    OUT2,
    TAUS,
    VARLEN,
    KV_IDXS,
    KV_CUBCOUNT,
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
    mblocks,
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

    ## -- before we update the pointers, let's get the number of "good" nblocks here --
    KV_CUBCOUNT += off_hz * mblocks + start_m
    nblocks_start = tl.load(KV_CUBCOUNT).to(tl.int64)
    nblocks_end = tl.load(KV_CUBCOUNT + 1).to(tl.int64)
    good_nblocks = nblocks_end - nblocks_start  # fmt: skip

    ## -- update offsets --
    Q += qvk_offset + start_m * BLOCK_M * H_DIM
    K += qvk_offset
    V += qvk_offset
    OUT += qvk_offset + start_m * BLOCK_M * H_DIM

    if NEED_BACKWARD:
        OUT2 += qvk_offset + start_m * BLOCK_M * H_DIM

    TAUS += off_hz * stride_th + start_m * BLOCK_M
    KV_IDXS += nblocks_start

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

    if IS_CAUSAL:
        ## -- get the idxs of q --
        q_idxs = offsets_m + start_m * BLOCK_M

    ## -- compute output --
    acc = tl.zeros([BLOCK_M, H_DIM], dtype=tl.float32)
    if NEED_BACKWARD:
        acc2 = tl.zeros([BLOCK_M, H_DIM], dtype=tl.float32)
        supp_size = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for bidx in range(0, good_nblocks):

        ## -- load the idx of the next good kv block for this block of q's --
        c_block = tl.load(KV_IDXS + bidx).to(tl.int32)

        if IS_VARLEN or IS_CAUSAL:
            ## -- get the idxs of k --
            k_idxs = c_block * BLOCK_N + offsets_n

        if IS_CAUSAL:
            ## -- build causal_mask --
            causal_mask = q_idxs[:, None] >= k_idxs[None, :]

        ## -- load k --
        if IS_VARLEN:
            kv_mask = k_idxs < seqlen
            k = tl.load(k_ptrs + c_block * kv_jump, mask=kv_mask[:, None], other=0).to(input_dtype)  # fmt: skip
            v = tl.load(v_ptrs + c_block * kv_jump, mask=kv_mask[:, None], other=0).to(input_dtype)  # fmt: skip
        else:
            k = tl.load(k_ptrs + c_block * kv_jump).to(input_dtype)  # fmt: skip
            v = tl.load(v_ptrs + c_block * kv_jump).to(input_dtype)  # fmt: skip

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
    Q_IDXS,
    Q_CUBCOUNT,
    ##
    alpha,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    stride_qh,
    stride_th,
    stride_dh,
    nblocks,
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

    ## -- before we update the pointers, let's get the number of "good" nblocks here --
    Q_CUBCOUNT += off_hz * nblocks + start_n
    mblocks_start = tl.load(Q_CUBCOUNT).to(tl.int64)
    mblocks_end = tl.load(Q_CUBCOUNT + 1).to(tl.int64)
    good_mblocks = mblocks_end - mblocks_start  # fmt: skip

    if good_mblocks == 0:
        return

    ## -- update offsets --
    Q += qkv_offset
    K += qkv_offset + start_n * BLOCK_N * H_DIM
    V += qkv_offset + start_n * BLOCK_N * H_DIM
    DO += qkv_offset
    DK += qkv_offset + start_n * BLOCK_N * H_DIM
    DV += qkv_offset + start_n * BLOCK_N * H_DIM

    D += off_hz * stride_dh
    TAUS += off_hz * stride_th
    Q_IDXS += mblocks_start

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
        ## -- create kv_idxs --
        kv_idxs = offsets_n + start_n * BLOCK_N

    for bidx in range(0, good_mblocks):

        ## -- load the idx of the next good q block for this block of k's --
        c_block = tl.load(Q_IDXS + bidx).to(tl.int32)

        if IS_VARLEN or IS_CAUSAL:
            ## -- get the idxs of q --
            q_idxs = c_block * BLOCK_M + offsets_m

        ## -- load q --
        if IS_VARLEN:
            q_mask = q_idxs < seqlen
            q = tl.load(q_ptrs + c_block * q_jump, mask=q_mask[:, None], other=0).to(input_dtype)  # fmt: skip
        else:
            q = tl.load(q_ptrs + c_block * q_jump).to(input_dtype)

        ## -- load tau beforehand --
        if IS_VARLEN:
            t = tl.load(t_ptrs + c_block * BLOCK_M, mask=q_mask, other=0)
        else:
            t = tl.load(t_ptrs + c_block * BLOCK_M)

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
            do = tl.load(do_ptrs + c_block * q_jump, mask=q_mask[:, None], other=0).to(input_dtype)  # fmt: skip
        else:
            do = tl.load(do_ptrs + c_block * q_jump).to(input_dtype)

        ## -- compute dv --
        dv += tl.dot(tl.trans(qk_act), do, input_precision="ieee")

        ## -- load delta --
        if IS_VARLEN:
            delta = tl.load(d_ptrs + c_block * BLOCK_M, mask=q_mask, other=0)
        else:
            delta = tl.load(d_ptrs + c_block * BLOCK_M)

        ## -- compute dp --
        dp = tl.dot(do, v, input_precision="ieee")

        ## -- calculate u --
        u = tl.where(qk_mask, tl.exp2(qk_log * coeff_f), 0)

        ## -- compute ds --
        ds = u * (dp - delta[:, None])
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
    KV_IDXS,
    KV_CUBCOUNT,
    ##
    alpha,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ##
    stride_qh,
    stride_th,
    stride_dh,
    mblocks,
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

    ## -- before we update the pointers, let's get the number of "good" nblocks here --
    KV_CUBCOUNT += off_hz * mblocks + start_m
    nblocks_start = tl.load(KV_CUBCOUNT).to(tl.int64)
    nblocks_end = tl.load(KV_CUBCOUNT + 1).to(tl.int64)
    good_nblocks = nblocks_end - nblocks_start  # fmt: skip

    ## -- update offsets --
    Q += qkv_offset + start_m * BLOCK_M * H_DIM
    K += qkv_offset
    V += qkv_offset
    DO += qkv_offset + start_m * BLOCK_M * H_DIM
    DQ += qkv_offset + start_m * BLOCK_M * H_DIM

    D += off_hz * stride_dh + start_m * BLOCK_M
    TAUS += off_hz * stride_th + start_m * BLOCK_M
    KV_IDXS += nblocks_start

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

    if IS_CAUSAL:
        ## -- get the idxs of q --
        q_idxs = offsets_m + start_m * BLOCK_M

    ## -- to accumulate dq --
    dq = tl.zeros([BLOCK_M, H_DIM], dtype=tl.float32)

    for bidx in range(0, good_nblocks):

        ## -- load the idx of the next good kv block for this block of q's --
        c_block = tl.load(KV_IDXS + bidx).to(tl.int32)

        if IS_VARLEN or IS_CAUSAL:
            ## -- get the idxs of k --
            k_idxs = c_block * BLOCK_N + offsets_n

        if IS_CAUSAL:
            ## -- build causal_mask --
            causal_mask = q_idxs[:, None] >= k_idxs[None, :]

        ## -- load k --
        if IS_VARLEN:
            kv_mask = k_idxs < seqlen
            k = tl.load(k_ptrs + c_block * kv_jump, mask=kv_mask[:, None], other=0).to(input_dtype)  # fmt: skip
        else:
            k = tl.load(k_ptrs + c_block * kv_jump).to(input_dtype)

        ## -- compute scores --
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")

        qk_mask = qk > t[:, None]
        if IS_CAUSAL:
            qk_mask &= causal_mask
        if IS_VARLEN and not IS_CAUSAL:
            qk_mask &= kv_mask

        ## -- load v now --
        if IS_VARLEN:
            v = tl.load(v_ptrs + c_block * kv_jump, mask=kv_mask[:, None], other=0).to(input_dtype)  # fmt: skip
        else:
            v = tl.load(v_ptrs + c_block * kv_jump).to(input_dtype)

        ## -- calculate u, it's all we need --
        qk_log = tl.log2(qk - t[:, None])
        u = tl.where(qk_mask, tl.exp2(qk_log * coeff_f), 0).to(input_dtype)

        ## -- compute dp and ds --
        dp = tl.dot(do.to(input_dtype), tl.trans(v), input_precision="ieee")

        ds = u * (dp - delta[:, None])

        ## -- compute dq --
        dq += tl.dot(ds.to(input_dtype), k, input_precision="ieee")

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
        BLOCK_N = 16
        BLOCK_M_TINY = 16

        mblocks = triton.cdiv(MAX_CTX, BLOCK_M)
        nblocks = triton.cdiv(MAX_CTX, BLOCK_N)
        mblocks_tiny = triton.cdiv(MAX_CTX, BLOCK_M_TINY)
        grid_tau = (mblocks, N_H, B)

        ## -- allocate bmask --
        ## TODO: alternatively pass this as a argument,
        #        however it needs to be fixed size to the MAX_LEN possible (power of two)
        bmask = torch.zeros((B, N_H, nblocks, mblocks_tiny), device=device, dtype=torch.bool,).contiguous()  # fmt: skip

        _get_tau[grid_tau](
            q,
            k,
            taus,
            varlen,
            bmask,
            ##
            alpha,
            sm_scale,
            niter,
            IS_CAUSAL,
            IS_VARLEN,
            ##
            BLOCK_M,
            BLOCK_M_TINY,
            BLOCK_N,
            ##
            N_H,
            H_DIM,
            MAX_CTX,
            ##
            q.stride(1),
            taus.stride(1),
            bmask.stride(1),
            bmask.stride(2),
            ##
            num_warps=4,
            num_stages=3,
        )
        ###################################################

        ## -- grid: get_output --
        BLOCK_M = 16
        BLOCK_N = 16

        mblocks = triton.cdiv(MAX_CTX, BLOCK_M)
        nblocks = triton.cdiv(MAX_CTX, BLOCK_N)
        grid_out = (mblocks, N_H, B)

        q_bidxs, q_cubcount, kv_bidxs, kv_cubcount = compute_bidxs_and_cubcounts(
            bmask, B, N_H, mblocks, nblocks, NEED_BACKWARD=NEED_BACKWARD
        )  # fmt: skip

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
            kv_bidxs,
            kv_cubcount,
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
            mblocks,
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
            kv_bidxs,
            kv_cubcount,
            q_bidxs,
            q_cubcount,
        )
        ctx.sm_scale = sm_scale
        ctx.alpha = alpha
        ctx.IS_CAUSAL = IS_CAUSAL
        ctx.IS_VARLEN = IS_VARLEN
        ctx.MAX_CTX = MAX_CTX
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N

        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, taus, varlen, kv_bidxs, kv_cubcount, q_bidxs, q_cubcount = ctx.saved_tensors

        ## -- constants and flags--
        B, N_H, _, H_DIM = q.shape
        alpha = ctx.alpha
        device = q.device
        sm_scale = ctx.sm_scale
        IS_CAUSAL = ctx.IS_CAUSAL
        IS_VARLEN = ctx.IS_VARLEN
        MAX_CTX = ctx.MAX_CTX
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N

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
            q_bidxs,
            q_cubcount,
            ##
            alpha,
            sm_scale,
            IS_CAUSAL,
            IS_VARLEN,
            ##
            q.stride(1),
            taus.stride(1),
            delta.stride(1),
            nblocks,
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
            kv_bidxs,
            kv_cubcount,
            ##
            alpha,
            sm_scale,
            IS_CAUSAL,
            IS_VARLEN,
            ##
            q.stride(1),
            taus.stride(1),
            delta.stride(1),
            mblocks,
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


# TODO: Support the post_niter parameter.
def sparse_attn(q, k, v, alpha=1.5, is_causal=False, varlen=None, niter=10):
    return _sparse_attention.apply(q, k, v, alpha, is_causal, varlen, niter)
