"""Split-KV paged attention for speculative-decode batches (a few query tokens per
request, GQA), Triton.

Neither vLLM's FlashAttention-2 path nor its Triton unified attention split the KV
sequence across SMs when a request has more than one query token: with MTP k=4
(5 queries) on a 24-head model that leaves 24 (FA) or ~8 (Triton) thread blocks on an
82-SM RTX 3090 and the attention layer takes ~57 us for a 1.5k-token context.
This kernel gives every (request, kv-head, query tile) NUM_SEGMENTS blocks, each
computing an online-softmax partial over its slice of the KV cache for all its query
rows at once (G = query heads per kv head), followed by a tiny combine kernel.

Long blocks: the query rows of one request (q_len x G of them) are split into tiles of
BLOCK_M rows, so the kernel is not capped at BLOCK_M // G query tokens. That cap is what
made lookup-augmented drafting unaffordable past a block of 7 -- a 16-token verify fell
back to FA2 and doubled the step at 25k context. Each tile re-reads the KV segment, so
BLOCK_M is chosen to keep the tile count at 1 where the register budget allows.

Layout matches vLLM's FLASH_ATTN backend: q [T, Hq, D], key/value cache
[num_blocks, block_size, Hkv, D] (block_size any multiple of 16), block_table
[num_reqs, max_blocks], seqused_k [num_reqs] = kv length including the new tokens,
cu_seqlens_q [num_reqs + 1]. Query token i of a request sits at kv position
seqused_k - q_len + i and attends causally.

Restrictions: q_len <= QMAX_TOKENS per request, D a power of two <= 256, no sliding
window / softcap / alibi.
"""
import os

import torch
import triton
import triton.language as tl

NUM_SEGMENTS = 32
BLOCK_M = 64       # query rows (q_len * G) per program at the default register budget
BLOCK_M_BIG = 128  # ... and with 8 warps, which keeps a 16-token block in one tile
QMAX_TOKENS = 64   # query tokens per request the caller may ask for


@triton.jit
def _e4m3_to_fp32(b):
    """Decode E4M3 bytes (as uint8) to fp32: sign|exp(4, bias 7)|mantissa(3)."""
    b = b.to(tl.int32)
    sign = tl.where((b & 0x80) != 0, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    mant_f = mant.to(tl.float32)
    normal = sign * tl.exp2(exp.to(tl.float32) - 7.0) * (1.0 + mant_f / 8.0)
    subnormal = sign * (mant_f / 8.0) * tl.exp2(-6.0)
    return tl.where(exp == 0, subnormal, normal)


@triton.jit
def _spec_attn_partial(
    q_ptr, k_ptr, v_ptr, bt_ptr, seqused_ptr, cu_q_ptr, total_tokens, nblocks,
    part_o_ptr, part_m_ptr, part_l_ptr,
    scale,
    k_scale, v_scale,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    G: tl.constexpr, Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    KV_FP8: tl.constexpr, D_KV: tl.constexpr,
    MAX_REQS: tl.constexpr,
    BLOCK_M: tl.constexpr, TILE: tl.constexpr, NSEG: tl.constexpr, QT: tl.constexpr,
    NTILE: tl.constexpr, NO_KV: tl.constexpr,
):
    pid = tl.program_id(0)
    req = pid // NTILE
    qtile = pid % NTILE
    kvh = tl.program_id(1)
    seg = tl.program_id(2)

    # Raw launch-time values, written before any load: the last slot left in the buffer
    # says how far the kernel got, and the values themselves (request index, tile, head,
    # segment) reveal a bad grid or a bad request count at replay time. Slots 27..31.

    # The partial buffers hold MAX_REQS requests (their capacity at allocation).
    # A request slot beyond that must not index them: under CUDA-graph replay the
    # caller cannot re-validate the batch, so the kernel bounds itself. The grid is
    # built from the caller's request count, which can exceed the buffers when it is
    # derived from a padded persistent buffer instead of the exact batch size.
    if req >= MAX_REQS:
        return

    # Reached the kernel body (before the three metadata loads that follow): if this
    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    kv_len = tl.load(seqused_ptr + req)

    # cu_seqlens_q is a padded persistent buffer: only [:num_reqs + 1] is written each
    # step, so the tail keeps the previous (larger) batch's prefix sums. seqused_k pads
    # are zero and stay zero, so a request with no KV cannot be a real one, and a slot
    # whose query rows fall outside this step's token count cannot be one either. A slot
    # failing either test is inert (the combine kernel applies the same test, so it never
    # reads a partial that was not written).
    if kv_len <= 0 or q_start < 0 or q_start + q_len > total_tokens:
        return
    # The block table read uses the LIVE kv_len against the table width captured in
    # stride_bt; if the two disagree the column index walks past the row.
    if kv_len > stride_bt * BLOCK_SIZE or kv_len > nblocks * BLOCK_SIZE:
        return
    if q_len > QMAX:
        # The partial buffers are sized by QMAX (fixed at capture time); a query
        # block longer than that would index past them. Record and skip.
        return

    # rows: r = i * G + g  -> query token qtile * QT + i (0..q_len-1), head kvh*G + g
    r = tl.arange(0, BLOCK_M)
    ri = qtile * QT + r // G
    rg = r % G
    row_ok = (r < QT * G) & (ri < q_len)
    q_pos = kv_len - q_len + ri                      # kv position of each query row
    d = tl.arange(0, D)
    dkv = tl.arange(0, D_KV)   # KV head extent: bytes when the cache is fp8
    # Bound every address by construction: a captured CUDA graph replays these kernels
    # with arguments fixed at capture time, so a single stale value must not be able to
    # turn into a wild address (2026-09-12: an Xid 31 read ~4.2 GB below the KV cache).
    q_start_c = tl.minimum(tl.maximum(q_start, 0), tl.maximum(total_tokens - 1, 0))
    q_row = tl.minimum(q_start_c + ri, tl.maximum(total_tokens - 1, 0))
    q_ptrs = q_ptr + q_row[:, None] * stride_qt + (kvh * G + rg)[:, None] * stride_qh + d[None, :]
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)

    # separates the KV cache reads from the rest of the kernel. Partials written this way
    # are meaningless (the results are wrong); this is not a production switch.
    if NO_KV:
        tiles_total = 0
    else:
        tiles_total = (kv_len + TILE - 1) // TILE
    tiles_per_seg = (tiles_total + NSEG - 1) // NSEG
    t0 = seg * tiles_per_seg
    t1 = tl.minimum(t0 + tiles_per_seg, tiles_total)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    qs = (q * scale).to(tl.bfloat16)

    for t in range(t0, t1):
        pos = t * TILE + tl.arange(0, TILE)
        k_ok = pos < kv_len
        blk = tl.load(bt_ptr + req * stride_bt + pos // BLOCK_SIZE, mask=k_ok, other=0)
        # A block id outside the cache is never valid; treat those positions as absent
        # rather than letting the address run away. Under CUDA-graph replay the caller
        # cannot validate the table (no Python runs), so this is the only backstop:
        # 2026-09-12 a garbage id put the read ~4.2 GB below the cache base and faulted
        # the GPU (Xid 31) on prefix-cache-hit steps.
        bad = (blk < 0) | (blk >= nblocks)
        if tl.max(bad.to(tl.int32)) != 0:
            first = tl.min(tl.where(bad, pos, 1 << 30))
        k_pos = k_ok & ~bad   # 1-D position validity (drives the score mask too)
        slot = pos % BLOCK_SIZE
        # Every address component of the gather, written unconditionally: the fault is
        # inside this gather (skipping it is stable, running it crashes) and the block-id
        # guard never fires, so one of the strides or extents must be the wild one.
        # Bound the gather offset explicitly, against the cache extent derived from the
        # same parameters the strides came from. Every earlier guard constrains an index
        # (block id, slot, kv length) but not the composed address, so a parameter set
        # that is internally consistent yet stale relative to the live tensors still walks
        # out of the cache; masking on the composed offset closes that hole.
        k_off = (blk[:, None] * stride_kb + slot[:, None] * stride_ks
                 + kvh * stride_kh + dkv[None, :])
        v_off = (blk[:, None] * stride_vb + slot[:, None] * stride_vs
                 + kvh * stride_vh + dkv[None, :])
        k_lim = nblocks * stride_kb
        v_lim = nblocks * stride_vb
        g_ok = k_pos[:, None] & (k_off >= 0) & (k_off < k_lim)
        gv_ok = k_pos[:, None] & (v_off >= 0) & (v_off < v_lim)
        k_ptrs = k_ptr + k_off
        v_ptrs = v_ptr + v_off
        if KV_FP8:
            # sm80's Triton cannot load float8e4nv, so read the raw bytes and decode
            # e4m3 in software: sign(1) | exp(4) | mantissa(3), bias 7.
            kb = tl.load(k_ptrs, mask=g_ok, other=0).to(tl.uint8)
            vb = tl.load(v_ptrs, mask=gv_ok, other=0).to(tl.uint8)
            k = (_e4m3_to_fp32(kb) * k_scale).to(tl.bfloat16)
            v = (_e4m3_to_fp32(vb) * v_scale).to(tl.bfloat16)
        else:
            # Unquantized cache: load verbatim (no conversion) so the bf16 fast path is
            # byte-for-byte what it was before the fp8 support was added.
            k = tl.load(k_ptrs, mask=g_ok, other=0.0)
            v = tl.load(v_ptrs, mask=gv_ok, other=0.0)
        s = tl.dot(qs, tl.trans(k)).to(tl.float32)            # [BLOCK_M, TILE]
        allowed = k_pos[None, :] & (pos[None, :] <= q_pos[:, None]) & row_ok[:, None]
        s = tl.where(allowed, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i - m_safe))
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m_i = m_new

    # store partials at flat index ((req*Hq + head)*QMAX + i)*NSEG + seg
    hrow = kvh * G + rg
    pidx = ((req * Hq + hrow) * QMAX + ri) * NSEG + seg
    tl.store(part_o_ptr + pidx[:, None] * D + d[None, :], acc, mask=row_ok[:, None])
    tl.store(part_m_ptr + pidx, m_i, mask=row_ok)
    tl.store(part_l_ptr + pidx, l_i, mask=row_ok)


@triton.jit
def _spec_attn_combine(
    part_o_ptr, part_m_ptr, part_l_ptr, out_ptr, cu_q_ptr, seqused_ptr, total_tokens,
    stride_ot, stride_oh,
    Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, NSEG: tl.constexpr,
    NOOP: tl.constexpr, MAX_REQS: tl.constexpr,
):
    if NOOP:
        return
    req = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)
    if req >= MAX_REQS:
        return
    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    # Same padded-buffer guard as the partial kernel, so the two agree on which slots
    # have partials: skip slots with no KV, and slots whose query rows fall outside this
    # step's token count (a stale cu_seqlens_q tail would otherwise write past them).
    if (
        i < q_len
        and q_len <= QMAX
        and tl.load(seqused_ptr + req) > 0
        and q_start >= 0
        and q_start + q_len <= total_tokens
    ):
        base = ((req * Hq + h) * QMAX + i) * NSEG
        segs = tl.arange(0, NSEG)
        m = tl.load(part_m_ptr + base + segs)
        l = tl.load(part_l_ptr + base + segs)
        m_max = tl.max(m, 0)
        m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
        w = tl.exp(m - m_max)                       # segments with -inf give 0
        l_tot = tl.sum(l * w, 0)
        d = tl.arange(0, D)
        o = tl.load(part_o_ptr + (base + segs)[:, None] * D + d[None, :])   # [NSEG, D]
        o = tl.sum(o * w[:, None], 0) / tl.maximum(l_tot, 1e-30)
        row = tl.minimum(tl.maximum(q_start + i, 0), tl.maximum(total_tokens - 1, 0))
        tl.store(out_ptr + row * stride_ot + h * stride_oh + d, o.to(out_ptr.dtype.element_ty))


# Must match mamba_utils' one-shot layout (hook_guard region).



class SpecDecodeAttention:
    """Holds the partial buffers; call .run(...) per layer."""

    def __init__(self, max_num_reqs, num_heads, head_dim, device, qmax,
                 num_segments=NUM_SEGMENTS):
        # The partial buffers are allocated once, for the longest query block the caller
        # will ever pass (qmax): a CUDA graph captures their addresses, so growing them
        # later would leave the captured decode graph pointing at freed memory.
        self.nseg = num_segments
        self.qmax = qmax
        self.max_num_reqs = max_num_reqs
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        n = max_num_reqs * num_heads * qmax * self.nseg
        self.part_o = torch.empty(n, head_dim, dtype=torch.float32, device=device)
        self.part_m = torch.empty(n, dtype=torch.float32, device=device)
        self.part_l = torch.empty(n, dtype=torch.float32, device=device)

    def _plan(self, q_len, G, D):
        """(BLOCK_M, tokens per tile, tile count, warps). One tile is preferred: every
        extra tile re-reads this request's KV segment."""
        override = os.environ.get("VLLM_SPEC_ATTN_BLOCK_M")
        rows = q_len * G
        if override:
            block_m = int(override)
        elif rows <= 32:
            block_m = 32
        elif rows <= 64:
            block_m = 64
        else:
            block_m = BLOCK_M_BIG
        block_m = max(block_m, G if G > 1 else 1)
        block_m = 1 << (block_m - 1).bit_length()
        qt = max(1, block_m // G)
        return block_m, qt, triton.cdiv(q_len, qt), 8 if block_m >= 128 else 4

    def run(self, q, key_cache, value_cache, out, cu_seqlens_q, seqused_k, block_table,
            scale, num_reqs, max_query_len, k_scale=1.0, v_scale=1.0):


        Hq, D = q.shape[1], q.shape[2]
        Hkv = key_cache.shape[2]
        G = Hq // Hkv
        assert max_query_len <= self.qmax, "too many query tokens per request for this kernel"
        assert num_reqs <= self.max_num_reqs
        # shared memory on sm86 is 99 KB: q tile + one K and one V tile + scores must fit
        block_m, qt, ntile, warps = self._plan(max_query_len, G, D)
        # Live arguments at (possibly) replay time: scalars here are frozen at capture,

        # The live-tensor dumps used to run here. They are the only kernels between the
        # the fault is in them or in the partial kernel itself.
        # TILE=64 up to D=256 (measured on the 170HX at the production shape
        # Hq=24/Hkv=4/D=256/q=8, kv=117,535, block_size=832: 0.876 ms/layer against
        # 1.450 at TILE=32, 1.66x; tied at batch 8; 0.076 vs 0.086 ms at kv=2.3k, i.e.
        # noise). Shared memory at BLOCK_M=64/D=256/TILE=64: q 32 KB + K 32 KB + V 32 KB
        # + scores 16 KB = 112 KB, under sm80's 164 KB.
        tile = 64 if (block_m <= 32 or D <= 256) else 32
        grid = (num_reqs * ntile, Hkv, self.nseg)
        import os as _os

        _spec_attn_partial[grid](
            q, key_cache, value_cache, block_table, seqused_k, cu_seqlens_q,
            q.shape[0], key_cache.shape[0],
            self.part_o, self.part_m, self.part_l,
            scale,
            k_scale, v_scale,
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0),
            G=G, Hq=Hq, QMAX=self.qmax, D=D, BLOCK_SIZE=key_cache.shape[1], BLOCK_M=block_m,
            KV_FP8=key_cache.dtype == torch.uint8,
            # Per-head extent in the *elements of the pointer's dtype*: the kernel
            # indexes key_cache with its own strides, so the extent is D for a bf16
            # cache and D for an fp8 cache viewed as uint8 (one byte per element).
            # The old `D * 2 if uint8` assumed a bf16 cache reinterpreted as bytes;
            # against a real fp8 cache it doubled the head extent and walked out of
            # the block on every tile.
            D_KV=D,
            MAX_REQS=self.max_num_reqs,
        NO_KV=_os.environ.get("VLLM_SPEC_ATTN_NO_KV", "0") == "1",
            TILE=tile, NSEG=self.nseg, QT=qt, NTILE=ntile,
            num_warps=warps, num_stages=1,
        )
        _spec_attn_combine[(num_reqs, Hq, max_query_len)](
            self.part_o, self.part_m, self.part_l, out, cu_seqlens_q, seqused_k,
            q.shape[0],
            out.stride(0), out.stride(1),
            Hq=Hq, QMAX=self.qmax, D=D, NSEG=self.nseg,
            NOOP=os.environ.get("VLLM_SPEC_ATTN_NOOP_COMBINE", "0") == "1",
            MAX_REQS=self.max_num_reqs,
            num_warps=4,
        )
        return out
