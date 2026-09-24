// Chunked-prefill attention for MiMo-V2.6 global (full-attention) layers,
// re-tiled from the diffbot SM120 TP2 kernel for SM86 TP8.
//   Q [nq, HQ=8, DK=192] bf16 (varlen cu_seqlens_q), paged packed KV cache
//   [pages, HKV=1, page_size, DK+DV] fp8-e4m3 or bf16, block_table, causal
//   with prefix. Out [nq, HQ, DV=128] bf16.
// CTA = 16 query tokens x 8 heads (GQA) = 128 rows, 8 warps; warp w owns
// tokens row0+2w (row group g) and row0+2w+1 (row group g+8), so each warp's
// causal boundary differs per row group. K/V stream through a 2-stage smem
// ring as fp16 (converted on load — e4m3 via exact bit-repack, sm86 has no
// cvt.rn.f16x2.e4m3x2), 64 keys per tile, ldmatrix + mma.m16n8k16 f16->f32,
// FA2 online softmax (base 2). Per-tensor descales fold into the softmax
// scale and the output epilogue. Global layers carry no sink bias.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int HQ = 8, HKV = 1, GQA = HQ / HKV, DK = 192, DV = 128, KVROW = DK + DV;
constexpr int TOK = 16, ROWS = TOK * GQA;     // 128 rows per CTA, 8 warps x 16 rows
constexpr int TK = 64;                         // keys per tile
constexpr int KS = DK + 8, VS = DV + 8;        // smem row strides (halves): 400 B / 272 B
constexpr int STAGES = 2;
constexpr int SMEM_K = TK * KS * 2, SMEM_V = TK * VS * 2, SMEM_STAGE = SMEM_K + SMEM_V;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_f16x2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// Exact e4m3 -> fp16 for one value (sm86: no cvt.rn.f16x2.e4m3x2). e4m3
// values (incl. subnormals, m * 2^-9) are exactly representable in fp16.
// e4m3: s(1) e(4, bias 7) m(3); f16: s(1) e(5, bias 15) m(10).
__device__ __forceinline__ uint32_t e4m3_to_f16(uint16_t x) {
  const uint32_t s = (uint32_t)(x & 0x80) << 8;   // sign -> f16 bit 15
  const uint32_t e = (x >> 3) & 0xF, m = x & 7;
  if (e == 0) {                                    // zero / subnormal m * 2^-9
    if (m == 0) return s;
    const int k = 31 - __clz(m);                   // MSB position 0..2
    const uint32_t fe = (uint32_t)(15 - 9 + k);    // biased exponent 6..8
    const uint32_t mm = (uint32_t)((m ^ (1u << k)) << (10 - k));  // 10-bit mantissa
    return s | (fe << 10) | mm;
  }
  if (e == 15 && m == 7) return s | 0x7e00;        // NaN (0x7F / 0xFF)
  return s | ((e + 8) << 10) | (m << 7);           // bias 7 -> 15, mantissa 3 -> 10
}
// 8 fp8 (one uint2: v.x = bytes 0-3, v.y = bytes 4-7, little-endian)
// -> 8 fp16 (uint4 of four half2 words, element order preserved).
__device__ __forceinline__ uint4 fp8x8_to_f16x8(uint2 v) {
  uint4 o;
  uint16_t r[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    r[j] = (uint16_t)e4m3_to_f16((v.x >> (8 * j)) & 0xFF);
    r[4 + j] = (uint16_t)e4m3_to_f16((v.y >> (8 * j)) & 0xFF);
  }
  o.x = (uint32_t)r[0] | ((uint32_t)r[1] << 16);
  o.y = (uint32_t)r[2] | ((uint32_t)r[3] << 16);
  o.z = (uint32_t)r[4] | ((uint32_t)r[5] << 16);
  o.w = (uint32_t)r[6] | ((uint32_t)r[7] << 16);
  return o;
}
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v) {
  __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&v);
  float2 f = __bfloat1622float2(b);
  return pack_f16x2(f.x, f.y);
}

// Load one 64-key tile (K 192 + V 128 per key) into registers: per thread
// 5 x 16 fp8 (or 5 x 8 bf16), then convert to fp16 halves in smem.
template <bool FP8>
struct TileLoader {
  static constexpr int CH = 16;
  static constexpr int CPR = KVROW / CH;              // 20 chunks per key row
  static constexpr int NCH = TK * CPR;                // 1280
  static constexpr int PER_T = NCH / 256;             // 5
  uint4 reg[PER_T][FP8 ? 1 : 2];

  __device__ __forceinline__ void load(const void* __restrict__ cache,
                                       const int32_t* __restrict__ bt,
                                       int page_size, int key0, int seq_len,
                                       int tid) {
#pragma unroll
    for (int i = 0; i < PER_T; ++i) {
      const int c = tid + 256 * i, row = c / CPR, ch = c - row * CPR;
      const int key = key0 + row;
      if (key < seq_len) {
        const int page = bt[key / page_size];
        const int off = key - (key / page_size) * page_size;
        const size_t base = ((size_t)page * HKV + 0) * page_size + off;
        if (FP8) {
          reg[i][0] = __ldg(reinterpret_cast<const uint4*>(
              static_cast<const uint8_t*>(cache) + base * KVROW + ch * CH));
        } else {
          const uint4* p = reinterpret_cast<const uint4*>(
              static_cast<const __nv_bfloat16*>(cache) + base * KVROW + ch * CH);
          reg[i][0] = __ldg(p);
          reg[i][1] = __ldg(p + 1);
        }
      } else {
        reg[i][0] = make_uint4(0, 0, 0, 0);
        if (!FP8) reg[i][1] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  __device__ __forceinline__ void store(uint8_t* stage, int tid) {
    __half* ks = reinterpret_cast<__half*>(stage);
    __half* vs = reinterpret_cast<__half*>(stage + SMEM_K);
#pragma unroll
    for (int i = 0; i < PER_T; ++i) {
      const int c = tid + 256 * i, row = c / CPR, ch = c - row * CPR;
      uint4 lo, hi;
      if (FP8) {
        lo = fp8x8_to_f16x8(make_uint2(reg[i][0].x, reg[i][0].y));
        hi = fp8x8_to_f16x8(make_uint2(reg[i][0].z, reg[i][0].w));
      } else {
        lo = make_uint4(bf16x2_to_f16x2(reg[i][0].x), bf16x2_to_f16x2(reg[i][0].y),
                        bf16x2_to_f16x2(reg[i][0].z), bf16x2_to_f16x2(reg[i][0].w));
        hi = make_uint4(bf16x2_to_f16x2(reg[i][1].x), bf16x2_to_f16x2(reg[i][1].y),
                        bf16x2_to_f16x2(reg[i][1].z), bf16x2_to_f16x2(reg[i][1].w));
      }
      const int e0 = ch * CH;
      if (e0 < DK) {
        uint4* d = reinterpret_cast<uint4*>(ks + row * KS + e0);
        d[0] = lo; d[1] = hi;
      } else {
        uint4* d = reinterpret_cast<uint4*>(vs + row * VS + (e0 - DK));
        d[0] = lo; d[1] = hi;
      }
    }
  }
};

template <bool FP8>
__global__ void __launch_bounds__(256, 1) prefill_attn_sm86_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ cache,
    const int32_t* __restrict__ block_table, int bt_stride, int page_size,
    const int32_t* __restrict__ cu_seqlens_q, const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ qblk_seq, const int32_t* __restrict__ qblk_start,
    float scale_log2, float v_descale, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c = lane & 3;
  const int qb = blockIdx.x;
  const int seq = qblk_seq[qb];
  const int q_start = cu_seqlens_q[seq], q_end = cu_seqlens_q[seq + 1];
  const int row0 = qblk_start[qb];
  const int seq_len = seq_lens[seq], ctx = seq_len - (q_end - q_start);
  // warp w owns tokens t0 = row0 + 2w (row group g) and t0+1 (row group g+8)
  const int t0 = row0 + 2 * warp;
  const bool valid0 = t0 < q_end, valid1 = t0 + 1 < q_end;
  const int pos0 = ctx + (t0 - q_start);            // keys <= pos0 allowed
  const int pos1 = ctx + (t0 + 1 - q_start);
  const int32_t* bt = block_table + (size_t)seq * bt_stride;
  const int last_tok = min(row0 + TOK - 1, q_end - 1);
  const int n_keys = ctx + (last_tok - q_start) + 1;
  const int n_tiles = (n_keys + TK - 1) / TK;

  // ---- Q fragments: A[row][k]; row group g = (t0, head g), g+8 = (t0+1, head g)
  uint32_t qa[DK / 16][4];
  {
    const __nv_bfloat16* qp0 = valid0 ? q + ((size_t)t0 * HQ + g) * DK : q;
    const __nv_bfloat16* qp1 = valid1 ? q + ((size_t)(t0 + 1) * HQ + g) * DK : q;
#pragma unroll
    for (int ks = 0; ks < DK / 16; ++ks) {
      const __nv_bfloat16* r0 = qp0 + ks * 16 + 2 * c;
      const __nv_bfloat16* r1 = qp1 + ks * 16 + 2 * c;
      uint32_t v0 = *reinterpret_cast<const uint32_t*>(r0);
      uint32_t v1 = *reinterpret_cast<const uint32_t*>(r1);
      uint32_t v2 = *reinterpret_cast<const uint32_t*>(r0 + 8);
      uint32_t v3 = *reinterpret_cast<const uint32_t*>(r1 + 8);
      qa[ks][0] = bf16x2_to_f16x2(v0);
      qa[ks][1] = bf16x2_to_f16x2(v1);
      qa[ks][2] = bf16x2_to_f16x2(v2);
      qa[ks][3] = bf16x2_to_f16x2(v3);
      if (!valid0) { qa[ks][0] = 0u; qa[ks][2] = 0u; }
      if (!valid1) { qa[ks][1] = 0u; qa[ks][3] = 0u; }
    }
  }
  float o[DV / 8][4];
#pragma unroll
  for (int n = 0; n < DV / 8; ++n) {
    o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  }
  // row-group running stats: [0] = token t0, [1] = token t0+1
  float m_r[2] = {-1e30f, -1e30f}, l_r[2] = {0.f, 0.f};

  TileLoader<FP8> ld;
  ld.load(cache, bt, page_size, 0, n_keys, tid);
  ld.store(smem, tid);
  __syncthreads();

  for (int t = 0; t < n_tiles; ++t) {
    uint8_t* cur = smem + (t & 1) * SMEM_STAGE;
    uint8_t* nxt = smem + ((t + 1) & 1) * SMEM_STAGE;
    if (t + 1 < n_tiles)
      ld.load(cache, bt, page_size, (t + 1) * TK, n_keys, tid);
    const __half* ks = reinterpret_cast<const __half*>(cur);
    const __half* vs = reinterpret_cast<const __half*>(cur + SMEM_K);
    const int key0 = t * TK;

    // ---- S = Q K^T : 16 rows x 64 keys
    float s[TK / 8][4];
#pragma unroll
    for (int n = 0; n < TK / 8; ++n) {
      s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
    }
#pragma unroll
    for (int kk = 0; kk < DK / 16; ++kk) {
#pragma unroll
      for (int np = 0; np < TK / 16; ++np) {
        const int key = np * 16 + (lane & 15), dd = kk * 16 + (lane >> 4) * 8;
        uint32_t r[4];
        ldmatrix_x4(r, smem_u32(ks + key * KS + dd));
        mma16816(s[2 * np], qa[kk], r[0], r[2]);
        mma16816(s[2 * np + 1], qa[kk], r[1], r[3]);
      }
    }
    // ---- scale + mask: group 0 (token t0) has pos0 = pos1 - 1, so the mask
    // must fire on the EARLIER position: the 64-aligned tile ending exactly
    // at pos1 lets t0 see its next token otherwise (look-ahead leak).
    const bool need_mask =
        (key0 + TK - 1 > pos0) || !valid0 || !valid1;
#pragma unroll
    for (int n = 0; n < TK / 8; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v = s[n][e] * scale_log2;
        if (need_mask) {
          const int key = key0 + n * 8 + 2 * c + (e & 1);
          if (e < 2) {  // c[0],c[1]: row g (token t0)
            if (key > pos0 || !valid0) v = -1e30f;
          } else {      // c[2],c[3]: row g+8 (token t0+1)
            if (key > pos1 || !valid1) v = -1e30f;
          }
        }
        s[n][e] = v;
      }
    }
    // ---- online softmax (base 2)
    float mx[2] = {m_r[0], m_r[1]};
#pragma unroll
    for (int n = 0; n < TK / 8; ++n) {
      mx[0] = fmaxf(mx[0], fmaxf(s[n][0], s[n][1]));
      mx[1] = fmaxf(mx[1], fmaxf(s[n][2], s[n][3]));
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
    }
    float alpha[2] = {exp2f(m_r[0] - mx[0]), exp2f(m_r[1] - mx[1])};
    float rs[2] = {0.f, 0.f};
    uint32_t pa[TK / 16][4];
#pragma unroll
    for (int n = 0; n < TK / 8; ++n) {
      const float p0 = exp2f(s[n][0] - mx[0]), p1 = exp2f(s[n][1] - mx[0]);
      const float p2 = exp2f(s[n][2] - mx[1]), p3 = exp2f(s[n][3] - mx[1]);
      rs[0] += p0 + p1;
      rs[1] += p2 + p3;
      const int kp = n >> 1, hi = n & 1;
      pa[kp][hi ? 2 : 0] = pack_f16x2(p0, p1);
      pa[kp][hi ? 3 : 1] = pack_f16x2(p2, p3);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      rs[r] += __shfl_xor_sync(0xffffffff, rs[r], 1);
      rs[r] += __shfl_xor_sync(0xffffffff, rs[r], 2);
    }
    l_r[0] = l_r[0] * alpha[0] + rs[0];
    l_r[1] = l_r[1] * alpha[1] + rs[1];
    m_r[0] = mx[0];
    m_r[1] = mx[1];
#pragma unroll
    for (int n = 0; n < DV / 8; ++n) {
      o[n][0] *= alpha[0]; o[n][1] *= alpha[0];
      o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
    }
    // ---- O += P V : k = 64 keys, n = 128 d; V row-major -> ldmatrix.trans.
    // Fragment rows: c[0],c[1] = row g (token t0); c[2],c[3] = row g+8 (t0+1).
#pragma unroll
    for (int kp = 0; kp < TK / 16; ++kp) {
#pragma unroll
      for (int np = 0; np < DV / 16; ++np) {
        const int key = kp * 16 + (lane & 15), dd = np * 16 + (lane >> 4) * 8;
        uint32_t r[4];
        ldmatrix_x4_trans(r, smem_u32(vs + key * VS + dd));
        mma16816(o[2 * np], pa[kp], r[0], r[1]);
        mma16816(o[2 * np + 1], pa[kp], r[2], r[3]);
      }
    }
    __syncthreads();
    if (t + 1 < n_tiles) ld.store(nxt, tid);
    __syncthreads();
  }
  // ---- epilogue: O / l * v_descale; (c0,c1) = token t0 head g,
  // (c2,c3) = token t0+1 head g; cols 2c, 2c+1
  if (!valid0 && !valid1) return;
  const float inv0 = valid0 ? v_descale / l_r[0] : 0.f;
  const float inv1 = valid1 ? v_descale / l_r[1] : 0.f;
  __nv_bfloat16* op0 = out + ((size_t)t0 * HQ + g) * DV;
  __nv_bfloat16* op1 = out + ((size_t)(t0 + 1) * HQ + g) * DV;
#pragma unroll
  for (int n = 0; n < DV / 8; ++n) {
    const int dd = n * 8 + 2 * c;
    if (valid0)
      *reinterpret_cast<__nv_bfloat162*>(op0 + dd) =
          __floats2bfloat162_rn(o[n][0] * inv0, o[n][1] * inv0);
    if (valid1)
      *reinterpret_cast<__nv_bfloat162*>(op1 + dd) =
          __floats2bfloat162_rn(o[n][2] * inv1, o[n][3] * inv1);
  }
}

}  // namespace

// qblk_seq / qblk_start: per q-block metadata built by the caller
// (16-token blocks per sequence).
void prefill_attn_sm86(torch::Tensor q, torch::Tensor kv_cache,
                       torch::Tensor block_table, torch::Tensor cu_seqlens_q,
                       torch::Tensor seq_lens, torch::Tensor qblk_seq,
                       torch::Tensor qblk_start, double scale, double k_descale,
                       double v_descale, torch::Tensor out) {
  TORCH_CHECK(q.size(1) == HQ && q.size(2) == DK && kv_cache.size(1) == HKV &&
                  kv_cache.size(3) == KVROW,
              "shape");
  const bool fp8 = kv_cache.dtype() == torch::kFloat8_e4m3fn;
  const int page = kv_cache.size(2), nblk = qblk_seq.size(0);
  auto stream = c10::cuda::getCurrentCUDAStream();
  const float scale_log2 = (float)(scale * k_descale * 1.4426950408889634);
  dim3 grid(nblk, HKV);
  const size_t smem = STAGES * SMEM_STAGE;
  auto launch = [&](auto fp8c) {
    constexpr bool F = decltype(fp8c)::value;
    cudaFuncSetAttribute(prefill_attn_sm86_kernel<F>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    prefill_attn_sm86_kernel<F><<<grid, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        kv_cache.data_ptr(), block_table.data_ptr<int32_t>(),
        (int)block_table.stride(0), page, cu_seqlens_q.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(), qblk_seq.data_ptr<int32_t>(),
        qblk_start.data_ptr<int32_t>(), scale_log2, (float)v_descale,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  };
  if (fp8)
    launch(std::true_type{});
  else
    launch(std::false_type{});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill_attn_sm86", &prefill_attn_sm86);
}
