/*
 * Shared 4096-bin single-CTA TopK helpers.
 */

#ifndef TOPK_HISTOGRAM_4096_CUH_
#define TOPK_HISTOGRAM_4096_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include "core/cuda_compat.h"

namespace vllm {
namespace topk_histogram_4096 {

constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t RADIX = 256;
constexpr uint32_t kMaxTies = 1024;
static_assert(kMaxTies <= kBlockSize,
              "tie_handle requires kMaxTies <= kBlockSize");
// These kernels are written for 32-lane warps. On wave64 each half of a
// wavefront acts as one such warp, so ballots and shuffles stay per half.
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;

// Register path
constexpr uint32_t kHist4096VecsPerThread = 4;
constexpr uint32_t kHist4096MaxLen =
    kHist4096VecsPerThread * 4 * kBlockSize;  // 16384

struct alignas(16) MatchBin {
  uint32_t bin, above_count, equal_count;
};
struct alignas(8) Tie {
  uint32_t idx;
  float score;
};

__device__ __forceinline__ void load_float4_predicated(const float* ptr,
                                                       int base, int seq_len,
                                                       float& v0, float& v1,
                                                       float& v2, float& v3) {
#ifdef USE_ROCM
  v0 = base < seq_len ? ptr[0] : -__builtin_huge_valf();
  v1 = base + 1 < seq_len ? ptr[1] : -__builtin_huge_valf();
  v2 = base + 2 < seq_len ? ptr[2] : -__builtin_huge_valf();
  v3 = base + 3 < seq_len ? ptr[3] : -__builtin_huge_valf();
#else
  uint32_t r0, r1, r2, r3;
  const int p0 = (base < seq_len);
  const int p1 = (base + 1 < seq_len);
  const int p2 = (base + 2 < seq_len);
  const int p3 = (base + 3 < seq_len);
  asm volatile(
      "{\n"
      "  .reg .pred pr0, pr1, pr2, pr3;\n"
      "  setp.ne.u32 pr0, %4, 0;\n"
      "  setp.ne.u32 pr1, %5, 0;\n"
      "  setp.ne.u32 pr2, %6, 0;\n"
      "  setp.ne.u32 pr3, %7, 0;\n"
      "  mov.u32 %0, 0xFF800000;\n"
      "  mov.u32 %1, 0xFF800000;\n"
      "  mov.u32 %2, 0xFF800000;\n"
      "  mov.u32 %3, 0xFF800000;\n"
      "  @pr0 ld.global.cg.u32 %0, [%8];\n"
      "  @pr1 ld.global.cg.u32 %1, [%8+4];\n"
      "  @pr2 ld.global.cg.u32 %2, [%8+8];\n"
      "  @pr3 ld.global.cg.u32 %3, [%8+12];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
#endif
}

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float4 load_score4(const float* ptr) {
  return *reinterpret_cast<const float4*>(ptr);
}
__device__ __forceinline__ float4 load_score4(const __nv_bfloat16* ptr) {
  const uint2 raw = *reinterpret_cast<const uint2*>(ptr);
  return make_float4(
      __uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xFFFF0000u),
      __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xFFFF0000u));
}
__device__ __forceinline__ float2 load_score2(const float* ptr) {
  return *reinterpret_cast<const float2*>(ptr);
}
__device__ __forceinline__ float2 load_score2(const __nv_bfloat16* ptr) {
  const uint32_t raw = *reinterpret_cast<const uint32_t*>(ptr);
  return make_float2(__uint_as_float(raw << 16),
                     __uint_as_float(raw & 0xFFFF0000u));
}

__device__ __forceinline__ void load_float4_predicated(const __nv_bfloat16* ptr,
                                                       int base, int seq_len,
                                                       float& v0, float& v1,
                                                       float& v2, float& v3) {
  v0 = base < seq_len ? to_float(ptr[0]) : __uint_as_float(0xFF800000u);
  v1 = base + 1 < seq_len ? to_float(ptr[1]) : __uint_as_float(0xFF800000u);
  v2 = base + 2 < seq_len ? to_float(ptr[2]) : __uint_as_float(0xFF800000u);
  v3 = base + 3 < seq_len ? to_float(ptr[3]) : __uint_as_float(0xFF800000u);
}

// converts the float32 score to a 32-bit ordered unsigned integer — the full
// precision key for radix sorting
__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  // Read the bits through an opaque move: otherwise nvcc lowers
  // `bits | 0x80000000u` to the FP op -|x|, which canonicalizes NaN to
  // 0x7FFFFFFF (the key of -0.0) while the FP16 coarse bin puts NaN on top.
#ifdef USE_ROCM
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
#else
  uint32_t bits;
  asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(x));
  // cvt.rn.f16.f32 drops a NaN's sign, so the FP16 coarse bins put every NaN
  // on top; key sign-set NaNs above +inf too.
  return static_cast<int32_t>(bits) > static_cast<int32_t>(0xFF800000u)
             ? (bits | 0x80000000u)
             : ~bits;
#endif
}

// Converts each score to a 12-bit bin (FP16 sign-magnitude -> top 12 bits ->
// bin 0-4095)
template <uint32_t kBits>
__device__ __forceinline__ uint32_t extract_coarse_bin_N(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

// Coarse bins of the two BF16 scores in `w` (low half first): the bins
// extract_coarse_bin_N gives, from one packed FP16 conversion and key flip.
template <uint32_t kBits>
__device__ __forceinline__ uint2 extract_coarse_bins_bf16x2(uint32_t w) {
  const __half2 h = __floats2half2_rn(__uint_as_float(w << 16),
                                      __uint_as_float(w & 0xFFFF0000u));
  const uint32_t bits = *reinterpret_cast<const uint32_t*>(&h);
  const uint32_t sign = bits & 0x80008000u;
  // Per half: ~bits if negative, else bits | 0x8000.
  const uint32_t key = bits ^ ((sign - (sign >> 15)) | 0x80008000u);
  return make_uint2((key & 0xFFFFu) >> (16 - kBits), key >> (32 - kBits));
}

// Four consecutive scores held in registers: FP32 as float4, BF16 packed in a
// uint2 until binned.
template <typename InT>
using ScoreVec4 = std::conditional_t<sizeof(InT) == 2, uint2, float4>;

__device__ __forceinline__ float vec_score(const float4& v, uint32_t e) {
  return reinterpret_cast<const float*>(&v)[e];
}
__device__ __forceinline__ float vec_score(const uint2& v, uint32_t e) {
  const uint32_t w = e < 2 ? v.x : v.y;
  return __uint_as_float(e % 2 ? w & 0xFFFF0000u : w << 16);
}

// Coarse bin of score e: BF16 bins all four scores up front with two packed
// conversions (`packed`), FP32 bins each score when it is used.
template <uint32_t kBits>
__device__ __forceinline__ uint4 packed_bins(const float4&) {
  return {};
}
template <uint32_t kBits>
__device__ __forceinline__ uint4 packed_bins(const uint2& v) {
  const uint2 lo = extract_coarse_bins_bf16x2<kBits>(v.x);
  const uint2 hi = extract_coarse_bins_bf16x2<kBits>(v.y);
  return make_uint4(lo.x, lo.y, hi.x, hi.y);
}
template <uint32_t kBits>
__device__ __forceinline__ uint32_t vec_bin(const float4& v, const uint4&,
                                            uint32_t e) {
  return extract_coarse_bin_N<kBits>(vec_score(v, e));
}
template <uint32_t kBits>
__device__ __forceinline__ uint32_t vec_bin(const uint2&, const uint4& packed,
                                            uint32_t e) {
  return reinterpret_cast<const uint32_t*>(&packed)[e];
}

__device__ __forceinline__ void load_vec4(const float* ptr, float4& v) {
  v = *reinterpret_cast<const float4*>(ptr);
}
__device__ __forceinline__ void load_vec4(const __nv_bfloat16* ptr, uint2& v) {
  v = *reinterpret_cast<const uint2*>(ptr);
}
// Scores at or past `length` load as -inf.
__device__ __forceinline__ void load_vec4_predicated(const float* ptr, int base,
                                                     int length, float4& v) {
  load_float4_predicated(ptr, base, length, v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ void load_vec4_predicated(const __nv_bfloat16* ptr,
                                                     int base, int length,
                                                     uint2& v) {
  const auto* raw = reinterpret_cast<const uint16_t*>(ptr);
  uint32_t h[4];
#pragma unroll
  for (int e = 0; e < 4; e++) h[e] = base + e < length ? raw[e] : 0xFF80u;
  v = make_uint2(h[0] | (h[1] << 16), h[2] | (h[3] << 16));
}

// running sum within each warp — thread 0 gets its own value, thread 1 gets
// thread 0 + thread 1, thread 2 gets threads 0+1+2, etc.
__device__ __forceinline__ uint32_t warp_inclusive_sum(uint32_t lane,
                                                       uint32_t v) {
#pragma unroll
  for (uint32_t o = 1; o < 32; o *= 2) {
    uint32_t n = VLLM_SHFL_UP_SYNC_WIDTH(v, o, kWarpSize);
    if (lane >= o) v += n;
  }
  return v;
}

// Returns the sum of a value across all 32 threads in the warp, and every
// thread gets the same result. SM80+ uses redux.sync.add.u32, a single PTX
// instruction for hardware warp-wide reduction. Older targets use the
// __shfl_xor_sync butterfly tree, like warp::reduce_sum() (5 shuffles for 32
// lanes).
__device__ __forceinline__ uint32_t warp_reduce_sum_full(uint32_t v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t r;
  asm("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
  return r;
#else
  #pragma unroll
  for (uint32_t mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    v += VLLM_SHFL_XOR_SYNC_WIDTH(v, mask, kWarpSize);
  }
  return v;
#endif
}

// Bitwise OR across the warp.
__device__ __forceinline__ uint32_t warp_reduce_or_full(uint32_t v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_or_sync(0xFFFFFFFF, v);
#else
  #pragma unroll
  for (uint32_t mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    v |= VLLM_SHFL_XOR_SYNC_WIDTH(v, mask, kWarpSize);
  }
  return v;
#endif
}

// ============================================================================
// Deterministic emit (#50576). Claiming output slots with first-come-first-
// served atomics makes the output order, and the set chosen among keys equal
// to the pivot, depend on warp scheduling; the sparse indexer feeds the indices
// to an order-sensitive online softmax. The sweeps below instead visit
// contiguous index spans and place every selected index with a block prefix
// scan: ascending index order, lowest indices winning pivot ties.
// ============================================================================

// Block-wide exclusive prefix (.x) and total (.y) of a per-thread packed
// (definite << 16 | tie) count. Sweeps alternate between two kNumWarps-entry
// `warp_sum` buffers so one barrier per iteration suffices.
__device__ __forceinline__ uint2 block_packed_prefix(uint32_t packed,
                                                     uint32_t* warp_sum) {
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  const uint32_t inclusive = warp_inclusive_sum(lane, packed);
  if (lane == kWarpSize - 1) warp_sum[warp] = inclusive;
  __syncthreads();
  const uint32_t warp_total = warp_sum[lane];
  const uint32_t total = warp_reduce_sum_full(warp_total);
  const uint32_t before = warp_reduce_sum_full(lane < warp ? warp_total : 0);
  return make_uint2(before + inclusive - packed, total);
}

// Write this thread's selected indices `base + e` for the set bits of
// def_bits/tie_bits. A definite index lands after the definites and admitted
// ties before it; a tie is admitted while fewer than tie_slots precede it.
template <uint32_t Elems>
__device__ __forceinline__ void write_index_ordered(
    int32_t* __restrict__ output, uint32_t top_k, uint32_t base,
    uint32_t def_bits, uint32_t tie_bits, uint32_t def_prefix,
    uint32_t tie_prefix, uint32_t tie_slots) {
#pragma unroll
  for (uint32_t e = 0; e < Elems; e++) {
    if (def_bits & (1u << e)) {
      const uint32_t pos = def_prefix + min(tie_prefix, tie_slots);
      if (pos < top_k) output[pos] = static_cast<int32_t>(base + e);
      def_prefix++;
    } else if (tie_bits & (1u << e)) {
      const uint32_t pos = def_prefix + tie_prefix;
      if (tie_prefix < tie_slots && pos < top_k) {
        output[pos] = static_cast<int32_t>(base + e);
      }
      tie_prefix++;
    }
  }
}

// Emit the indices of keys > pivot plus the first tie_slots keys == pivot, in
// ascending index order; together they must number top_k. `warp_sums` is
// 2 * kNumWarps words of shared scratch.
template <uint32_t VecSize, typename InT>
__device__ void emit_index_ordered(const InT* __restrict__ scores,
                                   int32_t* __restrict__ output,
                                   uint32_t length, uint32_t top_k,
                                   uint32_t pivot, uint32_t tie_slots,
                                   uint32_t* warp_sums) {
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  const uint32_t tx = threadIdx.x;
  uint32_t def_run = 0, tie_run = 0;
  // The scratch may alias state other threads are still reading.
  __syncthreads();
  for (uint32_t chunk = 0, iter = 0; chunk < length;
       chunk += kBlockSize * VecSize, ++iter) {
    const uint32_t base = chunk + tx * VecSize;
    float values[VecSize];
    if (base + VecSize <= length) {
      if constexpr (VecSize == 4) {
        const float4 v = load_score4(scores + base);
        values[0] = v.x;
        values[1] = v.y;
        values[2] = v.z;
        values[3] = v.w;
      } else if constexpr (VecSize == 2) {
        const float2 v = load_score2(scores + base);
        values[0] = v.x;
        values[1] = v.y;
      } else {
        values[0] = to_float(scores[base]);
      }
    } else {
#pragma unroll
      for (uint32_t e = 0; e < VecSize; e++) {
        values[e] = base + e < length ? to_float(scores[base + e]) : 0.0f;
      }
    }
    uint32_t def_bits = 0, tie_bits = 0;
#pragma unroll
    for (uint32_t e = 0; e < VecSize; e++) {
      if (base + e >= length) continue;
      const uint32_t key = convert_to_uint32_v2(values[e]);
      if (key > pivot) {
        def_bits |= 1u << e;
      } else if (key == pivot) {
        tie_bits |= 1u << e;
      }
    }
    const uint32_t packed = (static_cast<uint32_t>(__popc(def_bits)) << 16) |
                            static_cast<uint32_t>(__popc(tie_bits));
    const uint2 scan =
        block_packed_prefix(packed, warp_sums + (iter & 1) * kNumWarps);
    write_index_ordered<VecSize>(output, top_k, base, def_bits, tie_bits,
                                 def_run + (scan.x >> 16),
                                 tie_run + (scan.x & 0xFFFF), tie_slots);
    def_run += scan.y >> 16;
    tie_run += scan.y & 0xFFFF;
  }
  __syncthreads();
}

// The first and last radix-256 rounds (round r covers key bits 31-8r..24-8r)
// that can split ties whose keys differ from a reference key in `diff`: the
// ties agree on every other bit. Equal keys take one round, to place them.
__device__ __forceinline__ uint2 tie_rounds(uint32_t diff) {
  return diff ? make_uint2(__clz(diff) / 8, (32 - __ffs(diff)) / 8)
              : make_uint2(0, 0);
}

// ============================================================================
// Tie refinement (single CTA): radix-256 topK on the full FP32 ordered key.
// Each round narrows by 8 bits until ties are fully resolved. With
// kSkipSharedRounds, rounds over bits all ties share are skipped (BF16 scores
// widen to keys ending in 16 zeros).
// ============================================================================

template <uint32_t TopK, bool kSkipSharedRounds = true>
__device__ void tie_handle(const Tie* ties, uint32_t num_ties,
                           uint32_t num_above, int32_t* output, void* _smem) {
  struct TS {
    alignas(128) uint32_t counter;
    alignas(128) MatchBin match;
    uint32_t histogram[RADIX];
    uint32_t warp_sum[kNumWarps];
  };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize, wi = tx / kWarpSize;

  // Each thread loads one tie element.
  const bool has = tx < num_ties;
  const auto tie = has ? ties[tx] : Tie{0, 0.0f};
  const uint32_t key = convert_to_uint32_v2(tie.score);
  const uint32_t ref =
      kSkipSharedRounds ? convert_to_uint32_v2(ties[0].score) : 0;

  bool active = has;  // tracks whether this thread's tie is still a candidate.
  uint32_t remain =
      TopK - num_above;  // decreases each round as ties are resolved.
  uint32_t wpos = TopK;  // wpos will hold the final output position.
  s->counter = 0;
  __syncthreads();  // ties are in registers; `s` may overlap them

  // One radix round over key bits 31-8r..24-8r; returns whether all ties are
  // resolved.
  auto round = [&](uint32_t r, bool last) {
    uint32_t sh = 24 - r * 8;  // round 0: bits 31-24, round 1: 23-16, etc.
    uint32_t bin = (key >> sh) & 0xFF;  // this tie's 8-bit bin for this round

    // Step 1: Build 256-bin histogram.
    if (tx < RADIX) s->histogram[tx] = 0;
    __syncthreads();
    if (active) atomicAdd(&s->histogram[bin], 1);
    __syncthreads();

    // Step 2: Prefix scan to find threshold
    uint32_t hv = 0, wi2 = 0;
    if (tx < RADIX) {
      hv = s->histogram[tx];
      wi2 = warp_inclusive_sum(li, hv);
      if (li == kWarpSize - 1) s->warp_sum[wi] = wi2;
    }
    __syncthreads();

    if (tx < RADIX) {
      auto tmp = (li < RADIX / kWarpSize) ? s->warp_sum[li] : 0;
      auto tot = warp_reduce_sum_full(tmp);
      auto inter = warp_reduce_sum_full(li < wi ? tmp : 0);
      auto above = tot - (inter + wi2);
      if (above < remain && above + hv >= remain) {
        s->match = {tx, above, remain - above};
      }
    }
    __syncthreads();

    // Step 3: Scatter
    auto [thr, na, _] = s->match;  // threshold bin, num above, unused
    if (active) {
      if (bin > thr) {
        wpos = num_above +
               atomicAdd(&s->counter, 1);  // above -> place in output directly
        active = false;
      } else if (bin < thr)
        active = false;  // below -> discard
      else if (last)
        wpos = TopK - atomicAdd(&s->match.equal_count,
                                -1u);  // last round: place remaining
    }
    remain -= na;
    return remain == 0;
  };
  if constexpr (kSkipSharedRounds) {
    const uint32_t warp_diff = warp_reduce_or_full(has ? key ^ ref : 0);
    if (li == 0) s->warp_sum[wi] = warp_diff;
    __syncthreads();
    const uint2 rounds = tie_rounds(warp_reduce_or_full(s->warp_sum[li]));
    for (uint32_t r = rounds.x; r <= rounds.y; r++) {
      if (round(r, r == rounds.y)) break;
    }
  } else {
#pragma unroll
    for (uint32_t r = 0; r < 4; r++) {
      if (round(r, r == 3)) break;
    }
  }
  // Final write
  if (wpos < TopK) output[wpos] = tie.idx;
}

// Extended tie_handle for TopK > kBlockSize (e.g. TopK=2048).
// tie_handle assumes 1 tie per thread (max 1024).
// This version handles 2 ties per thread via kPerThread=2
template <uint32_t TopK, uint32_t MaxTies = TopK>
__device__ void tie_handle_large(const Tie* ties, uint32_t num_ties,
                                 uint32_t num_above, int32_t* output,
                                 void* _smem) {
  static_assert(MaxTies > kBlockSize);
  struct TS {
    alignas(128) uint32_t counter;
    alignas(128) MatchBin match;
    uint32_t histogram[RADIX];
    uint32_t warp_sum[kNumWarps];
  };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize;
  const auto wi = tx / kWarpSize;

  constexpr uint32_t kPerThread = (MaxTies + kBlockSize - 1) / kBlockSize;
  Tie my_ties[kPerThread];
  uint32_t keys[kPerThread];
  bool active[kPerThread];

  const uint32_t ref = convert_to_uint32_v2(ties[0].score);
  uint32_t diff = 0;
  for (uint32_t e = 0; e < kPerThread; e++) {
    uint32_t idx = e * kBlockSize + tx;
    if (idx < num_ties) {
      my_ties[e] = ties[idx];
      keys[e] = convert_to_uint32_v2(ties[idx].score);
      active[e] = true;
      diff |= keys[e] ^ ref;
    } else {
      my_ties[e] = {0, 0.0f};
      keys[e] = 0;
      active[e] = false;
    }
  }

  uint32_t remain = TopK - num_above;
  s->counter = 0;
  __syncthreads();  // ties are in registers; `s` may overlap them
  const uint32_t warp_diff = warp_reduce_or_full(diff);
  if (li == 0) s->warp_sum[wi] = warp_diff;
  __syncthreads();
  const uint2 rounds = tie_rounds(warp_reduce_or_full(s->warp_sum[li]));

  for (uint32_t r = rounds.x; r <= rounds.y; r++) {
    uint32_t sh = 24 - r * 8;
    if (tx < RADIX) {
      s->histogram[tx] = 0;
    }
    __syncthreads();

    for (uint32_t e = 0; e < kPerThread; e++) {
      if (active[e]) {
        atomicAdd(&s->histogram[(keys[e] >> sh) & 0xFF], 1);
      }
    }
    __syncthreads();

    uint32_t hv = 0;
    if (tx < RADIX) {
      hv = s->histogram[tx];
      auto wi2 = warp_inclusive_sum(li, hv);
      if (li == kWarpSize - 1) {
        s->warp_sum[wi] = wi2;
      }
    }
    __syncthreads();
    if (tx < RADIX) {
      auto tmp2 = (li < RADIX / kWarpSize) ? s->warp_sum[li] : 0;
      auto total = warp_reduce_sum_full(tmp2);
      auto inter = warp_reduce_sum_full(li < wi ? tmp2 : 0);
      auto wi2 = warp_inclusive_sum(li, hv);
      auto above = total - (inter + wi2);
      if (above < remain && above + hv >= remain) {
        s->match = {
            .bin = tx, .above_count = above, .equal_count = remain - above};
      }
    }
    __syncthreads();

    auto thr = s->match.bin;
    auto na = s->match.above_count;

    for (uint32_t e = 0; e < kPerThread; e++) {
      if (!active[e]) {
        continue;
      }
      uint32_t bin = (keys[e] >> sh) & 0xFF;
      if (bin > thr) {
        uint32_t wpos = num_above + atomicAdd(&s->counter, 1);
        if (wpos < TopK) {
          output[wpos] = my_ties[e].idx;
        }
        active[e] = false;
      } else if (bin < thr) {
        active[e] = false;
      } else if (r == rounds.y) {
        uint32_t wpos = TopK - atomicAdd(&s->match.equal_count, -1u);
        if (wpos < TopK) {
          output[wpos] = my_ties[e].idx;
        }
      }
    }

    num_above += na;
    remain -= na;
    __syncthreads();
    s->counter = 0;
    __syncthreads();
  }
}

// ============================================================================
// Register-based single-CTA fast path for seq_len <= 16384
// 4 float4 per thread × 1024 threads = 16384 elements max
// Uses 4096-bin (12-bit) histogram for better precision
// ============================================================================

template <uint32_t TopK, uint32_t HIST_BITS>
struct Histogram4096Smem {
  static constexpr uint32_t HIST_BINS = 1 << HIST_BITS;
  static constexpr uint32_t TIE_CAPACITY = TopK > kMaxTies ? TopK : kMaxTies;
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union {
    uint32_t histogram[HIST_BINS];
    Tie tie_buffer[TIE_CAPACITY];
  };
};

// Defined in topk_histogram_4096_overflow.cuh. Only kFp32Rescan instantiations
// call it, and their callers include that header.
template <uint32_t TopK, uint32_t BlockSize, uint32_t VecSize, typename InT>
__device__ __noinline__ void exact_topk_rescan_cold(
    const InT* __restrict__ scores, int32_t* __restrict__ output,
    uint32_t length, void* smem);

// kRegisterValues refines from the values already held in registers and
// falls back to the FP32 rescan when the coarse bin cannot bound them.
enum class OverflowRecovery { kFp32Rescan, kRegisterValues };

// Defined in topk_histogram_4096_overflow.cuh, like exact_topk_rescan_cold.
template <uint32_t TopK, uint32_t HistBits, uint32_t VecsPerThread,
          typename Vec>
__device__ __forceinline__ bool exact_topk_refine_registers(
    const Vec (&vecs)[VecsPerThread], int32_t* __restrict__ output,
    uint32_t length, uint32_t coarse_bin, uint32_t num_above, void* _smem);

template <uint32_t TopK, uint32_t BlockSize, uint32_t VecSize = 1,
          uint32_t StartRound = 0, uint32_t FinalCandidateCapacity = 0,
          typename InT>
__device__ void exact_topk_rescan(const InT* __restrict__ scores,
                                  int32_t* __restrict__ output, uint32_t length,
                                  void* _smem, uint32_t initial_prefix = 0,
                                  uint32_t initial_remaining = TopK);

template <uint32_t TopK, uint32_t HIST_BITS,
          uint32_t VECS_PER_THREAD = kHist4096VecsPerThread,
          bool UsePredicatedLoads = false, OverflowRecovery Recovery,
          typename InT>
__device__ void histogram_4096_topk(const InT* __restrict__ scores,
                                    int32_t* __restrict__ output,
                                    uint32_t length, void* _smem) {
  constexpr uint32_t HIST_BINS = 1 << HIST_BITS;
  constexpr uint32_t ITEMS_PER_THREAD = HIST_BINS / kBlockSize;
  static_assert(HIST_BINS >= kBlockSize,
                "HIST_BITS must give >= kBlockSize bins");

  using Smem = Histogram4096Smem<TopK, HIST_BITS>;
  auto* smem = static_cast<Smem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpSize;
  const auto warp_id = tx / kWarpSize;

  // Phase 1: Load all data into RF + build histogram
  ScoreVec4<InT> vecs[VECS_PER_THREAD];  // 4 vectors x 4 = 16 elements/thread
  if constexpr (ITEMS_PER_THREAD >= 4) {
    // Zero the histogram (SMEM writes)
    for (uint32_t i = 0; i < ITEMS_PER_THREAD / 4; i++)
      reinterpret_cast<uint4*>(
          smem->histogram)[tx * (ITEMS_PER_THREAD / 4) + i] =
          make_uint4(0, 0, 0, 0);
  } else {
    if (tx < HIST_BINS) smem->histogram[tx] = 0;
  }
  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  if constexpr (UsePredicatedLoads || sizeof(InT) == 2) {
    const bool row_aligned =
        (reinterpret_cast<uintptr_t>(scores) % sizeof(vecs[0])) == 0;
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
      const uint32_t base = (tx + v * kBlockSize) * 4;
      if (base < length) {
        if (row_aligned && base + 3 < length) {
          load_vec4(scores + base, vecs[v]);
        } else {
          load_vec4_predicated(scores + base, static_cast<int>(base),
                               static_cast<int>(length), vecs[v]);
        }
      }
    }
  } else {
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
      const uint32_t base = (tx + v * kBlockSize) * 4;
      if (base < length) {
        load_vec4(scores + base, vecs[v]);
      }
    }
  }
  __syncthreads();

  // Build histogram from RF via atomic adds into the shared histogram
  bool done = false;
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
    const uint4 bins = packed_bins<HIST_BITS>(vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4 && !done; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) {
        done = true;
      } else {
        atomicAdd(&smem->histogram[vec_bin<HIST_BITS>(vecs[v], bins, e)], 1);
      }
    }
  }
  __syncthreads();

  // Phase 2: Prefix scan to find threshold bin
  // Multi-element scan (4096 bins: 4 per thread)
  uint32_t orig[ITEMS_PER_THREAD];
  uint32_t local_sum = 0;

  // Step 1: Each thread sums its 4 bins
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    orig[i] = smem->histogram[tx * ITEMS_PER_THREAD + i];
    local_sum += orig[i];
  }

  // Step 2: Warp-level inclusive prefix sum on local_sum
  const auto warp_inc = warp_inclusive_sum(lane_id, local_sum);
  if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
  __syncthreads();

  // Step 3: Inter-warp prefix across warp sums.
  const auto tmp = smem->warp_sum[lane_id];
  uint32_t prefix = warp_reduce_sum_full(
      lane_id < warp_id ? tmp : 0);  // sum of all prior warps
  prefix +=
      warp_inc - local_sum;  // exclusive prefix within this thread's position

  // Step 4: Find threshold - scan 4 bins, accumulate prefix
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    prefix += orig[i];
    const auto above = length - prefix;  // elements in bins ABOVE this one
    if (above < TopK && above + orig[i] >= TopK) {
      smem->match = {.bin = tx * ITEMS_PER_THREAD + i,
                     .above_count = above,
                     .equal_count = orig[i]};
    }
  }

  __syncthreads();

  // Phase 3: exact-threshold refinement, deterministic and buffer-free
  // (#50576): radix-refine the threshold coarse bin down to the exact ordered
  // key and the number of slots left among keys equal to it. Only histogram
  // counts are involved, so no coarse-bin occupancy can overflow a buffer and
  // `Recovery` is not needed.
  const auto [thr_bin, num_above, num_equal] = smem->match;
  const bool need_tie = (num_equal + num_above > TopK);

  uint32_t exact_thr = 0;  // full ordered key of the k-th value
  uint32_t tie_slots = TopK - num_above;  // slots for key == exact_thr

  if (need_tie) {
    static_assert(VECS_PER_THREAD * 4 <= 32, "one candidate bit per element");
    // Candidates: coarse bin == thr_bin and key matching the refined prefix.
    uint32_t active = 0;
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
      const uint4 bins = packed_bins<HIST_BITS>(vecs[v]);
#pragma unroll
      for (uint32_t e = 0; e < 4; e++) {
        const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
        if (idx < length && vec_bin<HIST_BITS>(vecs[v], bins, e) == thr_bin) {
          active |= 1u << (v * 4 + e);
        }
      }
    }

    uint32_t remain = TopK - num_above;
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const uint32_t sh = 24 - r * 8;
      if (tx < RADIX) smem->histogram[tx] = 0;
      __syncthreads();

#pragma unroll
      for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
#pragma unroll
        for (uint32_t e = 0; e < 4; e++) {
          if (active & (1u << (v * 4 + e))) {
            const uint32_t key = convert_to_uint32_v2(vec_score(vecs[v], e));
            atomicAdd(&smem->histogram[(key >> sh) & 0xFF], 1);
          }
        }
      }
      __syncthreads();

      // Suffix counts over the 256 sub-bins to locate the threshold sub-bin.
      uint32_t hv = 0, winc = 0;
      if (tx < RADIX) {
        hv = smem->histogram[tx];
        winc = warp_inclusive_sum(lane_id, hv);
        if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = winc;
      }
      __syncthreads();
      if (tx < RADIX) {
        const auto tmp =
            (lane_id < RADIX / kWarpSize) ? smem->warp_sum[lane_id] : 0;
        const auto tot = warp_reduce_sum_full(tmp);
        const auto inter = warp_reduce_sum_full(lane_id < warp_id ? tmp : 0);
        const auto above = tot - (inter + winc);
        if (above < remain && above + hv >= remain) {
          smem->match = {.bin = tx, .above_count = above, .equal_count = hv};
        }
      }
      __syncthreads();

      const uint32_t sub_bin = smem->match.bin;
      remain -= smem->match.above_count;
      exact_thr |= sub_bin << sh;

      // Narrow the candidates to the chosen sub-bin.
      uint32_t next_active = 0;
#pragma unroll
      for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
#pragma unroll
        for (uint32_t e = 0; e < 4; e++) {
          if (active & (1u << (v * 4 + e))) {
            const uint32_t key = convert_to_uint32_v2(vec_score(vecs[v], e));
            if (((key >> sh) & 0xFF) == sub_bin) {
              next_active |= 1u << (v * 4 + e);
            }
          }
        }
      }
      active = next_active;
      __syncthreads();
    }
    tie_slots = remain;
  }

  // Phase 4: index-ordered selection sweep from registers. Iteration v covers
  // the contiguous index span [v*4*kBlockSize, (v+1)*4*kBlockSize).
  uint32_t* sweep_sums = smem->histogram;  // free again: 2 * kNumWarps words
  static_assert(HIST_BINS >= 2 * kNumWarps);
  uint32_t def_run = 0, tie_run = 0;
  __syncthreads();
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
    const uint4 bins = packed_bins<HIST_BITS>(vecs[v]);
    const uint32_t base = (tx + v * kBlockSize) * 4;
    uint32_t def_bits = 0, tie_bits = 0;
#pragma unroll
    for (uint32_t e = 0; e < 4; e++) {
      if (base + e >= length) continue;
      const uint32_t bin = vec_bin<HIST_BITS>(vecs[v], bins, e);
      if (bin > thr_bin) {
        def_bits |= 1u << e;
      } else if (bin == thr_bin) {
        if (!need_tie) {
          def_bits |= 1u << e;
        } else {
          const uint32_t key = convert_to_uint32_v2(vec_score(vecs[v], e));
          if (key > exact_thr) {
            def_bits |= 1u << e;
          } else if (key == exact_thr) {
            tie_bits |= 1u << e;
          }
        }
      }
    }

    const uint32_t packed = (static_cast<uint32_t>(__popc(def_bits)) << 16) |
                            static_cast<uint32_t>(__popc(tie_bits));
    const uint2 scan =
        block_packed_prefix(packed, sweep_sums + (v & 1) * kNumWarps);
    write_index_ordered<4>(output, TopK, base, def_bits, tie_bits,
                           def_run + (scan.x >> 16),
                           tie_run + (scan.x & 0xFFFF), tie_slots);
    def_run += scan.y >> 16;
    tie_run += scan.y & 0xFFFF;
  }
}

template <uint32_t TopK, uint32_t HIST_BITS,
          uint32_t VECS_PER_THREAD = kHist4096VecsPerThread,
          OverflowRecovery Recovery, typename InT>
__device__ __noinline__ void histogram_4096_topk_predicated(
    const InT* __restrict__ scores, int32_t* __restrict__ output,
    uint32_t length, void* _smem) {
  histogram_4096_topk<TopK, HIST_BITS, VECS_PER_THREAD, true, Recovery>(
      scores, output, length, _smem);
}

}  // namespace topk_histogram_4096
}  // namespace vllm

#endif  // TOPK_HISTOGRAM_4096_CUH_
