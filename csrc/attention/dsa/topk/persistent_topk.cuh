/*
 * Persistent TopK Scheduler for DSA Indexer
 */

#ifndef PERSISTENT_TOPK_CUH_
#define PERSISTENT_TOPK_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>

#include "topk_histogram_4096.cuh"
#include "persistent_topk_overflow.cuh"

namespace vllm {
namespace persistent {

using topk_histogram_4096::convert_to_uint32_v2;

// ============================================================================
// Constants
// ============================================================================

constexpr int kThreadsPerBlock = 1024;
constexpr int RADIX = 256;

// Medium path: all shared state in dynamic smem (no static __shared__,
// which would inflate the kernel's smem footprint and kill occupancy
// for the short and trivial paths).
constexpr size_t kMediumHistBytes = 2 * (RADIX + 128) * sizeof(int);  // 3072
constexpr size_t kMediumScalarsBytes = 5 * sizeof(int);               // 20
constexpr size_t kMediumHeaderSize =
    (kMediumHistBytes + kMediumScalarsBytes + 127) & ~size_t(127);  // 3200
constexpr int MAX_BUFFERED_ITEMS = 4096;
constexpr size_t kSmemMedium =
    kMediumHeaderSize + 2 * MAX_BUFFERED_ITEMS * sizeof(int);  // 35968
static_assert(topk_histogram_4096::kExactCandidateOffset +
                  MAX_BUFFERED_ITEMS * sizeof(int) <=
              kSmemMedium);
constexpr uint32_t RADIX_THRESHOLD = 32768;

// Short and medium path constants
constexpr int kCoarseBins = 2048;
constexpr uint32_t SHORT_EXACT_THRESHOLD = 8192;

// Large path: fixed shared memory for histograms + scalars
constexpr size_t kFixedSmemLarge =
    ((RADIX + RADIX + 5) * sizeof(uint32_t) + 15) & ~size_t(15);

// ============================================================================
// Large path: inter-CTA coordination state (one per group)
// ============================================================================

struct RadixRowState {
  uint32_t histogram[3][256];  // Triple-buffered histograms
  uint32_t remaining_k;
  uint32_t prefix;
  int arrival_counter;
  int output_counter;
};

// ============================================================================
// Kernel parameters
// ============================================================================

template <typename InT>
struct PersistentTopKParams {
  const InT* __restrict__ input;        // [num_rows, stride]
  int32_t* __restrict__ output;         // [num_rows, top_k]
  const int32_t* __restrict__ lengths;  // [num_rows]
  RadixRowState* row_states;            // large path: per-group state
  uint32_t num_rows;
  uint32_t stride;
  uint32_t top_k;           // actual k value for output stride
  uint32_t chunk_size;      // large path: elements per CTA
  uint32_t ctas_per_group;  // 1=medium, >1=large
  uint32_t max_seq_len;     // max seq_len across all rows (for early CTA exit)
};

// ============================================================================
// Coarse selection shared by the medium and filtered paths
// ============================================================================

template <int BlockSize, int Bins>
struct CoarseSuffixSmem {
  int histogram[Bins + 1];
  uint32_t warp_sum[BlockSize / 32];
};

// Rewrite the coarse counts as suffix counts (histogram[b] = keys in bins >= b)
// and call on_select(bin) from the one thread whose bin holds rank k. Uses the
// shuffle scan of histogram_4096_topk: a CUB block scan spills under the
// persistent kernel's 32-register budget.
template <int BlockSize, int Bins, typename OnSelect>
__device__ __forceinline__ void select_coarse_bin(
    CoarseSuffixSmem<BlockSize, Bins>* coarse, int k, OnSelect on_select) {
  namespace hist4096 = topk_histogram_4096;
  constexpr int kBinsPerThread = Bins / BlockSize;
  static_assert(Bins % BlockSize == 0);
  static_assert(BlockSize == 32 * 32, "one warp sum per lane");
  const int tx = threadIdx.x;
  const uint32_t lane = tx % 32;
  const uint32_t warp = tx / 32;
  const int first_bin = kBinsPerThread * tx;
  int counts[kBinsPerThread];
  uint32_t local_sum = 0;
#pragma unroll
  for (int i = 0; i < kBinsPerThread; ++i) {
    counts[i] = coarse->histogram[first_bin + i];
    local_sum += counts[i];
  }
  const uint32_t warp_inclusive = hist4096::warp_inclusive_sum(lane, local_sum);
  if (lane == 31) coarse->warp_sum[warp] = warp_inclusive;
  __syncthreads();
  const uint32_t warp_total = coarse->warp_sum[lane];
  const uint32_t total = hist4096::warp_reduce_sum_full(warp_total);
  const uint32_t warps_before =
      hist4096::warp_reduce_sum_full(lane < warp ? warp_total : 0);
  // Each thread rewrites only its own bins, so no barrier is needed here.
  int suffix =
      static_cast<int>(total - warps_before - warp_inclusive + local_sum);
#pragma unroll
  for (int i = 0; i < kBinsPerThread; ++i) {
    coarse->histogram[first_bin + i] = suffix;
    suffix -= counts[i];
  }
  if (tx == 0) coarse->histogram[Bins] = 0;
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kBinsPerThread; ++i) {
    const int bin = first_bin + i;
    if (coarse->histogram[bin] > k && coarse->histogram[bin + 1] <= k) {
      on_select(bin);
    }
  }
  __syncthreads();
}

// ============================================================================
// Medium path: exact 11-bit FP32 histogram, then the two remaining digits of
// the exact radix selection and an index-ordered emit.
// For sequences 8K < seq_len <= 32K.
//
// The buffered refinement this replaces claimed output slots and resolved
// pivot ties first-come-first-served, so the output order and the tie set
// depended on warp scheduling (#50576).
// ============================================================================

template <int TopK, uint32_t VecSize, typename InT>
__device__ __noinline__ void histogram_medium_topk(
    const InT* __restrict__ logits, int* __restrict__ output_indices,
    int seq_len) {
  // All shared state lives in dynamic shared memory to avoid static
  extern __shared__ char medium_smem[];

  struct CoarseSmem {
    CoarseSuffixSmem<kThreadsPerBlock, kCoarseBins> suffix;
    int threshold;
  };
  static_assert(sizeof(CoarseSmem) <= kSmemMedium);
  auto* coarse = reinterpret_cast<CoarseSmem*>(medium_smem);
  int* coarse_histogram = coarse->suffix.histogram;
  const int thread_id = threadIdx.x;

  for (int bin = thread_id; bin < kCoarseBins + 1; bin += kThreadsPerBlock) {
    coarse_histogram[bin] = 0;
  }
  __syncthreads();

  topk_histogram_4096::scan_scores<kThreadsPerBlock, VecSize>(
      logits, seq_len, [&](uint32_t, float score) {
        atomicAdd(&coarse_histogram[convert_to_uint32_v2(score) >> 21], 1);
      });
  __syncthreads();

  select_coarse_bin(&coarse->suffix, TopK,
                    [&](int bin) { coarse->threshold = bin; });

  const int threshold_bin = coarse->threshold;
  const int count_above = coarse_histogram[threshold_bin + 1];
  overflow::recover_medium<TopK, kThreadsPerBlock, VecSize>(
      logits, output_indices, seq_len, medium_smem,
      static_cast<uint32_t>(threshold_bin) << 21,
      static_cast<uint32_t>(TopK - count_above));
}

// ============================================================================
// Inter-CTA sync primitives
// ============================================================================

__device__ __forceinline__ int ld_acquire(int* ptr) {
  int state = 0;
#if defined(USE_ROCM)
  state = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#elif (__CUDA_ARCH__ >= 700)
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(state)
               : "l"(ptr));
#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif
  return state;
}

__device__ __forceinline__ void red_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
               :
               : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicAdd(ptr, val);
#endif
}

__device__ __forceinline__ void st_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicExch(ptr, val);
#endif
}

__device__ __forceinline__ void wait_ge(int* ptr, int target_val,
                                        int thread_idx) {
  if (thread_idx == 0) {
#pragma unroll 1
    while (ld_acquire(ptr) < target_val) {
    }
  }
  __syncthreads();
}

// ============================================================================
// Large path: multi-CTA radix select for sequences > 32K
//
// Each row is processed by a group of CTAs. Each CTA loads its chunk into
// shared memory as ordered uint32, then participates in 4 rounds of
// coordinated radix select via global-memory histograms and barriers.
// ============================================================================

// ============================================================================
// Multi-CTA cooperative RadixTopK for a single large row.
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

template <int TopK, uint32_t VEC_SIZE, typename InT>
__device__ void radix_topk(const InT* __restrict__ row_input,
                           int32_t* __restrict__ row_output, uint32_t seq_len,
                           uint32_t my_chunk_start, uint32_t chunk_size,
                           uint32_t* local_histogram, uint32_t* suffix_sum,
                           uint32_t* shared_scalars, uint32_t* shared_ordered,
                           RadixRowState* state, uint32_t cta_in_group,
                           uint32_t ctas_per_group, int& barrier_phase,
                           uint32_t radix_iter, uint32_t tx) {
  const uint32_t my_chunk_end = (my_chunk_start + chunk_size < seq_len)
                                    ? my_chunk_start + chunk_size
                                    : seq_len;
  const uint32_t actual_chunk_size =
      (my_chunk_start < seq_len) ? (my_chunk_end - my_chunk_start) : 0;

  // -- Stage 1: Load chunk to shared memory as ordered uint32 --
  {
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

    for (uint32_t i = tx * VEC_SIZE; i < aligned_size;
         i += kThreadsPerBlock * VEC_SIZE) {
      const InT* src = row_input + my_chunk_start + i;
      if constexpr (VEC_SIZE == 4) {
        float4 v = topk_histogram_4096::load_score4(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
        shared_ordered[i + 2] = convert_to_uint32_v2(v.z);
        shared_ordered[i + 3] = convert_to_uint32_v2(v.w);
      } else if constexpr (VEC_SIZE == 2) {
        float2 v = topk_histogram_4096::load_score2(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
      } else {
        shared_ordered[i] =
            convert_to_uint32_v2(topk_histogram_4096::to_float(*src));
      }
    }
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size;
         i += kThreadsPerBlock) {
      shared_ordered[i] = convert_to_uint32_v2(
          topk_histogram_4096::to_float(row_input[my_chunk_start + i]));
    }
  }
  __syncthreads();

  // -- Init radix select state --
  if (tx == 0) {
    shared_scalars[0] = 0;     // prefix
    shared_scalars[1] = TopK;  // remaining_k
  }
  __syncthreads();

  // Round 0's working histogram must be zero. The triple-buffer chain only
  // guarantees that when the previous group iteration was also a large row
  // (its round 2 zeroed this buffer); a short-row iteration in between runs
  // no rounds, leaving stale counts that corrupt the round-0 threshold (and
  // with it the pivot, the selected set, and the output bounds). Zero it
  // explicitly behind the initial barrier — no extra synchronization needed.
  if (cta_in_group == 0) {
    uint32_t* round0_hist = state->histogram[(radix_iter * 4) % 3];
    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      round0_hist[i] = 0;
    }
    __syncthreads();
  }

  // -- Initial barrier --
  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  if (cta_in_group == 0 && tx == 0) {
    st_release(&state->output_counter, 0);
  }

  // -- Stage 2: 4 rounds of radix select --
  for (uint32_t round = 0; round < 4; round++) {
    const uint32_t global_round = radix_iter * 4 + round;
    const uint32_t shift = 24 - round * 8;
    const uint32_t prefix = shared_scalars[0];
    const uint32_t remaining_k = shared_scalars[1];

    uint32_t* current_hist = state->histogram[global_round % 3];
    uint32_t* next_hist = state->histogram[(global_round + 1) % 3];

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      local_histogram[i] = 0;
    }
    __syncthreads();

    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      uint32_t ordered = shared_ordered[i];
      uint32_t mask = (round == 0) ? 0u : (~0u << (32 - round * 8));
      if ((ordered & mask) == prefix) {
        uint32_t bucket = (ordered >> shift) & 0xFF;
        atomicAdd(&local_histogram[bucket], 1);
      }
    }
    __syncthreads();

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      if (local_histogram[i] > 0) {
        atomicAdd(&current_hist[i], local_histogram[i]);
      }
    }

    if (cta_in_group == 0) {
      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        next_hist[i] = 0;
      }
    }
    __syncthreads();

    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    wait_ge(&state->arrival_counter,
            (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
    barrier_phase++;
    __syncthreads();

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      suffix_sum[i] = current_hist[i];
    }
    __syncthreads();

    for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
      uint32_t val = 0;
      if (tx < RADIX) {
        val = suffix_sum[tx];
        if (tx + stride < RADIX) val += suffix_sum[tx + stride];
      }
      __syncthreads();
      if (tx < RADIX) suffix_sum[tx] = val;
      __syncthreads();
    }

    if (tx == 0) {
      shared_scalars[2] = 0;
      shared_scalars[3] = remaining_k;
    }
    __syncthreads();

    if (tx < RADIX) {
      uint32_t count_ge = suffix_sum[tx];
      uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
      if (count_ge >= remaining_k && count_gt < remaining_k) {
        shared_scalars[2] = tx;
        shared_scalars[3] = remaining_k - count_gt;
      }
    }
    __syncthreads();

    if (tx == 0) {
      shared_scalars[0] = prefix | (shared_scalars[2] << shift);
      shared_scalars[1] = shared_scalars[3];
    }
    __syncthreads();
  }  // end 4 radix rounds

  // -- Stage 3: deterministic collection (see #50576) --
  //
  // Claiming per-CTA output ranges on a global counter and resolving == pivot
  // ties across CTAs first-come-first-served made both the output order and
  // the tie set depend on CTA/warp scheduling. Instead each CTA publishes its
  // (gt, eq) counts, a barrier makes them visible, then every CTA derives its
  // exclusive bases from lower-numbered CTAs (chunks are contiguous index
  // ranges, so CTA order == index order) and writes via index-ordered block
  // prefix scans. Output: all > pivot indices in ascending order, then the
  // first (TopK - total_gt) == pivot indices in ascending order.
  const uint32_t ordered_pivot = shared_scalars[0];
  const uint32_t tie_slots = shared_scalars[1];

  if (tx == 0) {
    suffix_sum[0] = 0;
    suffix_sum[1] = 0;
  }
  __syncthreads();
  {
    uint32_t my_gt = 0, my_eq = 0;
    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      const uint32_t v = shared_ordered[i];
      my_gt += (v > ordered_pivot);
      my_eq += (v == ordered_pivot);
    }
    my_gt = topk_histogram_4096::warp_reduce_sum_full(my_gt);
    my_eq = topk_histogram_4096::warp_reduce_sum_full(my_eq);
    if (tx % 32 == 0) {
      if (my_gt > 0) atomicAdd(&suffix_sum[0], my_gt);
      if (my_eq > 0) atomicAdd(&suffix_sum[1], my_eq);
    }
  }
  __syncthreads();

  // Publish packed counts. chunk_size is bounded by shared memory
  // (< 64K elements), so 16 bits per component suffice. The counts buffer
  // reuses the histogram slot that is not referenced again until it is
  // zeroed (behind a barrier) in the next iteration's round 0/1.
  uint32_t* counts_buf = state->histogram[(radix_iter * 4 + 5) % 3];
  if (tx == 0) {
    counts_buf[cta_in_group] = (suffix_sum[0] << 16) | suffix_sum[1];
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  uint32_t my_gt_base = 0, my_eq_base = 0, total_gt = 0;
  for (uint32_t c = 0; c < ctas_per_group; ++c) {
    const uint32_t packed = counts_buf[c];
    const uint32_t gt_c = packed >> 16;
    if (c < cta_in_group) {
      my_gt_base += gt_c;
      my_eq_base += packed & 0xFFFF;
    }
    total_gt += gt_c;
  }

  // Index-ordered selection sweep over this CTA's chunk.
  static_assert(RADIX >= 2 * topk_histogram_4096::kNumWarps);
  uint32_t def_run = 0, tie_run = 0;
  for (uint32_t chunk = 0, sweep = 0; chunk < actual_chunk_size;
       chunk += kThreadsPerBlock, ++sweep) {
    const uint32_t i = chunk + tx;
    bool is_def = false, is_tie = false;
    if (i < actual_chunk_size) {
      const uint32_t v = shared_ordered[i];
      is_def = (v > ordered_pivot);
      is_tie = (v == ordered_pivot);
    }

    const uint32_t packed = (is_def ? 0x10000u : 0u) | (is_tie ? 1u : 0u);
    const uint2 scan = topk_histogram_4096::block_packed_prefix(
        packed,
        local_histogram + (sweep & 1) * topk_histogram_4096::kNumWarps);

    if (is_def) {
      const uint32_t pos = my_gt_base + def_run + (scan.x >> 16);
      if (pos < static_cast<uint32_t>(TopK)) {
        row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
      }
    } else if (is_tie) {
      const uint32_t rank = my_eq_base + tie_run + (scan.x & 0xFFFF);
      if (rank < tie_slots && total_gt + rank < static_cast<uint32_t>(TopK)) {
        row_output[total_gt + rank] = static_cast<int32_t>(my_chunk_start + i);
      }
    }
    def_run += scan.y >> 16;
    tie_run += scan.y & 0xFFFF;
  }
  // The CTA's next row reuses this smem; finish the sweep before it does.
  __syncthreads();
}

// ============================================================================
// Persistent kernel — short/medium/large paths with RadixTopK.
// ============================================================================

template <int TopK = 2048, uint32_t VEC_SIZE = 1, typename InT = float>
__global__ void __launch_bounds__(kThreadsPerBlock, 2)
    persistent_topk_kernel(PersistentTopKParams<InT> params) {
  const uint32_t tx = threadIdx.x;
  extern __shared__ uint8_t smem_raw[];

  // ========================================================================
  // Group mode: multi-CTA groups with static round-robin row assignment.
  // Non-large rows: CTA-0 handles the trivial, short, and medium paths.
  // Large rows: all CTAs in the group cooperate via RadixTopK.
  // ========================================================================
  const uint32_t ctas_per_group = params.ctas_per_group;
  const uint32_t group_id = blockIdx.x / ctas_per_group;
  const uint32_t cta_in_group = blockIdx.x % ctas_per_group;
  const uint32_t num_groups = gridDim.x / ctas_per_group;
  const uint32_t chunk_size = params.chunk_size;

  if (blockIdx.x >= num_groups * ctas_per_group) return;

  // Early exit: non-CTA-0 threads are never needed if no large rows exist
  if (cta_in_group != 0 && params.max_seq_len <= RADIX_THRESHOLD) return;

  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  uint32_t* shared_ordered =
      reinterpret_cast<uint32_t*>(smem_raw + kFixedSmemLarge);

  // RadixRowState for multi-CTA cooperative radix.
  // Zero-initialization is done host-side via cudaMemsetAsync in topk.cu
  // before launch — that gives a stream-ordered happens-before edge for all
  // CTAs, which the previous in-kernel init (CTA-0 only + intra-CTA
  // __syncthreads) did not provide and which manifested as a race against
  // CTA-1+'s first red_release on arrival_counter.
  RadixRowState* state = &params.row_states[group_id];

  int barrier_phase = 0;
  uint32_t radix_iter = 0;
  const uint32_t total_iters = (params.num_rows + num_groups - 1) / num_groups;

  for (uint32_t iter = 0; iter < total_iters; iter++) {
    // Static round-robin: all CTAs in the group implicitly agree on the row
    uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= params.num_rows) break;

    // Clamp the row length before any decision is made on it.
    //
    // `lengths` is int32 and is consumed here as uint32, so a negative value
    // (e.g. a padded decode slot whose per-token context length underflowed)
    // would reinterpret as ~4e9 and sail past every threshold below. Any
    // value beyond the row width would also read into the next row.
    //
    // Clamping to max_seq_len additionally keeps this per-row decision
    // consistent with the `cta_in_group != 0` early exit above, which is
    // taken from the host-side scalar: when max_seq_len <= RADIX_THRESHOLD
    // the non-leader CTAs return immediately, so a leader that reached the
    // cooperative radix path would wait on the inter-CTA barrier for peers
    // that no longer exist and spin until the kernel is killed.
    const int32_t raw_len = params.lengths[row_idx];
    const uint32_t row_bound =
        params.stride < params.max_seq_len ? params.stride : params.max_seq_len;
    const uint32_t non_negative_len =
        raw_len > 0 ? static_cast<uint32_t>(raw_len) : 0u;
    const uint32_t seq_len =
        non_negative_len < row_bound ? non_negative_len : row_bound;
    int32_t* row_output = params.output + row_idx * params.top_k;
    const InT* row_input = params.input + row_idx * params.stride;

    if (seq_len <= RADIX_THRESHOLD) {
      if (cta_in_group == 0) {
        if (seq_len <= static_cast<uint32_t>(TopK)) {
          // Trivial case: seq_len <= TopK
          for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
               i += kThreadsPerBlock) {
            row_output[i] = (i < seq_len) ? static_cast<int32_t>(i) : -1;
          }
        } else if (seq_len <= static_cast<uint32_t>(SHORT_EXACT_THRESHOLD)) {
          topk_histogram_4096::exact_topk_rescan_cold<TopK, kThreadsPerBlock,
                                                      VEC_SIZE>(
              row_input, row_output, seq_len, smem_raw);
        } else {
          histogram_medium_topk<TopK, VEC_SIZE>(row_input, row_output,
                                                seq_len);
        }
      }
      continue;
    }

    const uint32_t my_chunk_start = cta_in_group * chunk_size;
    radix_topk<TopK, VEC_SIZE>(
        row_input, row_output, seq_len, my_chunk_start, chunk_size,
        local_histogram, suffix_sum, shared_scalars, shared_ordered, state,
        cta_in_group, ctas_per_group, barrier_phase, radix_iter, tx);
    radix_iter++;
  }
}

}  // namespace persistent

// ============================================================================
// FilteredTopK — one CTA per row, for the non-cluster path.
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================
namespace filtered_topk {

namespace hist4096 = topk_histogram_4096;

#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename T, size_t N>
struct vec_t {
  T data[N];

  FLASHINFER_INLINE T& operator[](size_t i) { return data[i]; }
  FLASHINFER_INLINE const T& operator[](size_t i) const { return data[i]; }

  template <typename InT>
  FLASHINFER_INLINE void cast_load(const InT* ptr) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      data[i] = topk_histogram_4096::to_float(ptr[i]);
    }
  }
};
#undef FLASHINFER_INLINE

// FilteredTopK traits for different data types
template <typename DType, bool UseWideCoarse = false>
struct FilteredTopKTraits;

// Both float paths use the exact ordered-FP32 key; only the coarse-bin width
// and refinement digit layout differ.
template <bool UseWideCoarse>
struct FilteredTopKTraits<float, UseWideCoarse> {
  using OrderedType = uint32_t;
  static constexpr int COARSE_BITS = UseWideCoarse ? 12 : 11;
  static constexpr int COARSE_RADIX = 1 << COARSE_BITS;
  static constexpr int NUM_REFINE_ROUNDS = UseWideCoarse ? 2 : 3;
  static constexpr int FIRST_REFINE_SHIFT = UseWideCoarse ? 10 : 16;
  static constexpr int REFINE_BITS = UseWideCoarse ? 10 : 8;

  __device__ __forceinline__ static uint16_t ToCoarseKey(float x) {
    return static_cast<uint16_t>(ToOrdered(x) >> (32 - COARSE_BITS));
  }

  __device__ __forceinline__ static OrderedType ToOrdered(float x) {
    return topk_histogram_4096::convert_to_uint32_v2(x);
  }
};

constexpr uint32_t FILTERED_TOPK_BLOCK_THREADS = 1024;
constexpr uint32_t FILTERED_TOPK_SMEM_INPUT_SIZE =
    16 * 1024;  // 16K indices per buffer
constexpr size_t FILTERED_TOPK_SMEM_DYNAMIC =
    sizeof(int) * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE;  // 128KB
constexpr uint32_t FILTERED_TOPK_WIDE_COARSE_MIN_LENGTH = 64 * 1024;
constexpr uint32_t FILTERED_TOPK_WIDE_COARSE_MAX_LENGTH = 192 * 1024;
static_assert(hist4096::kExactCandidateOffset +
                  FILTERED_TOPK_SMEM_INPUT_SIZE * sizeof(int) <=
              FILTERED_TOPK_SMEM_DYNAMIC);

template <uint32_t MAX_K, bool UseWideCoarse = false>
struct FilteredTopKStorage {
  static constexpr int RADIX = UseWideCoarse ? 1024 : 256;
  alignas(128) int histogram[2][RADIX + 128];
  alignas(128) int counter;
  alignas(128) int threshold_bin;
  alignas(128) int num_input[2];
  alignas(128) int indices[MAX_K];
  int last_remain;
};

// With CheckOverflow, return false instead of recovering when the candidate
// stash overflows, so the caller can finish with its own exact selection.
template <typename DType, typename IdType, int VEC_SIZE, uint32_t MAX_K,
          bool UsePredicatedShortLoads, bool CheckOverflow = false,
          bool UseWideCoarse = false>
__device__ __forceinline__ bool filtered_topk_row(
    const DType* score, IdType* dst, int length, uint32_t top_k,
    FilteredTopKStorage<MAX_K, UseWideCoarse>& storage, uint32_t num_rows = 0) {
  extern __shared__ __align__(16) uint8_t dynamic_smem[];
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;
  using Traits = FilteredTopKTraits<float, UseWideCoarse>;
  constexpr int RADIX = 1 << Traits::REFINE_BITS;
  static_assert(RADIX == FilteredTopKStorage<MAX_K, UseWideCoarse>::RADIX);
  constexpr int COARSE_RADIX = Traits::COARSE_RADIX;
  using CoarseSmem = persistent::CoarseSuffixSmem<BLOCK_SIZE, COARSE_RADIX>;
  static_assert(sizeof(CoarseSmem) <= FILTERED_TOPK_SMEM_DYNAMIC);

  const int tx = threadIdx.x;

  // Trivial case: length <= top_k
  if (length <= static_cast<int>(top_k)) {
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
    }
    return true;
  }

  // Short path
  if (length <= 32768) {
    if constexpr (UsePredicatedShortLoads) {
      hist4096::histogram_4096_topk_predicated<
          MAX_K, 12, 8, hist4096::OverflowRecovery::kRegisterValues>(
          score, dst, length, dynamic_smem);
    } else {
      hist4096::histogram_4096_topk<
          MAX_K, 12, 8, false, hist4096::OverflowRecovery::kRegisterValues>(
          score, dst, length, dynamic_smem);
    }
    return true;
  }

  auto& s_threshold_bin_id = storage.threshold_bin;
  auto& s_histogram = storage.histogram[0];

  auto* coarse_smem = reinterpret_cast<CoarseSmem*>(dynamic_smem);
  auto* s_coarse_histogram = coarse_smem->histogram;

  // Stage 1: high bits of the exact ordered-FP32 key.
  for (int bin = tx; bin < COARSE_RADIX + 1; bin += BLOCK_SIZE) {
    s_coarse_histogram[bin] = 0;
  }
  __syncthreads();

  vec_t<float, VEC_SIZE> score_vec;

  const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
  for (int base = tx * VEC_SIZE; base < aligned_length;
       base += BLOCK_SIZE * VEC_SIZE) {
    score_vec.cast_load(&score[base]);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const auto bin = Traits::ToCoarseKey(score_vec[j]);
      atomicAdd(&s_coarse_histogram[bin], 1);
    }
  }
  // Handle tail
  for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
    const auto bin = Traits::ToCoarseKey(hist4096::to_float(score[i]));
    atomicAdd(&s_coarse_histogram[bin], 1);
  }
  __syncthreads();

  persistent::select_coarse_bin(coarse_smem, static_cast<int>(top_k),
                                [&](int bin) { s_threshold_bin_id = bin; });

  // Stage 2: finish the exact radix selection below the coarse digit and emit
  // in index order (#50576). The candidate stash this replaces claimed output
  // slots and resolved pivot ties first-come-first-served, so both the order
  // and the tie set depended on warp scheduling. Nothing is buffered, so the
  // row cannot overflow and CheckOverflow callers never need to fall back.
  overflow::recover_coarse<DType, IdType, VEC_SIZE, BLOCK_SIZE, MAX_K,
                           SMEM_INPUT_SIZE, Traits::COARSE_BITS>(
      score, dst, length, top_k, num_rows, s_coarse_histogram,
      s_threshold_bin_id, dynamic_smem, s_histogram);
  return true;
}

/*!
 * \brief Filtered Top-K kernel for ragged sequences.
 *
 * \tparam DType Data type (float, half, nv_bfloat16)
 * \tparam IdType Index type (int32_t)
 * \tparam VEC_SIZE Vector size for input loads (1, 2, 4, or 8)
 */
template <typename DType, typename IdType, int VEC_SIZE, uint32_t MAX_K = 2048,
          bool UsePredicatedShortLoads = false, bool UseWideCoarse = false>
__global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
    FilteredTopKUnifiedKernel(const DType* __restrict__ input,
                              IdType* __restrict__ output,
                              const IdType* __restrict__ lengths,
                              uint32_t num_rows, uint32_t top_k,
                              uint32_t max_len) {
  const uint32_t bid = blockIdx.x;
  if (bid >= num_rows) return;
  const int length = lengths ? lengths[bid] : static_cast<int>(max_len);
  __shared__ FilteredTopKStorage<MAX_K, UseWideCoarse> storage;
  filtered_topk_row<DType, IdType, VEC_SIZE, MAX_K, UsePredicatedShortLoads,
                    false, UseWideCoarse>(input + bid * max_len,
                                          output + bid * top_k, length, top_k,
                                          storage, num_rows);
}

// Helper to compute GCD for VEC_SIZE selection
constexpr uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Compute optimal VEC_SIZE based on max_len and dtype
// Returns 1, 2, 4, or 8
template <typename DType>
constexpr int ComputeFilteredTopKVecSize(uint32_t max_len) {
  constexpr int MAX_VEC = 4;
  // Use GCD to find largest power-of-2 divisor
  const uint32_t g = gcd(max_len, static_cast<uint32_t>(MAX_VEC));
  return static_cast<int>(g);
}

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(const DType* input,
                                        IdType* output_indices,
                                        const IdType* lengths,
                                        uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len,
                                        cudaStream_t stream = 0) {
  constexpr size_t smem_size = FILTERED_TOPK_SMEM_DYNAMIC;
  constexpr int MAX_VEC = 4;

  dim3 grid(num_rows);
  dim3 block(FILTERED_TOPK_BLOCK_THREADS);
  void* args[] = {&input,    &output_indices, &lengths,
                  &num_rows, &top_k_val,      &max_len};

  // Views like x[:, 1:] keep the stride but shift the base.
  int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);
  while (vec_size > 1 &&
         reinterpret_cast<uintptr_t>(input) % (vec_size * sizeof(DType)) != 0) {
    vec_size /= 2;
  }
  const bool use_wide_coarse =
      MAX_K == 1024 && max_len >= FILTERED_TOPK_WIDE_COARSE_MIN_LENGTH &&
      max_len < FILTERED_TOPK_WIDE_COARSE_MAX_LENGTH;

#define DISPATCH_VEC_SIZE(VS)                                               \
  if (vec_size == VS) {                                                     \
    auto kernel = FilteredTopKUnifiedKernel<DType, IdType, VS, MAX_K,       \
                                            (VS != MAX_VEC), false>;        \
    if constexpr (MAX_K == 1024) {                                          \
      if (use_wide_coarse) {                                                \
        kernel = FilteredTopKUnifiedKernel<DType, IdType, VS, MAX_K,        \
                                           (VS != MAX_VEC), true>;          \
      }                                                                     \
    }                                                                       \
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                              \
        reinterpret_cast<const void*>(kernel),                              \
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));           \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, \
                                          smem_size, stream));              \
    return cudaSuccess;                                                     \
  }

  DISPATCH_VEC_SIZE(1)
  DISPATCH_VEC_SIZE(2)
  DISPATCH_VEC_SIZE(4)
  if constexpr (MAX_VEC >= 8) {
    DISPATCH_VEC_SIZE(8)
  }
#undef DISPATCH_VEC_SIZE

  return cudaSuccess;
}

}  // namespace filtered_topk

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(const DType* input,
                                        IdType* output_indices,
                                        const IdType* lengths,
                                        uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len,
                                        cudaStream_t stream = 0) {
  return filtered_topk::FilteredTopKRaggedTransform<DType, IdType, MAX_K>(
      input, output_indices, lengths, num_rows, top_k_val, max_len, stream);
}

}  // namespace vllm

#endif  // PERSISTENT_TOPK_CUH_
