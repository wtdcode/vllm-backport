#pragma once

#include "custom_collective_common.cuh"

namespace vllm {

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

constexpr int kQBlock = 32;

struct __align__(16) QBlock {
  int4 lo, hi;
};

template <int ngpus, bool ABSOLUTE = false>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage_int8(RankData* _dp, RankSignals sg,
                                    Signal* self_sg,
                                    int8_t* __restrict__ result,
                                    nv_bfloat16* __restrict__ result_s,
                                    int rank, int nblk, int64_t scale_off,
                                    int64_t tmp_scale_off) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int part = nblk / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? nblk : start + part;
  int largest_part = part + nblk % ngpus;

  const QBlock* qptr[ngpus];
  const nv_bfloat16* sptr[ngpus];
  QBlock* tmp_q[ngpus];
  nv_bfloat16* tmp_s[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = ABSOLUTE ? i : (rank + i) % ngpus;
    auto base = (const char*)_dp->ptrs[target];
    qptr[i] = (const QBlock*)base;
    sptr[i] = (const nv_bfloat16*)(base + scale_off);
    auto t = (char*)get_tmp_buf<QBlock>(sg.signals[target]);
    tmp_q[i] = (QBlock*)t;
    tmp_s[i] = (nv_bfloat16*)(t + tmp_scale_off);
  }
  auto self_t = (char*)get_tmp_buf<QBlock>(sg.signals[rank]);
  auto self_q = (QBlock*)self_t;
  auto self_s = (nv_bfloat16*)(self_t + tmp_scale_off);
  barrier_at_start<ngpus>(sg, self_sg, rank);

  for (int idx = start + tid; idx < end; idx += stride) {
    float acc[kQBlock];
#pragma unroll
    for (int j = 0; j < kQBlock; j++) acc[j] = 0.0f;
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      QBlock v = qptr[i][idx];
      float sc = __bfloat162float(sptr[i][idx]);
      auto b = reinterpret_cast<const int8_t*>(&v);
#pragma unroll
      for (int j = 0; j < kQBlock; j++) acc[j] += float(b[j]) * sc;
    }
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < kQBlock; j++) amax = fmaxf(amax, fabsf(acc[j]));
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    QBlock o;
    auto ob = reinterpret_cast<int8_t*>(&o);
#pragma unroll
    for (int j = 0; j < kQBlock; j++) {
      int q = __float2int_rn(acc[j] * inv);
      ob[j] = (int8_t)min(max(q, -127), 127);
    }
    self_q[idx - start] = o;
    self_s[idx - start] = __float2bfloat16(amax / 127.0f);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ABSOLUTE ? i : ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((QBlock*)result)[dst_idx] = tmp_q[i][idx];
        result_s[dst_idx] = tmp_s[i][idx];
      }
    }
  }
}

static __global__ void __launch_bounds__(256)
    quantize_blockwise_int8(const nv_bfloat16* __restrict__ inp,
                            int8_t* __restrict__ out, int nblk,
                            int64_t scale_off) {
  auto out_s = (nv_bfloat16*)(out + scale_off);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nblk;
       idx += gridDim.x * blockDim.x) {
    const int4* src = (const int4*)(inp + (int64_t)idx * kQBlock);
    int4 v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
    nv_bfloat16 raw[kQBlock];
    reinterpret_cast<int4*>(raw)[0] = v0;
    reinterpret_cast<int4*>(raw)[1] = v1;
    reinterpret_cast<int4*>(raw)[2] = v2;
    reinterpret_cast<int4*>(raw)[3] = v3;

    float x[kQBlock], amax = 0.0f;
#pragma unroll
    for (int j = 0; j < kQBlock; j++) {
      x[j] = __bfloat162float(raw[j]);
      amax = fmaxf(amax, fabsf(x[j]));
    }
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    QBlock o;
    auto ob = reinterpret_cast<int8_t*>(&o);
#pragma unroll
    for (int j = 0; j < kQBlock; j++) {
      int q = __float2int_rn(x[j] * inv);
      ob[j] = (int8_t)min(max(q, -127), 127);
    }
    ((QBlock*)out)[idx] = o;
    out_s[idx] = __float2bfloat16(amax / 127.0f);
  }
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  // Full NVLink or xGMI connection between GPUs.
  bool fully_connected_;

  RankSignals sg_;
  // Stores a map from a pointer to its peer pointers from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph
  // capture time. Therefore, during capture, we increment the rank data
  // pointer and use that as the argument to the kernel. The kernel arguments
  // are stored in graph_unreg_buffers_. The actual peer pointers will be
  // filled in at the memory pointed to by the pointers in
  // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second
   * section is for storing the intermediate results required by some
   * allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool fully_connected = true)
      : rank_(rank),
        world_size_(world_size),
        fully_connected_(fully_connected),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr, rangeStartAddrAttr,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the
  // remote possibility of different allocation patterns between ranks. For
  // example, rank 1 may get the same input address for the second allreduce,
  // but rank 2 got a different address. IPC handles have internal reference
  // counting mechanism so overhead should be small.
  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search.
   * Using 36 blocks give the best or close to the best runtime on the devices
   * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
   * only take a small amount of SMs. Not quite sure the underlying reason,
   * but my guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size,
                 int threads = 512, int block_limit = defaultBlockLimit) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    // Check environment variable once
    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }

#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
    // One-shot has every rank read the whole payload from all N-1 peers;
    // two-shot reduce-scatters then all-gathers, moving ~4x less over the links
    // but paying an extra barrier round. Below this size the barrier dominates
    // and one-shot wins. Measured on 8xA100 NVLink (hidden 4096, bf16,
    // cudagraph-captured, us/call, in-tree communicator bench / an independent
    // probe):
    //
    //     128 KB  1stage 12 / 22.8   2stage 19 / 24.2
    //     256 KB  1stage 16 / 26.2   2stage 23 / 28.2
    //     384 KB  1stage 20 / 28.1   2stage 23 / 29.1
    //     512 KB  1stage 24 / 30.1   2stage 23 / 29.1   <- crossover
    //       1 MB  1stage 40 / 48.3   2stage 24 / 33.8
    //
    // The previous 256 KB put [256 KB, 512 KB) on two-shot while one-shot was
    // still faster there. This threshold was measured at world_size 8;
    // world_size 4 already used 512 KB. NOTE the value is applied to world_size
    // 6 WITHOUT a measurement -- it sits between a shipped 512 (ws 4) and a
    // measured 512 (ws 8), and one-shot's cost grows with N (N-1 peer reads),
    // so a monotone crossover puts ws 6 at or above ws 8's 512.
    constexpr size_t kOneShotMaxBytes = 512 * 1024;

#define REDUCE_CASE(ngpus)                       \
  case ngpus: {                                  \
    if (force_1stage) {                          \
      KL(ngpus, cross_device_reduce_1stage);     \
    } else if (force_2stage) {                   \
      KL(ngpus, cross_device_reduce_2stage);     \
    } else {                                     \
      if (world_size_ == 2) {                    \
        KL(ngpus, cross_device_reduce_1stage);   \
      } else if (fully_connected_) {             \
        if (bytes < kOneShotMaxBytes) {          \
          KL(ngpus, cross_device_reduce_1stage); \
        } else {                                 \
          KL(ngpus, cross_device_reduce_2stage); \
        }                                        \
      }                                          \
    }                                            \
    break;                                       \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual "
            "num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  void allreduce_int8(cudaStream_t stream, int8_t* input, int8_t* output,
                      nv_bfloat16* output_scales, int nblk, int64_t scale_off,
                      int64_t tmp_scale_off, int threads = 512,
                      int block_limit = defaultBlockLimit) {
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));
    if (nblk % world_size_ != 0)
      throw std::runtime_error(
          "int8 custom allreduce requires the block count to be divisible by "
          "the world size; got " +
          std::to_string(nblk) + " blocks on " + std::to_string(world_size_) +
          " ranks");

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }
    int blocks = std::min(block_limit, (nblk + threads - 1) / threads);

    const char* abs_env = std::getenv("VLLM_AR_INT8_ABSOLUTE_ORDER");
    bool absolute = abs_env != nullptr && std::strcmp(abs_env, "1") == 0;

#define KL_INT8(ngpus)                                                \
  if (absolute)                                                       \
    cross_device_reduce_2stage_int8<ngpus, true>                      \
        <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                         output_scales, rank_, nblk,  \
                                         scale_off, tmp_scale_off);   \
  else                                                                \
    cross_device_reduce_2stage_int8<ngpus, false>                     \
        <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                         output_scales, rank_, nblk,  \
                                         scale_off, tmp_scale_off);

    switch (world_size_) {
      case 2:
        KL_INT8(2);
        break;
      case 4:
        KL_INT8(4);
        break;
      case 6:
        KL_INT8(6);
        break;
      case 8:
        KL_INT8(8);
        break;
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef KL_INT8
  }

  void allgather(cudaStream_t stream, void* input, void* output, int size_bytes,
                 int threads = 512, int block_limit = defaultBlockLimit);
  template <typename T>
  void mnnvl_lamport_allgather(cudaStream_t stream, T* input, T* output,
                               void* local_buffer, void* multicast_buffer,
                               uint32_t* epochs, int size_bytes,
                               int stage_size_bytes);
  template <typename T>
  void reduce_scatter(cudaStream_t stream, T* input, T* output, int size,
                      int threads = 512, int block_limit = defaultBlockLimit);
  template <typename T>
  void mnnvl_lamport_reduce_scatter(cudaStream_t stream, T* input, T* output,
                                    void* local_buffer, uint32_t* epochs,
                                    int size, int stage_size_bytes);

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};

/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and
 * add a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 *                                                       half *, int, int, int);
 */
}  // namespace vllm
