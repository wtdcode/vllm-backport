# VLLM Backport

A VLLM fork that focuses on running Deepseek V4 Flash 0731 on Ampere at this moment.

Currently achieving 3435 tps prefill and 948 tps decoding on 8xA6000 (TP4PP2)  and this should also work on A100.

## Docker Usage

Prebuilt images are published to Docker Hub on every push:

| Image | Target GPUs |
| --- | --- |
| `lazymio/vllm-backport:latest-sm86` (also `:latest`) | Ampere sm86 (A6000, RTX 30xx) |
| `lazymio/vllm-backport:latest-sm80` | Ampere sm80 (A100) |
| `lazymio/vllm-backport:latest-sm89` | Ada sm89 (RTX 4090, L40S) |
| `lazymio/vllm-backport:v0.6.4-sm86` / `-sm80` / `-sm89` | pinned release builds |

Images are single-arch builds (no FA3/Hopper kernels), so pick the tag matching your GPU. The entrypoint is `vllm serve`.

On sm89, set `VLLM_TEST_FORCE_FP8_MARLIN=1` in the container environment.

`:latest*` tags track the main branch; each release also ships versioned tags like `:v0.6.1-sm86` if you want to pin.

### Docker Compose

```yaml
services:
  vllm:
    image: lazymio/vllm-backport:latest-sm86  # A100: use :latest-sm80
    command: >
      deepseek-ai/DeepSeek-V4-Flash-0731
      --tensor-parallel-size 8
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}
    ipc: host
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Then:

```bash
docker compose up -d
curl http://localhost:8000/v1/models
```

Adjust the model and `--tensor-parallel-size` to your setup; `ipc: host` is required for multi-GPU tensor parallelism.

## Recommend Setup

```bash
vllm serve /path/to/your/deepseek \
  --tensor-parallel-size 8 \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8_ds_mla \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64],"max_cudagraph_capture_size":64}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --host 0.0.0.0 --port 8000 \
  --hf-overrides '{"head_dtype": "float32"}' \
  --served-model-name deepseek-v4-flash
```

with these environment variables (required for `FULL_AND_PIECEWISE`, see the cudagraph tip below):

```bash
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
```

Tips:

- `FULL_AND_PIECEWISE` (v0.6.x default recommendation, previously `PIECEWISE`) captures the whole decode step — attention, MoE dispatch, NCCL all-reduce and the DSpark draft loop — into one CUDA graph. Measured on 4x/8x A6000: single-stream decode 45.6 -> 67-70 tok/s (+47%); prefill is unchanged (compute-bound). Two prerequisites:
    - Pin NCCL with `NCCL_ALGO=Ring NCCL_PROTO=Simple` (and prefer `--disable-custom-all-reduce` unless enabling `VLLM_MHC_AR_INT8`, which requires custom all-reduce). Graph replay must re-issue the exact captured collective; NCCL's size-adaptive algorithm switching is what made FULL capture "crash on Ampere" — Ampere itself is fine.
    - Bound `cudagraph_capture_sizes` as shown. FULL graphs keep private memory pools; capturing every batch size up to `--max-num-seqs` can cost >800 MB per GPU and OOM warmup at high `--gpu-memory-utilization`.
- Adjust your TP (--tensor-parallel-size) and PP (--pipeline-parallel-size) accordingly.
- head dtype override helps reduce garbage outputs.
- DSpark is not working very well if you have PP>1.
- The single quotes around the `--speculative-config` JSON are required — without them bash brace-expands the braces at the comma and vLLM receives the literal `method:dspark` (`Value method:dspark cannot be converted` error).
- Keep `num_speculative_tokens` at 5 on Ampere. Values below 5 (the checkpoint's `dspark_block_size`) are rejected, and 7 needs ~200 KB of shared memory vs the 163 KB Ampere limit (`triton OutOfResources` error). 6 does start, but draft positions past the native block are almost never accepted (3–13% in our measurements), so it only wastes draft compute — output quality and speed are the same as 5.
- Requests that set neither `thinking` nor `reasoning_effort` now get thinking mode with high effort, matching the official 0731 API mapping (`reasoning_effort: "none"` restores plain chat mode). Agentic/tool-calling clients should pass a `reasoning_effort` explicitly from the first turn of a session — sessions that run without the effort prefix gradually stop thinking and can enter self-reinforcing reasoning loops.

## Environment Variables (Warning: Huge AI generated contents!)

All knobs this fork has added over stock vLLM. Defaults are what the images ship with; you normally don't need to touch anything.

### On by default (correctness / batch-invariance)

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_DETERMINISTIC_MOE_ALIGN` | `1` | Deterministic MoE token grouping (stable sort instead of atomic-order). `0` restores the historical CUDA kernel. |
| `VLLM_DSV4_FIXED_DECODE_SPLITS` | `16` | Pin the sparse-decode attention split-k so a request's numerics don't depend on what else is co-batched. `0` restores the batch-adaptive heuristic. |
| `VLLM_TOKEN_BUCKET_PAD` | `1` | Pad batches to fixed token buckets (16/32/64/128/256, then ×256) so GEMM tiling stops shifting with exact batch size. `0` disables. |
| `VLLM_DSPARK_FUSED_MARKOV` | `1` | Fused DSpark Markov draft-sampling chain. `0` falls back to the eager op chain. |
| `VLLM_DSV4_LOGITS_ROW_CHUNK` | `128` | Row-chunk the sparse-indexer prefill logits so the `[chunk_rows, context/4]` fp32 transient stays bounded at long context (fixes crashes beyond ~134k tokens; needed for 256k+). `0` restores the monolithic allocation; the unprefixed `DSV4_LOGITS_ROW_CHUNK` spelling also works. |

### Opt-in performance knobs (default `0` — measure on your topology first)

| Variable | Meaning |
| --- | --- |
| `VLLM_MHC_PRENORM_SHARD` | Shard the mHC prenorm GEMM across TP ranks (pays off at TP8, hurts at TP4). Requires `VLLM_MHC_POST_FUSE_SQRSUM=1`; startup fails if the shard is requested without it. |
| `VLLM_MHC_POST_FUSE_SQRSUM` | Fold the mHC prenorm row-sqrsum into `mhc_post`; required by `VLLM_MHC_PRENORM_SHARD`. |
| `VLLM_MHC_AR_INT8` | Transport prefill attention and MoE all-reduces as blockwise int8 and consume them directly in `mhc_post`. Requires custom all-reduce and a sufficiently large `VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE`; changes numerics. |
| `VLLM_UNREPLICATE_ATTN_GEMMS` | De-duplicate attention GEMMs that are replicated across TP ranks. |
| `VLLM_INDEXER_QUERY_SHARD` / `VLLM_INDEXER_QUERY_SHARD_QPATH` | Shard the sparse-indexer query projection across TP ranks. |
| `VLLM_SPARSE_RAGGED_FAST_SCAN` | Faster ragged-index scan in sparse prefill. |
| `VLLM_SPARSE_PREFILL_EXACT_TILE` | Mask-free sparse-prefill kernel specialization for exact-tile shapes. |
| `VLLM_DSPARK_VOCAB_SHARD` | Vocab-sharded DSpark greedy draft selection (less draft-side communication). |
| `VLLM_MARLIN_FP8_DEQUANT_BF16` | Route dense block-fp8 GEMMs through cuBLAS (dequant→bf16) instead of Marlin. |
| `VLLM_HIER_ALL_REDUCE` | Island-aware hierarchical all-reduce for boxes with multiple PCIe islands. |
| `VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE` | Override the custom all-reduce payload cap (MB). |
| `VLLM_MHC_FIXED_NUM_SPLIT` | Pin the mHC TileLang GEMM split-k (only reachable on DeepGEMM-capable GPUs; no effect on sm86/sm80). |

### Ops / debug

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | `16` | Worker message-queue chunk size. Lower it (e.g. `1`) when the container has a small `/dev/shm` and you cannot use `--ipc=host`. |
| `VLLM_DISABLE_MULTI_STREAM_PARALLEL` | `0` | Debug kill-switch: run aux-stream work serially on the default stream. |

For strict temperature-0 stability under concurrency, also consider `--hf-overrides '{"head_dtype": "float32"}'` (fp32 logits head) — a CLI flag, not an env.
