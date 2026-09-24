# vLLM Backport

A vLLM fork that focuses on running frontier models on older cards like A6000, 3090 and A100.

Status:

| Supported Models | Quantization | Status |
| --- | --- | --- |
| `DeepSeek-v4-Flash-0731` | Native FP4 | Fully Supported (0.6.0+) |
| `Qwen3.8-27B` | BF16, AWQ W4A16 | Fully Supported (v0.8.0+) |
| `Qwen3.8-Flash-Next` | FP8, [AWQ W4A16](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) | Fully Supported (v0.9.0+) |
| `GLM-5.3-Flash` | FP8, [AWQ W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16) | Fully Supported (v0.11.2+) |
| `DeepSeek-V4-Flash-Vision-Exp` | Native FP4 | Fully Supported (v0.13.0+) |
| `DeepSeek-V4.1-Flash` | Native FP4 with Engram | Fully Supported (v0.13.0+) |
| `MiMo-V2.6-Flash-RL` | Native MXFP4 (fp8 QKV/o_proj) | Fully Supported (main) |

Note we have a paired [LMCache](https://github.com/wtdcode/LMCache/tree/vllm-backport) fork for production kvcache serving, **which is also built into our docke images.**

## Docker Usage

Prebuilt images are published to Docker Hub on every push:

| Image | Target GPUs |
| --- | --- |
| `lazymio/vllm-backport:latest-sm86` (also `:latest`) | Ampere sm86 (A6000, RTX 30xx) |
| `lazymio/vllm-backport:latest-sm80` | Ampere sm80 (A100) |
| `lazymio/vllm-backport:latest-sm89` | Ada sm89 (RTX 4090, L40S) |
| `lazymio/vllm-backport:v0.13.0-sm86` / `-sm80` / `-sm89` | pinned release builds |

Images are single-arch builds (no FA3/Hopper kernels), so pick the tag matching your GPU. The entrypoint is `vllm serve` and `lmcache` is also available within the same image!

`:latest*` tags track the main branch; each release also ships versioned tags like `:v0.13.0-sm86` if you want to pin. Check [Dockerhub](https://hub.docker.com/r/lazymio/vllm-backport/tags) or [Github](https://github.com/wtdcode/vllm-backport/tags) for available latest tags.

### Docker Compose Sample

Note the sample below includes `LMCache`. If you do not have enough RAM or disk, you could remove the `--kv-transfer-config` and the full `LMCache` section.

```yaml
services:
  vllm:
    image: lazymio/vllm-backport:latest
    depends_on: [lmcache] # Remove this line if you remove the lmcache section below
    command:
      - deepseek-ai/DeepSeek-V4-Flash-0731
      - '--kv-transfer-config={"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5556}}'
      - ... (model specific commands see below)
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
  lmcache:
    image: lazymio/vllm-backport:latest
    entrypoint: ["lmcache", "server"]
    command:
      - --host=127.0.0.1
      - --port=5556
      - --http-port=18556
      - --chunk-size=1024 # must be a multiple of the model's KV block size, see each model's LMCache note below
      - --separate-object-groups
      - --l1-size-gb=1024
      - --eviction-policy=LRU
      - --max-workers 16
      - --l2-adapter '{"type":"fs_native","base_path":"path/to/l2","num_workers":32,"max_capacity_gb":1024,"eviction":{"eviction_policy":"LRU","trigger_watermark":0.8,"eviction_ratio":0.2}}'
    environment:
      - PYTHONUNBUFFERED=1
      - CUDA_DEVICE_ORDER=PCI_BUS_ID
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

## Recommend Setup

### General Tips

- `FULL_AND_PIECEWISE` captures the whole decode step — attention, MoE dispatch, NCCL all-reduce and the DSpark draft loop — into one CUDA graph. This improves decode but prefill is unchanged (compute-bound). Two prerequisites:
    - Pin NCCL with `NCCL_ALGO=Ring NCCL_PROTO=Simple` (and prefer `--disable-custom-all-reduce`). Graph replay must re-issue the exact captured collective; NCCL's size-adaptive algorithm switching is what made FULL capture "crash on Ampere" — Ampere itself is fine.
    - Bound `cudagraph_capture_sizes` as shown. FULL graphs keep private memory pools; capturing every batch size up to `--max-num-seqs` can cost >800 MB per GPU and OOM warmup at high `--gpu-memory-utilization`.
- Adjust your TP (--tensor-parallel-size), PP (--pipeline-parallel-size) and EP (--enable-expert-parallel) accordingly.

### Model Specific setups

#### GLM-5.3-Flash (v0.11.2+)

```bash
NCCL_ALGO=Ring NCCL_PROTO=Simple
vllm serve wtdcode/GLM-5.3-Flash-AWQ-W4A16 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name glm-5.3-flash \
  --tensor-parallel-size 4 \
  --max-model-len 524288 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

- Verified with MTP-3 on 4x A100-80GB (sm80). On 8x RTX A6000 (sm86) use `--tensor-parallel-size 8` and drop `--gpu-memory-utilization` to 0.85 — with the MTP draft loaded, 0.9+ OOMs during cudagraph warmup on 48 GB cards.
- Serving directly from the Hugging Face repository ID works; no manual snapshot-path resolution is needed.
- LMCache: the unified KV block size is 1152, so `--chunk-size` must be a multiple of 1152 (use `--chunk-size 1152 --separate-object-groups`), and add `--prefix-cache-retention-interval 1152 --max-num-batched-tokens 1152` to `vllm serve` (the LMCache fork requires a recurrent-state checkpoint at every chunk boundary and rejects MTP + align mode with a larger batch budget).

#### Qwen3.8-Flash-Next (v0.9.0+)

```bash
VLLM_PLE_CPU_OFFLOAD=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
vllm serve /path/to/your/qwen3.8 \
  --host=0.0.0.0 \
  --port=19088 \
  --served-model-name=qwen3.8-flash-next \
  --tensor-parallel-size=4 \
  --enable-expert-parallel \
  --max-model-len=1000000 \
  --max-num-seqs=8 \
  --max-num-batched-tokens=2048 \
  --gpu-memory-utilization=0.9 \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
  --enable-auto-tool-choice \
  --tool-call-parser=qwen3_xml \
  --reasoning-parser=qwen3 \
  --enable-prompt-tokens-details \
  --enable-prefix-caching \
  --mamba-cache-mode=align \
  --prefix-match-unit=16
```

- `VLLM_PLE_CPU_OFFLOAD=1` keeps the 51B n-gram embedding (fp8, ~51 GiB) in pinned host RAM via a separate `PleOffloadWorker` process. Without it the TP-sharded embedding adds ~12.8 GiB per GPU and KV memory goes negative on 48 GB cards.
- `--enable-expert-parallel` is required, not optional: with plain TP the 640-wide expert intermediate becomes 160 per rank, which is not a multiple of the 128x128 fp8 block, and vLLM then forces the Triton fp8 MoE kernel (no fp8 tensor cores on sm86). With EP the experts stay whole and the Marlin W8A16 backend is used.
- LMCache: the unified KV block size is 800, so `--chunk-size` must be a multiple of 800 (use `--chunk-size 800 --separate-object-groups`), and add `--prefix-cache-retention-interval 800 --max-num-batched-tokens 800` to `vllm serve` (same MTP + align-mode rule as GLM-5.3-Flash).
- AWQ W4A16 ([`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16), compressed-tensors `pack-quantized`, routed experts INT4 g128, everything else BF16): MTP speculative decoding works (the BF16 MTP draft is kept unquantized automatically). Verified on 4x A100-80GB: `VLLM_PLE_CPU_OFFLOAD=1 vllm serve wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 --tensor-parallel-size 4 --enable-expert-parallel --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. This achieves up to 936 tps.
- `--prefix-match-unit=16` is effectively mandatory. Prefix cache reuses whole blocks only, and this flag sets the block size; the remainder is never reused. It defaults to the mamba block size (800 here), and MTP drops one more block, so nothing under 1600 tokens can ever be reused and the hit rate reads a flat 0.0% no matter how often a prompt repeats. Setting it to 16 drops that floor to 32 tokens. The value must divide 800, so use 16 (or 25/40/50/80/100) -- 64 is rejected at startup. `--enable-mamba-fine-grained-prefix-cache` is optional on top; it only improves the worst-case hit.

#### DeepSeek V4 Flash (Preview, 0731, Vision-Exp)

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
  --served-model-name deepseek-v4-flash
```

- Keep `num_speculative_tokens` at 5 on Ampere. Values below 5 (the checkpoint's `dspark_block_size`) are rejected, and 7 needs ~200 KB of shared memory vs the 163 KB Ampere limit (`triton OutOfResources` error). 6 does start, but draft positions past the native block are almost never accepted (3–13% in our measurements), so it only wastes draft compute — output quality and speed are the same as 5.
- Requests that set neither `thinking` nor `reasoning_effort` now get thinking mode with high effort, matching the official 0731 API mapping (`reasoning_effort: "none"` restores plain chat mode). Agentic/tool-calling clients should pass a `reasoning_effort` explicitly from the first turn of a session — sessions that run without the effort prefix gradually stop thinking and can enter self-reinforcing reasoning loops.
- `--hf-overrides '{"head_dtype": "float32"}'` once helped reduce garbage outputs by improving precisions but might be not compulsory.
- LMCache: the KV block size is 256 (DeepSeek-V4-Flash and Vision-Exp), so `--chunk-size 1024` works for all three; add `--prefix-cache-retention-interval 1024` to `vllm serve` (the LMCache fork requires a state checkpoint at every chunk boundary for the sliding-window groups).

#### DeepSeek V4.1 Flash

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /path/to/DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --port 18005 \
  --served-model-name deepseek-v4.1-flash \
  --tensor-parallel-size 8 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload": true}' \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,24,48,96,192,384],"max_cudagraph_capture_size":384}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"use_local_argmax_reduction":true}' \
  --tokenizer-mode deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
```

- Expert parallel also works.
- LMCache: Note that the KV block size is 128.

#### MiMo-V2.6-Flash-RL

```bash
VLLM_MM_DISABLE_PINNED_H2D=1 \
vllm serve /home/ubuntu/models/MiMo-V2.6-Flash-RL-XiaomiMiMo \
  -tp 8 \
  --pipeline-parallel-size 1 \
  --trust-remote-code \
  --generation-config vllm \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --served-model-name mimo-v2.6-flash \
    --port 8000 \
    --host 0.0.0.0 \
    --uvicorn-log-level info \
    --max-num-seqs 8 \
    --seed 1234 \
    --max-model-len auto \
    -O3 \
    --no-use-tqdm-on-load \
    --performance-mode balanced \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512 \
    --enable-prefix-caching \
    --chat-template /home/ubuntu/models/MiMo-V2.6-Flash-RL-XiaomiMiMo/chat_template_fixed.jinja \
    --default-chat-template-kwargs '{"interleaved_thinking": true,"preserve_thinking": true}' \
    --gpu-memory-utilization 0.94 \
    --generation-config vllm \
    --attention-backend TRITON_ATTN_DIFFKV \
    --enable-prompt-tokens-details \
    --kv-cache-dtype fp8 \
    --skip-mm-profiling \
    --limit-mm-per-prompt '{"audio": 0, "image": 4, "video": 0}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

- Verified on 8x RTX 3090 (sm86, TP8) with MTP-3 and fp8 KV: single-stream decode ~177 t/s at 33K context (159 t/s with bf16 o_proj), prefill ~1840 tok/s at 33K prompts and ~2400 tok/s incremental on 20K chained-prefix context; 462 t/s aggregate at 16 concurrent streams; GSM8K 82.7%. The fused-fp8-QKV NB=4 sharding, MTP loader and sliding-window fixes are required for any serving of this checkpoint: TP8 previously produced garbage output (row-scrambled SWA layers), TP2/TP4 crashed at weight load.
- `--max-num-batched-tokens 512` is the measured sweet spot on this rig for single-stream latency (+15-29% incremental prefill vs 1024 at 20-50K context; 2048+ loses). At 512 the chunked-prefill all-reduces stay under the stock 8 MiB custom-AR cutoff, so no all-reduce tuning is needed. For multi-stream throughput raise `--max-num-seqs` instead (capture sizes auto-follow).
- `VLLM_MIMO_OPROJ_FP8=1` (opt-in): quantizes the 48 bf16 o_proj layers to online per-tensor fp8 (Marlin W8A16 on sm86) for +7-19% decode and +9% prefill at MNBT 1024; GSM8K parity measured (82.7% vs 82.0%). It is lossy (not bit-exact), hence opt-in. Clear `~/.cache/vllm/torch_compile_cache` once when toggling it, and keep the chunk budget below 2048 (Marlin W8A16 loses to bf16 cuBLAS at large chunk M).
- fp8 KV on sm<89 works on this path: the DiffKV kernels dequantize E4M3 through a LUT (no native fp8 math) and the store path quantizes via torch; the KV pool roughly doubles (787K tokens at 0.96 / MNBT 512 vs 331K fp16).
- The example shape (mm towers loaded — vision and audio load even at `audio: 0` — plus util 0.94) yields a ~433K-token fp8 KV pool; one live image costs ~150 MB on top. For text-only serving drop `--skip-mm-profiling` / `--limit-mm-per-prompt`, add `--language-model-only` and raise `--gpu-memory-utilization` to 0.96: the towers free ~1 GiB/card and the pool grows to ~650K+ tokens. Alternatively pin the pool size directly with `--kv-cache-memory` (the startup log prints the suggested value to fully utilize the card).
- `VLLM_CUSTOM_AR_ENFORCE=1` (opt-in, model-independent): auto-sizes the custom-all-reduce cutoff to the largest activation all-reduce so chunked prefill stays on the custom P2P path instead of falling back to NCCL above the 8 MiB default — measured +18-46% prefill at 13-21K prompts at MNBT 1024. Only relevant at chunk budgets >= 1024; unset keeps upstream behavior byte-for-byte.
- The `chat_template_fixed.jinja` patch adds `reasoning_content` round-trip and pre-opened thinking; `chat_utils` now also accepts `reasoning_content` on input for any model when `reasoning` is absent.
