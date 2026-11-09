# Setup, tuning and troubleshooting

Reference for running DeepSeek V4.1 Flash (Diffbot EXL3 3-bit pack) on two RTX PRO 6000 Blackwell cards with the configuration in
[`../compose/docker-compose.yml`](../compose/docker-compose.yml). Target workload: **up to 6 concurrent agents, 512k max context, compaction
at 400k.** All numbers were measured on one machine (Threadripper PRO 9965WX, 8×32 GiB DDR5-5600, 246 GiB usable, PCIe 5 NVMe) on
2026-11-09; see [BENCHMARKS.md](BENCHMARKS.md) for methodology.

## 1. What runs where

```
host RAM  (~77 GiB vLLM + LMCache RAM tier, 48 GiB here)       NVMe
  pinned: ~1/3 of the routed experts, embeddings              checkpoint shards (322 GiB), incl. the two Engram tables (103 GiB)
  LMCache server (same container): L1 RAM tier  ----------->  LMCache disk tier (128 GiB cap, LRU) + kernel JIT caches
GPU x2 (TP2): weights ~77 GiB each, 512-slot expert cache, fp8 KV pool 11 GiB each
CPU: 10 pinned cores per rank compute the host-resident experts during decode
```

One container, one vLLM process tree plus the LMCache server (they must share a container: the server opens vLLM's GPU memory through
CUDA IPC). Port 8001 requires an API key.

## 2. Rebuild from scratch

The full step-by-step version, with prerequisite checks and what a healthy boot looks like, is [INSTALL.md](INSTALL.md); this section is the condensed form.

Run scripts by path from a directory other than the download (interpreters load code from the current directory).
If a build script prints nothing and exits 0, check `echo $SHELLOPTS`: an exported `onecmd` option makes every child `bash` stop after
its first line (use `env -u SHELLOPTS bash script.sh`).

```bash
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir /mnt/models/dsv41-flash-exl3   # ~322 GiB
R=/mnt/models/dsv41-flash-exl3/recipe
BUILD_MEM=24g BUILD_CPUS=12 MAX_JOBS=6 bash $R/docker/build.sh   # recipe image dsv41-flash-1017-exl3-3bpw-sm120, ~10 min
MAX_JOBS=6 bash $R/kernels/build.sh                              # 5 sm_120 kernels, ~2 min, no GPU
bash $R/cpuexp/build.sh                                          # CPU host-expert runtime, ~1 min, built with -march=native for THIS cpu
bash $R/cpuexp/check_host.sh --bandwidth                         # server stopped; the recipe wants >=180 GB/s (we measured 251.8)
bash patches/lmcache/build_image.sh                              # derived image with lmcache 0.5.7 (--no-deps)
cp compose/.env.example compose/.env && $EDITOR compose/.env    # API_KEY, optional MODEL_DIR / CACHE_DIR / GPU_A / GPU_B
cd compose && docker compose up -d                               # ~5 min to healthy; first boot also compiles kernels
```

If the recipe's image changes (different vLLM, different recipe commit), regenerate the mounted patches from the new image's stock files:

```bash
IMG=dsv41-flash-1017-exl3-3bpw-sm120
V=/usr/local/lib/python3.12/dist-packages/vllm
docker run --rm --entrypoint cat $IMG $V/models/deepseek_v41/common/engram.py                     > engram.py
docker run --rm --entrypoint cat $IMG $V/model_executor/model_loader/ep_weight_filter.py          > ep_weight_filter.py
docker run --rm --entrypoint cat $IMG $V/entrypoints/serve/middleware/authenticate.py            > authenticate.py
python3 -I patches/engram-nvme/apply_patch.py engram.py ep_weight_filter.py patches/engram-nvme/patched
python3 -I patches/api-auth/apply_patch.py authenticate.py patches/api-auth/patched
```

Each patcher fails loudly if the code it patches has moved. Then run the model-free test in `patches/engram-nvme/README.md`.

## 3. Changing things: budgets and the tuning table

Per GPU: 97,887 MiB (95.6 GiB). Weights, the expert cache and CUDA graphs take ~83 GiB, the KV pool is whatever
`--kv-cache-memory` says, and a long prefill adds a transient peak on top. **Rule: peak VRAM during a ~500k-token prefill must stay
at least ~1.4 GiB under 97,887 MiB.** Measure with `bench/long_prefill.py 160000` (it samples `nvidia-smi` each second). The failure to
avoid is a VRAM OOM mid-prefill, which crashes the engine, not a slow request.

| Variant (all with CPU decode experts, LMCache, NVMe Engram) | max-model-len | seqs | chunk | KV/GPU | GPU pool | VRAM peak, 498k–1M prefill | notes |
|---|---|---|---|---|---|---|---|
| **Current (compose file)** | 524288 | 6 | 8192 | 11.0 GiB (11811160064) | 2,769,546 tok (5.3× a request) | 95,872 MiB (1.97 GiB margin) | six 404k requests all complete; cold prefill ~10–30% slower than chunk 16384 |
| 512k, big chunk | 524288 | 6 | 16384 | 8.5 GiB (9126805504) | 1,729,490 tok | 96,546 MiB (1.3 GiB margin) | fastest cold prefill (~10.5k tok/s); six 404k requests complete (178 s cold) |
| 1M window | 1048576 | 6 | 16384 | 6.5 GiB (6979321856) | 1,638,858 tok | 96,384 MiB (1.5 GiB margin) | the model's native max; compact around 850k |
| 1M, no LMCache | 1048576 | 6 | 16384 | 7.5 GiB | 1,891,020 tok | 96,460 MiB (1.4 GiB margin) | without LMCache `expandable_segments` can stay on, ~1.5 GiB more usable |
| Recipe default | 1048576 | 8 | 16384 | 4.25 GiB | ~1.07M tok | ~96,655 MiB (1.2 GiB margin, recipe's figure) | what the recipe ships |

To switch variants set `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--kv-cache-memory` and the
`--cudagraph-capture-sizes` list (it must reach `seqs × (1 + spec tokens)` = 36 for 6 sequences and k=5:
`1 2 4 6 8 12 16 24 32 36`). The Engram staging buffers follow `--max-num-batched-tokens` automatically. Compaction rule of thumb: the
window minus ~64k of output minus ~60k of overshoot, about 76–80% of it.

What each knob costs (measured):

- **Prefill chunk (`--max-num-batched-tokens`).** The VRAM peak is dominated by it, not by context length (498k tokens: +5.1 GiB over idle
  at 16384, 1M tokens: +6.1 GiB; 8192 saves ~3.2 GiB). Smaller chunk = more KV room, slower cold prefill, unchanged decode. Pool tokens
  per GiB also rise (203k/GiB at 16384, ~250k/GiB at 8192).
- **CPU host experts (`EXL3_CPU_EXPERTS`, `EXL3_CPU_EXPERTS_PREFILL`).** Decode +12–17% vs GPU-only (warm 1/4/6 streams: 258/488/574 vs
  231/427/490 tok/s). CPU prefill was 15–30% *slower*, so it stays off. The pools are 10 cores per rank on a 24-core part (4 CCDs × 6
  cores, one core per CCD left for vLLM); the CPU side is the weak link.
- **VRAM expert cache (`EXL3_HOST_CACHE=512` with `plan-1017-agent30-c512.json`): do not drop it to grow KV.** With 10 pool cores the CPU
  jobs went from ~0.5 ms to ~2.9 ms mean and decode fell 2.3× at 4 streams (149/201/214 vs 258/488/574 tok/s). The recipe's note that the
  cache "matters little with CPU experts" was measured on 42-core pools.
- **LMCache tiers (`DSV41_LMCACHE_L1_GB`, `_L2_GB`).** Entries are ~1.8 KB per token (60k tokens = 103 MiB on disk), so 48 GiB of RAM holds
  ~27M tokens and 128 GiB of disk ~70M. Every entry is written to both tiers; RAM is the hot read path. Restoring a 1M-token prompt takes
  ~1.9 s from RAM and ~2.7 s from disk after a restart (109 s cold). The RAM tier is pinned and allocated at boot.
- **Speculative head.** The released DSpark head with k=5 stays. The recipe's code head (`drafter-code-k7`) is for single-stream code; its own
  docs show four streams dropping from 511 to ~430 tok/s aggregate, so it is the wrong trade for six sessions.
- **Engram location.** Tables stay on the model drive. A copy on a PCIe-5 drive measured 35–45% *slower* for the gather (same Samsung
  9100 PRO model on both drives; small random reads are not link-bound).
- **Host RAM.** ~77 GiB for vLLM plus the LMCache RAM tier you choose; see *Host RAM requirements* below.

### Host RAM requirements

Measured on the running server (LMCache RAM tier set to 48 GiB), from the container's cgroup `memory.stat` and per-process RSS:

| Component | Memory | Notes |
|---|---|---|
| vLLM workers (2 × 37.3 GiB) + API server and engine (~2.5 GiB) | **~77 GiB** | non-reclaimable: pinned host-resident experts, embeddings, runtime; does not depend on any LMCache setting |
| LMCache RAM tier | 49 GiB resident at `DSV41_LMCACHE_L1_GB=48` | whatever you configure, plus ~1 GiB of overhead; pinned, allocated at boot |
| Page cache inside the container | ~99 GiB at the time | reclaimable; holds the Engram rows and weights that were read, so lookups on hot rows avoid the NVMe |
| OS / desktop | ~10 GiB | on our machine |

- **Minimum (estimate): about 96–112 GiB** = ~77 GiB (vLLM) + a small LMCache tier (8–16 GiB; entries are ~1.8 KB per token, so 8 GiB holds ~4M tokens) + OS and
  slack. At that size there is little page cache, so loading and Engram lookups run colder from NVMe (slower, not different in output).
- **Recommended: 128 GiB** with `DSV41_LMCACHE_L1_GB` around 16–24 (estimate).
- **What we ran: 246 GiB** with L1 = 48 GiB. The spare ~100 GiB acts as page cache for the 103 GiB of Engram tables.
- **More than ~250 GiB should not help** (77 GiB vLLM + 48 GiB LMCache + the whole 103 GiB Engram table cached + OS already fits): this is reasoning, not a measurement.
- Only 246 GiB was ever tested; the minimum and recommended figures are estimates from the breakdown above. Whatever you choose, set the compose
  `mem_limit` / `memswap_limit` to at least vLLM + L1 + a few GiB (page cache inside the limit is reclaimed under pressure), and `DSV41_LMCACHE_L1_GB`
  accordingly. The disk tier (`DSV41_LMCACHE_L2_GB`, 128 GiB cap) is independent of RAM.
- The stock recipe, without the Engram patch, pins ~230 GiB and does not fit on a 246 GiB host.

## 4. Why each non-obvious piece exists

- **Engram tables on NVMe.** The stock recipe pins ~230 GiB of host RAM, of which 103 GiB is the two int4 Engram tables (shards 47/48);
  on a 246 GiB host the kernel's OOM killer fires while the workers load (a worker holding ~35 GB anonymous plus ~75 GB shared/pinned
  memory is killed; the Python error you see is a `safetensors` `ValueError` from the dying worker). The patch skips those two
  tensors in the loader and serves lookups from the mmapped shards: each lookup copies the token-hash ids to a pinned buffer, a CUDA
  *host-function node* gathers the needed rows (32 threads, `posix_fadvise` hints for small batches) into pinned staging, they are copied to
  the GPU, and the *stock* int4 Triton kernel runs on that compact table. Everything is shape- and argument-static per token count, so it is
  captured inside the FULL decode CUDA graph (decode runs as a single graph, so a host sync there is impossible). Output is bit-exact with
  the stock kernel (`patches/engram-nvme/test_nvme_lookup.py`).
- **LMCache in the same container.** The multiprocess server shares GPU memory with vLLM via CUDA IPC. It needs `SYS_PTRACE` and
  `seccomp=unconfined` (`pidfd_getfd`), `PYTORCH_CUDA_ALLOC_CONF` unset (IPC does not work on expandable-segment or
  `--enable-cumem-allocator` VMM memory), and an `eviction` block in the `fs_native` adapter (otherwise `max_capacity_gb` is not
  enforced and the disk tier grows without bound).
- **`--api-key` plus the auth patch.** The port binds all interfaces on the host network. vLLM's stock check only accepts
  `Authorization: Bearer`, but LiteLLM's `anthropic/` provider (needed to keep thinking blocks working) sends `x-api-key`, so a protected
  server answered LiteLLM with `401`. `patches/api-auth` accepts either header and keeps the SHA-256 constant-time comparison. `/health` is
  unauthenticated (the compose healthcheck uses it). The key is visible in `docker inspect`/`ps` on the host, like any `--api-key`.
  If you use LiteLLM, see [`../examples/litellm.yaml`](../examples/litellm.yaml).
- **`restart: "no"`.** A crash loop leaks `vllm_offload_*.mmap`, `lmcache_l1_pool_*` and `psm_*` files in `/dev/shm`.

## 4b. Caching: what is reused (and what is not)

Both caches are **prefix caches**: a request reuses stored KV only for the longest run of identical tokens starting at token 0.

- **vLLM's automatic prefix cache** keeps KV blocks in GPU memory (on by default: the engine log shows `enable_prefix_caching=True`; block size 64). Each block's hash includes
  the blocks before it, so one different token invalidates everything after it.
- **LMCache** (RAM + disk) stores the same KV in 256-token chunks (`--chunk-size 256`) keyed by the token prefix, and restores whole chunks only. A restore that returned
  59,904 of a 60,080-token prompt (234 × 256) prefilled the remaining 176 tokens normally. Its non-prefix modes (`--enable-segmented-prefix`, `--enable-dedup-content`,
  `--enable-blend-reorder`) are off, and LMCache's docs list CacheBlend (position-independent reuse) as unsupported for this model.
- **Model specifics.** The sliding-window layers (window 128) are not stored: vLLM marks that cache group as not prefix-cacheable and replays the last 128 tokens of any hit, and
  LMCache skips the group. The other, prefix-cacheable groups (the compressed-attention caches and their indexers) are what get stored and restored. Cached tokens are not looked up in the Engram tables again; new tokens still hash the preceding token ids.

What that means in practice (`bench/prefix_demo.py`, 60k-token prompt, time to first token; numbers in [BENCHMARKS.md](BENCHMARKS.md#prefix-cache-behaviour)):

| Change to the prompt | Reuse | TTFT |
|---|---|---|
| none (cold) | none | 8.1 s |
| text appended at the end | everything but the new text | 0.75 s |
| one word changed at 50% | the first half | 2.3 s |
| one word changed at the very start | nothing | 4.5 s (a full recompute that ran faster than the cold run only because the OS page cache already held the Engram rows from run A) |

For agents: keep the start of the prompt stable (system prompt, tool definitions, earlier turns) and only append; editing or inserting near the start recomputes everything after
it, and compaction resets the prefix. The recipe strips Claude Code's `x-anthropic-billing-header` from system messages (`VLLM_DS41_API_SYSTEM_CLEAN=1`) so that header does not break the
prefix between requests. We did not measure hit rates in a real agent session. To see whether a restore happened, look for `Retrieved N tokens` and `Prefetch request completed` in the LMCache
server log; vLLM's cumulative `External prefix cache hit rate` moved only a little in our tests, so it is not a reliable per-request signal.

## 5. Troubleshooting (every one of these was hit)

| Symptom | Cause / fix |
|---|---|
| `bash script.sh` prints nothing, exits 0, `bash -x` shows no trace | An exported `SHELLOPTS` containing `onecmd`; run with `env -u SHELLOPTS`. `bash -c '...'` is unaffected. |
| Container dies ~1 min into load, kernel log `Out of memory: Killed process ... VLLM::Worker_TP` (`journalctl -k`), `ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'` | Host RAM exhausted by the pinned Engram tables. `DSV41_ENGRAM_NVME=1` plus the mounted patched files fix it. (`DSV41_ENGRAM_DISK`, set to 0 by the recipe, belongs to a different recipe's patch and is inert here.) |
| `RuntimeError: cannot pin 'torch.cuda.LongTensor'` at init | vLLM sets a CUDA default device during model init; CPU staging buffers need `device="cpu"` (already in the patch). |
| `Cannot copy between CPU and CUDA tensors during CUDA graph capture` | A host sync inside the captured decode graph; the Engram lookup must use the host-node design (no `.cpu()`). |
| `CUDA error: capturing stream has unjoined work` | Secondary error after an exception during graph capture; read the *first* traceback. |
| Prefill ~1k tok/s, GPUs alternate 0%/100%, one worker thread at ~93% CPU | The Engram gather is single-threaded because vLLM sets torch's intra-op threads to 1 for serving; the patch uses its own 32-thread pool. |
| `KV connector LMCacheMPConnector is incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Unset that variable; do **not** use `--enable-cumem-allocator` (next row). |
| LMCache `_share_cuda_` → `CUDA error: invalid argument` | CUDA IPC on VMM/cumem memory is unsupported; same fix (unset expandable segments, no cumem). |
| LMCache server `pidfd_getfd: Operation not permitted` | Add `SYS_PTRACE` and `seccomp=unconfined` to the service. |
| LMCache warns `max_capacity_gb ... no 'eviction' block` | Add `"eviction":{"eviction_policy":"LRU","trigger_watermark":0.8,"eviction_ratio":0.2}` to the adapter JSON (`start.sh` does). |
| Decode ~2.3× slower at 4 streams after removing the expert cache | See section 3: keep `EXL3_HOST_CACHE=512`; the cpuexp log line shows `mean ~2900 us` instead of ~500 us. |
| Clients using `x-api-key` get `401 {"error":"Unauthorized"}` | The `patches/api-auth/patched/authenticate.py` mount is missing or stale. |
| Compose: `required variable API_KEY is missing a value` | Create `compose/.env` (copy `.env.example`). The compose file refuses to start without a key so vLLM is never left unauthenticated. |
| `docker compose up --force-recreate` takes > 3 min | The old container is stopping (`stop_grace_period: 240s`) or its engine hung; run it in the background and poll. |
| `docker logs` shows two runs mixed | `docker restart` reuses the container; use `docker logs --since <boot time>` or recreate. |
| vLLM `/reset_prefix_cache` returns 200 but the next request is still a GPU cache hit | It did not evict. Prove an LMCache restore by pushing ~2M tokens of other prompts through, or restart (the disk tier persists). Check vLLM's `External prefix cache hit rate` and the server's `Retrieved N tokens` log lines. |
| Benchmark numbers look 5–10× too good on a repeat run | Page cache (Engram rows, prompts) and prefix caches; use a fresh seed for every cold measurement. |

## 6. Not done / ideas

- A host-expert plan built from your own routing statistics (`EXL3_EXPERT_STATS` + the recipe's `plans/make_plan.py`) could free VRAM for KV without
  hurting decode. Not needed here (the pool already holds six 400k sessions), so not attempted.
- `--scheduling-policy priority` could keep small requests from queueing behind huge cold prefills; untested.
- Quality beyond the recipe's suite: no long-horizon agentic evals and no 1M-token quality tests.
- With more CPU cores per pool, re-test CPU prefill and the VRAM-cache-off + bigger-KV trade.
