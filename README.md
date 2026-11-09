# DeepSeek V4.1 Flash on 2× RTX PRO 6000 (Blackwell)

How to run [DeepSeek V4.1 Flash](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) (Diffbot's EXL3 3-bit
pack) on **two RTX PRO 6000 Blackwell cards** (tested on a 246 GiB workstation; see the RAM requirements below), tuned for **up to 6 concurrent long-context agents**
(512k max context, compaction at 400k), with an API key, a RAM + disk prompt cache and measured numbers.

This repo is the set of changes, patches, configuration and measurements we needed on top of Diffbot's recipe. It does **not**
contain the model or the recipe; you download those from the Hugging Face repo above. Everything here was developed and measured on
one machine, so treat the numbers as a reference point, not a guarantee.

## Why this exists

The upstream recipe pins about 230 GiB of host RAM (103 GiB of it is the two Engram lookup tables). On our 246 GiB host the stock
recipe is killed by the kernel OOM killer while loading. Getting it to run, and then to run well for several agents at once, took four
changes plus a lot of measuring:

| Change | What it does | Where |
|---|---|---|
| **Engram tables on NVMe** | Keeps the 103 GiB of int4 Engram tables in the checkpoint shards and reads only the rows each step needs (bit-exact with the stock kernel, works inside the decode CUDA graph). The pinned memory drops from ~230 GiB to ~77 GiB (measured, vLLM workers). | [`patches/engram-nvme`](patches/engram-nvme) |
| **LMCache, RAM + disk** | Prompt-cache tiers that restore a 1M-token prompt in ~2–3 s instead of 109 s of prefill, and survive restarts. | [`patches/lmcache`](patches/lmcache) |
| **API key** | `--api-key` plus a small auth patch so clients that send `x-api-key` (LiteLLM's `anthropic/` provider) work. | [`patches/api-auth`](patches/api-auth) |
| **Tuned serving config** | 512k window, 6 sequences, prefill chunk 8192, 2.77M-token KV pool, CPU host-experts for decode only. | [`compose/docker-compose.yml`](compose/docker-compose.yml) |

## Results (our machine)

Threadripper PRO 9965WX (24 cores), 8×32 GiB DDR5-5600, two RTX PRO 6000 (96 GB), PCIe 5 NVMe. Method and caveats in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| | |
|---|---|
| Single-stream cold prefill (time to first token) | 3.0 s @16k, 6.8 s @60k, 18.6 s @196k, 44.7 s @505k tokens (~11k tok/s at long lengths; [details](docs/BENCHMARKS.md#prefill)) |
| Single-stream decode (greedy code edits) | ~250 tok/s warm |
| Aggregate decode, 1 / 4 / 6 streams | ~250 / ~450 / ~535 tok/s |
| Six concurrent ~404k-token requests (2.4M tokens) | all complete: 265 s cold, 1–4 s when repeated |
| Restore of a 1M-token prompt evicted from the GPU | 1.9 s from RAM, 2.7 s from disk after a restart (109 s cold) |
| GPU KV pool | 2.77M tokens (5.3× a full 512k request) |
| Quality (recipe's own suite) | GSM8K-200 98.5 %, HumanEval-164 96.3 % (matches upstream's published GPU-only numbers), needles to 448k, tools, images |
| 20-minute soak, six workers | 105 requests, 0 errors, no VRAM drift |

## Requirements

- Two RTX PRO 6000 Blackwell Workstation (sm_120, 96 GB each) with working peer-to-peer between them, and a CUDA 13-capable driver (tested 615.71.09). Other cards are untested.
- Host RAM: ~77 GiB for vLLM (measured) + the LMCache RAM tier you choose (48 GiB in the compose file; 8–16 GiB is plenty for a few sessions) + OS.
  **Estimated minimum ~96–112 GiB, recommended 128 GiB; we only ever ran it on 246 GiB**, where the spare RAM acts as page cache for the Engram tables.
  Breakdown and reasoning: [`docs/TUNING.md`](docs/TUNING.md#host-ram-requirements).
- Linux (tested: Ubuntu 26.04), Docker with the NVIDIA container toolkit, `hf` CLI. About 500 GiB of disk: 322 GiB model, ~36 GB of Docker images,
  up to 128 GiB of cache; a fast NVMe for the model directory and the cache directory.
- A CPU with AVX-512 (F/BW/VNNI/VBMI/VBMI2) for the optional CPU host-expert runtime; ≥180 GB/s read bandwidth is the recipe's threshold.

## Quick start

A fuller, step-by-step version with prerequisite checks, timings and what a healthy boot looks like is in [`docs/INSTALL.md`](docs/INSTALL.md); every knob and the troubleshooting table are in [`docs/TUNING.md`](docs/TUNING.md).

```bash
# 1. model + recipe (upstream, not in this repo)
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir /mnt/models/dsv41-flash-exl3
R=/mnt/models/dsv41-flash-exl3/recipe
# 2. build the recipe image, its kernels and the CPU runtime (run by path, from another directory)
BUILD_MEM=24g BUILD_CPUS=12 MAX_JOBS=6 bash $R/docker/build.sh
MAX_JOBS=6 bash $R/kernels/build.sh
bash $R/cpuexp/build.sh
# 3. derive the LMCache image from it
bash patches/lmcache/build_image.sh
# 4. configure and start
cp compose/.env.example compose/.env      # set API_KEY, and MODEL_DIR / CACHE_DIR / GPU_A / GPU_B if needed
(cd compose && docker compose up -d)
# 5. check (about 5 minutes later)
export API_KEY=<the key you put in compose/.env>
curl -s -H "Authorization: Bearer $API_KEY" localhost:8001/v1/models
bash bench/run_quality.sh                  # the recipe's validation suite, sandboxed
```

The patches were generated against the image that the recipe's `docker/build.sh` produced on 2026-11-08 (vLLM nightly `5555a2d8`).
If the recipe or its pinned vLLM changes, regenerate them with the `apply_patch.py` in each patch directory; they refuse to run if
the code they patch has moved.

## Repo layout

```
compose/        docker-compose.yml (the whole serving config, commented) and .env.example
patches/
  engram-nvme/  NVMe-backed Engram tables: patcher, patched files, diff, model-free GPU test
  lmcache/      LMCache server launcher (container entrypoint) and the derived-image script
  api-auth/     vLLM auth that also accepts x-api-key
bench/          decode / prefill / concurrency / soak / interference / LMCache-restore scripts and the sandboxed quality suite
docs/           INSTALL.md (step-by-step install and build), TUNING.md (tuning table, troubleshooting), BENCHMARKS.md (methodology and results)
examples/       LiteLLM model entry
LICENSES/       third-party license texts
```

## Known limits

- One machine, one workload mix. Decode numbers are for greedy code-edit prompts; free-form creative text at temperature 0.6 measured
  about 3× lower because speculative decoding accepts fewer drafts.
- Under saturation, small requests queue behind huge cold prefills (vLLM schedules FIFO); sessions already decoding are not slowed.
- Cold prefill is ~10–30% slower than with the recipe's 16384 chunk; that is the price of the larger KV pool (see the tuning table for
  the alternatives, including a 1M-token-window variant).
- Quality beyond the recipe's suite (GSM8K / HumanEval / needles / tool calls / images) is not measured; there is no long-horizon agentic eval.
- LMCache's docs list speculative decoding as unvalidated; it works here with the recipe's DSpark head, but you are off the beaten path.

## Credits and licenses

This is an independent write-up, not affiliated with Diffbot, DeepSeek, vLLM or LMCache. The model pack, recipe, kernels and the
CPU runtime are Diffbot's ([MIT](LICENSES/Diffbot-recipe-MIT.txt)); [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) is
patched in three places; [LMCache](https://github.com/LMCache/LMCache) (Apache-2.0) provides the cache server. Files in this repo
that are modified copies of upstream code are listed in [`NOTICE.md`](NOTICE.md). Original files here are MIT ([`LICENSE`](LICENSE)).
