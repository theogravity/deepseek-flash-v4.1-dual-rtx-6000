# Benchmarks and methodology

Everything below was measured on one machine on 2026-11-09: Threadripper PRO 9965WX (24 cores, 4 CCDs, AVX-512), 8×32 GiB DDR5-5600
(246 GiB usable), two RTX PRO 6000 Blackwell Workstation cards (97,887 MiB each, peer-to-peer enabled), PCIe 5 NVMe drives, the
recipe image built from Diffbot's pack (vLLM nightly `5555a2d8`). The scripts are in [`../bench`](../bench); the server is the one in
[`../compose/docker-compose.yml`](../compose/docker-compose.yml) unless a row says otherwise. Run-to-run noise is real (first passes
include cache warm-up), so where a number is "warm" it is the second pass.

## Methodology and pitfalls

- **Decode** (`bench/decode_bench2.py`): a ~25.6k-token Python module is the warm prefix, then each stream asks for
  type-hinted/docstring rewrites of a different slice of it (greedy, up to 1500 tokens). Per-stream rate = completion tokens / time after
  the first token; "aggregate" = tokens over the longest stream's decode window. 1, 4 and 6 concurrent streams, two passes. This
  prompt style mostly copies from context, so speculative decoding accepts many drafts (like an agent editing files). Free-form
  creative text at temperature 0.6 on the same server measured ~60–95 tok/s single-stream, about 3× lower.
- **Prefill** (`bench/lmc_ask.py`, `bench/long_prefill.py`): prompts of random "codeNNNNN" words (~3.1 tokens per word) with a needle at
  the start; time to first streamed token = prefill time. Every cold measurement uses a new seed: the OS page cache keeps the Engram rows
  and the prompts that were touched before, and an early run with a fixed seed looked 5–10× too fast.
- **VRAM peaks** sample `nvidia-smi` once a second (0.5 s for the mixed test) during the run; the margin is against 97,887 MiB.
- **Quality** (`bench/run_quality.sh`): Diffbot's own validation suite from the recipe. Its HumanEval step executes model-written code, and
  the upstream probe runs it on the host; `bench/quality_probe_sandboxed.py` runs each program in a throwaway container instead (no network,
  read-only root, non-root, no capabilities, memory/pid/CPU caps, with a startup self-test).

## Single stream (concurrency 1), final configuration

| Prompt | Cold time to first token | Prefill rate |
|---|---|---|
| 16k tokens | 3.0 s | ~5.3k tok/s (fixed per-request overhead dominates) |
| 60k tokens | 6.8 s | ~8.8k tok/s |
| 196k tokens | 18.6 s | ~10.5k tok/s |
| 505k tokens | 44.7 s | ~11.3k tok/s |

Decode, greedy code edits on the 25.6k warm prefix: **~250 tok/s warm** (192 on the first pass). Cached prompts (GPU prefix cache or LMCache)
return their first token in ~0.3–2.7 s at any size we tried (see LMCache below).

## Concurrency

Aggregate decode, warm second pass, final configuration: **1 / 4 / 6 streams = 251 / 446 / 534 tok/s** (first pass 206 / 427 / 548).

Six concurrent requests of ~404k tokens each (2.4M prompt tokens, more than the 2.77M-token KV pool can hold alongside decode), needle
at the start of each:

| Configuration | Cold pass wall time | Same six again | VRAM peak |
|---|---|---|---|
| Final (chunk 8192, KV 11 GiB, pool 2.77M tokens) | 265 s (all six found the needle) | 4 s total, 1.4–3.5 s each | 95,872 MiB |
| Chunk 16384, KV 8.5 GiB, pool 1.73M tokens | 178 s | 4 s total, 1.1–3.8 s each | 96,546 MiB |

A ~295k-token cold prefill does not slow sessions that are already decoding: three decode streams ran 142–153 tok/s alone and 146–149 tok/s
while the prefill ran alongside (28 s). Under saturation, however, *new* small requests wait behind big prefills (FIFO scheduling): in the
soak below, the median for a ~6k-token prompt was 54 s.

## CPU host-expert runtime (A/B, same server otherwise)

| | 1 stream | 4 streams | 6 streams |
|---|---|---|---|
| GPU-only (PCIe path) warm decode, tok/s | 231 | 427 | 490 |
| CPU experts for decode warm decode, tok/s | 258 | 488 | 574 |
| gain | +12% | +14% | +17% |

CPU experts also for **prefill** (`EXL3_CPU_EXPERTS_PREFILL=1`) were slower: 16k prefill 7.0–8.7k vs 10.3k tok/s, 60k prefill 9.5–10.3k vs
12.2–12.5k tok/s, six concurrent 60k requests 52 s vs 44 s. Hardware check: `cpuexp/check_host.sh --bandwidth` read 251.8 GB/s on the pool CPUs
(pools derived as `0-4,6-10;12-16,18-22`, 10 cores per rank).

Removing the 512-slot VRAM expert cache (to give VRAM to the KV pool) made the CPU jobs ~5× slower (mean 2.9 ms vs ~0.5 ms) and decode
149 / 201 / 214 tok/s at 1 / 4 / 6 streams, so the cache stays.

## LMCache (RAM 48 GiB + disk 128 GiB)

| Case | Time to first token |
|---|---|
| 60k-token prompt, cold | 7.6–8.1 s |
| same prompt, GPU prefix-cache hit | 0.3–0.8 s |
| same prompt after a full container restart (RAM tier empty, restored from disk) | 1.46 s (server log: 234/234 keys from L2 in 372 ms; 59,904 tokens retrieved in 0.09 s) |
| 1M-token prompt, cold | 109 s |
| 1M-token prompt pushed out of the GPU pool by another 1M prompt, then resent | 1.9 s (restored from RAM) |
| 1M-token prompt resent after a restart | 2.7 s (restored from disk) |

Entries are ~1.8 KB per token (a 60k-token prompt wrote 103 MiB; ~12M tokens of test prompts occupied 21 GiB on disk). The restored 60k-token output was identical to the cold run, and restored 1M-token prompts returned the correct needle. With LMCache on and `expandable_segments` off, the 1M-token cold prefill is ~109 s (9.3k tok/s) versus
92.7 s (10.8k tok/s) for the earlier no-LMCache configuration.

## Engram-on-NVMe micro-benchmark (`patches/engram-nvme/test_nvme_lookup.py`)

Rows drawn at random from the whole 384M-row table (cold), one layer, one rank, torch intra-op threads set to 1 as in serving:

| Lookup | Time |
|---|---|
| decode step, 6 tokens (72 rows) | 0.4–0.6 ms |
| decode step, 48 tokens (576 rows) | 1.6–4.6 ms across runs |
| prefill chunk, 16384 tokens (196,608 rows) | 0.42–0.55 s (0.93 s with the single-threaded first version) |

In serving, the first version made prefill run at ~1–1.4k tok/s (the gather ran at queue depth 1); the 32-thread gather brought it to
~10k tok/s. The same test on a copy of the shards on a different PCIe-5 NVMe drive was 35–45% slower (0.67–0.81 s vs 0.49–0.56 s per 16k
chunk with fresh rows on both).

## Quality

Diffbot's suite, via `bench/run_quality.sh` on this setup (their published numbers in parentheses, GPU-only / CPU-runtime):

| Check | Result |
|---|---|
| Short-context sweep | 11/11 |
| Tool calls (`tool_choice=auto` + reasoning parser) | OK |
| Image input | 4/4 (reads a code word from an image, colours, two images in one message) |
| Needle at ~100k / ~250k / ~448k tokens | found at 100,834 / 249,634 / 448,034 tokens |
| GSM8K first 200 (greedy, thinking off) | **197/200 = 98.5 %** (98.5 / 99.0 %) |
| HumanEval 164 (greedy, thinking off, sandboxed) | **158/164 = 96.3 %** (96.3 / 96.3 %) |

So the Engram patch, LMCache, the CPU runtime and the auth patch did not move these numbers. Not measured: long-horizon agentic evals,
quality at 1M tokens, anything outside this suite.

## Soak (`bench/soak.py 20`)

Six workers submitting continuously for 20 minutes with prompts of ~6k, ~90k and ~390–420k tokens (40 distinct prompts per size so the GPU and
LMCache caches churn), final configuration: **105 requests, 0 errors, 0 wrong answers.** VRAM max 96,206 MiB and identical at the first and last
sample; host RAM available 107 → 105 GiB; container healthy throughout; LMCache disk grew 23 → 47 GiB. Median latency: ~54 s for ~6k-token
prompts, ~73 s for ~90k, ~73–103 s for 380–420k-token prompts (queueing behind big prefills).

## The failure that started this

Stock recipe on this 246 GiB host, first boot: the kernel logged a global OOM (`constraint=CONSTRAINT_NONE`) and killed a `VLLM::Worker_TP`
holding ~35 GB anonymous and ~75 GB shared/pinned memory, about a minute into loading, with the container at ~213 GiB. With the Engram patch the
the vLLM workers hold ~77 GiB of non-reclaimable memory (plus the LMCache RAM tier) and loading completes in ~4–5 minutes.

Measured memory of the running server (LMCache RAM tier 48 GiB): cgroup anonymous memory 124.5 GiB (LMCache server 49.0 GiB, two vLLM workers 37.3 GiB each, API server
and engine ~2.5 GiB) and 98.9 GiB of reclaimable page cache; host `free`: 138 GiB used, 108 GiB available.
