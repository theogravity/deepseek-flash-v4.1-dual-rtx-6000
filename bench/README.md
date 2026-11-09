# Benchmarks and validation scripts

Run with `python3 -I <script>` (or `bash` for the shell ones) against the server on `localhost:8001`. They need the key the server was started
with: `export API_KEY=...`. Methodology, pitfalls and results: [`../docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

- `decode_bench2.py` — agentic-style greedy decode: ~25.6k-token warm code prefix, edits that copy from context; per-stream and aggregate tok/s at
  1/4/6 streams, two passes (use pass 2; pass 1 pays cache warm-up).
- `long_prefill.py <words> [seed]` — one huge prefill (320000 words is ~1M tokens, 160000 is ~500k) with a needle at the start, sampling GPU1/GPU2 VRAM each
  second. Edit the `-i 1,2` GPU indices in the script if your cards are elsewhere. Keep >=1.4 GiB margin at the peak.
- `mixed.py [seed]` — six concurrent requests (one ~1M tokens, five ~100k) with VRAM sampling. Its needle checks read False because `max_tokens=700` is
  eaten by reasoning; look at wall time, failures and peak VRAM.
- `six400.py [seed] [words]` — six concurrent ~404k-token requests, two passes (cold, then repeated).
- `lmc_ask.py <words> <seed>` / `lmc_test.py` — time to first token for a deterministic prompt; resend after pushing ~2M tokens through (or after a
  restart) to prove an LMCache restore: low TTFT, `External prefix cache hit rate` > 0 in vLLM's stats, `Retrieved N tokens` in `/root/.cache/lmcache-server.log`.
- `inline_system_test.py` — reproduces the Claude Code prefix-cache problem without Claude Code: plain vs inline-system vs folded messages, reuse read from vLLM's counters.
- `claude_code_cache.py` — real `claude -p` sessions through LiteLLM; per-turn prefix-cache hits from vLLM's counters (needs the `claude` CLI; see its docstring for env vars).
- `prefix_demo.py [words] [seed]` — time to first token for a cold prompt, the same prompt with text appended, one word changed in the middle and at the start: shows the caches are prefix caches.
- `interference.py` — decode tok/s of three streams alone vs while a ~295k-token cold prefill runs.
- `soak.py [minutes] [seed]` — six workers, mixed request sizes with repeated prompts; samples VRAM / host RAM / LMCache disk every 30 s and reports errors,
  answer correctness, latency and drift.
- `run_quality.sh` — Diffbot's validation suite (short sweep, tool calls, image, needles ~100k/250k/448k, GSM8K-200 + HumanEval-164 with thinking off).
  Needs the recipe checkout (`RECIPE`, default `/mnt/models/dsv41-flash-exl3/recipe`). HumanEval code runs through `quality_probe_sandboxed.py`
  (a modified copy of the recipe's probe) in a throwaway no-network container; it uses the recipe image by default (`SANDBOX_IMAGE`).
- `watch.sh` — logs host RAM, container RSS and VRAM every 5 s and stops the container if available RAM drops below 10 GiB (for first boots that risk a host OOM).

Use a new seed for every cold measurement; page and prefix caches make repeats look far faster than they are.
