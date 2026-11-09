"""Measure prefix-cache reuse in REAL Claude Code sessions against this server, from vLLM's own counters.

Runs `claude -p` turns (read-only tools, no MCP servers) in WORKDIR, one conversation resumed turn by turn plus a second new conversation, and reports per turn how many
prompt tokens vLLM served from its prefix cache (vllm:prompt_tokens_by_source_total: local_cache_hit / local_compute / external_kv_transfer = LMCache).
Needs the `claude` CLI. Idle server recommended (the counters are server-wide).
env: BASE_URL (default http://localhost:4000, LiteLLM), CLIENT_KEY (key for BASE_URL; default $API_KEY), METRICS_URL (default http://localhost:8001/metrics),
     MODEL (default "DeepSeek V4.1 Flash"), WORKDIR (default: this repo's root).   usage: python3 -I claude_code_cache.py"""
import json, os, subprocess, sys, time, urllib.request, uuid
BASE_URL = os.environ.get("BASE_URL", "http://localhost:4000")
KEY = os.environ.get("CLIENT_KEY") or os.environ.get("API_KEY")
if not KEY: raise SystemExit("set CLIENT_KEY (or API_KEY) to the key BASE_URL expects")
METRICS_URL = os.environ.get("METRICS_URL", "http://localhost:8001/metrics")
M = os.environ.get("MODEL", "DeepSeek V4.1 Flash")
WORKDIR = os.environ.get("WORKDIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE") and not k.startswith("ANTHROPIC")}
ENV.update({"ANTHROPIC_BASE_URL": BASE_URL, "ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": KEY,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": M, "ANTHROPIC_DEFAULT_SONNET_MODEL": M, "ANTHROPIC_DEFAULT_HAIKU_MODEL": M,
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "400000", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "524288", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
            "MAX_THINKING_TOKENS": "31999", "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
def metrics():
    t = urllib.request.urlopen(METRICS_URL, timeout=30).read().decode()
    def g(name, label=None):
        return sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines() if l.startswith(name + "{") and (label is None or label in l))
    return {"reqs": g("vllm:request_success_total"), "prompt": g("vllm:prompt_tokens_total"),
            "compute": g("vllm:prompt_tokens_by_source_total", 'source="local_compute"'),
            "local_hit": g("vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'),
            "external": g("vllm:prompt_tokens_by_source_total", 'source="external_kv_transfer"'), "gen": g("vllm:generation_tokens_total")}
totals = {"prompt": 0.0, "compute": 0.0}
def turn(label, prompt, session, resume=False):
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--allowedTools", "Read", "Glob", "Grep", "--strict-mcp-config", "--effort", "max",
           "--resume" if resume else "--session-id", session]
    before = metrics(); t0 = time.time()
    p = subprocess.run(cmd, cwd=WORKDIR, env=ENV, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0; after = metrics(); d = {k: after[k] - before[k] for k in before}
    try: out = json.loads(p.stdout)
    except Exception: out = {"result": (p.stdout or p.stderr)[-300:], "num_turns": "?"}
    pt = d["prompt"] or 1; totals["prompt"] += d["prompt"]; totals["compute"] += d["compute"]
    print(f"{label:30s} {dt:5.1f}s  requests {d['reqs']:3.0f}  prompt tokens {d['prompt']:7.0f}  prefix-cache hits {d['local_hit']:7.0f} ({100*d['local_hit']/pt:5.1f}%)  "
          f"LMCache {d['external']:6.0f}  computed {d['compute']:7.0f} ({100*d['compute']/pt:5.1f}%)", flush=True)
    print(f"    agent steps {out.get('num_turns')}; answer: {str(out.get('result'))[:100]!r}", flush=True)
s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
print("=== conversation 1 (resumed turn by turn)")
turn("1.1 read a doc section", "Read docs/TUNING.md and tell me in two sentences what section 4b says.", s1)
turn("1.2 read another file", "Now read patches/lmcache/start.sh. What is the fallback L1 size?", s1, True)
turn("1.3 search", "List every file in this repo that references API_KEY.", s1, True)
turn("1.4 no tools", "Summarize your three previous answers in one short paragraph.", s1, True)
print("=== conversation 2 (brand-new session, same directory and settings)")
turn("2.1 new session", "Read bench/soak.py and say in one sentence what it does.", s2)
turn("2.2 resume", "How many worker threads does it start?", s2, True)
print(f"\ntotal: {totals['prompt']:.0f} prompt tokens, {totals['compute']:.0f} computed ({100*totals['compute']/max(totals['prompt'],1):.1f}%)")
