"""Max-effort variant of decode_bench2.py (env EFFORT=max TEMP=1.0 TOP_P=0.95 MAX_TOKENS=6000): thinking ON, so the stream is reasoning then code.
Agentic-style decode benchmark: warm ~28k-token code prefix, greedy, edits that mostly copy from context.
Per-stream decode tok/s (completion tokens / time after first token) at 1, 4, 6 concurrent streams; run twice."""
import os as _os
def _api_key():
    k = _os.environ.get("DSV41_API_KEY") or _os.environ.get("LITELLM_MASTER_KEY")
    if not k:
        for line in open(_os.path.expanduser("~/llm/.env")):
            if line.startswith("LITELLM_MASTER_KEY="):
                k = line.split("=", 1)[1].strip()
    return k
AUTH = {"Authorization": "Bearer " + _api_key()}  # port 8001 requires the LiteLLM master key
import json, sys, threading, time, urllib.request
URL = "http://localhost:8001/v1/chat/completions"
funcs = []
for i in range(700):
    funcs.append(f"def func_{i}(x, y):\n    result = x * {i} + y\n    if result > {i * 3}:\n        return result - {i}\n    return result\n")
prefix = "Here is a Python module:\n\n" + "\n".join(funcs)
def stream(i, start, max_tokens, res):
    ask = f"\n\nRewrite func_{start} through func_{start + 40} with type hints and a one-line docstring each. Output only code."
    body = {"model": "DeepSeek V4.1 Flash", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens, "temperature": float(_os.environ.get("TEMP", "1.0")), "top_p": float(_os.environ.get("TOP_P", "0.95")), "reasoning_effort": _os.environ.get("EFFORT", "max"),
            "messages": [{"role": "user", "content": prefix + ask}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", **AUTH})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=3000) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0]["delta"].get("content") or ch[0]["delta"].get("reasoning") or ch[0]["delta"].get("reasoning_content")) and first is None:
                first = time.time()
    end = time.time()
    res[i] = (first - t0 if first else None, end - (first or t0), (usage or {}).get("completion_tokens", 0), (usage or {}).get("prompt_tokens", 0))
r0 = {}; stream(0, 0, 8, r0); print(f"warm-up: prompt {r0[0][3]} tokens, ttft {r0[0][0]:.1f}s", flush=True)
for rep in range(2):
    for n in (1, 4, 6):
        res = {}
        ths = [threading.Thread(target=stream, args=(k, 50 + 60 * k, int(_os.environ.get("MAX_TOKENS", "6000")), res)) for k in range(n)]
        t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]
        toks = sum(v[2] for v in res.values()); win = max(v[1] for v in res.values())
        per = [round(v[2] / v[1]) for v in res.values()]
        print(f"rep{rep} {n} streams: per-stream {per} tok/s, aggregate {toks / win:.0f} tok/s ({toks} tokens)", flush=True)
