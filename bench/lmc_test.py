"""LMCache restore test: cold prompt -> reset GPU prefix cache -> same prompt again; compare TTFT and greedy output.
usage: lmc_test.py <words> <seed>"""
import os as _os
def _api_key():
    k = _os.environ.get("API_KEY") or _os.environ.get("DSV41_API_KEY")
    if not k:
        raise SystemExit("set API_KEY to the key the server was started with (port 8001 requires it)")
    return k
AUTH = {"Authorization": "Bearer " + _api_key()}
import json, random, sys, time, urllib.request
n_words = int(sys.argv[1]); seed = int(sys.argv[2])
random.seed(seed)
words = [f"{random.choice(['alpha','beta','gamma','delta','omega','kappa'])}{random.randint(0,99999)}" for _ in range(n_words)]
prompt = "Here is a list of codes. Reply with only the FIRST code, nothing else.\n" + " ".join(words)
def ask(max_tokens=400):
    body = {"model": "DeepSeek V4.1 Flash", "stream": True, "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request("http://localhost:8001/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", **AUTH})
    t0 = time.time(); first = None; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=3000) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices") or []:
                piece = ch["delta"].get("content") or ch["delta"].get("reasoning") or ch["delta"].get("reasoning_content") or ""
                if piece and first is None: first = time.time() - t0
                if ch["delta"].get("content"): text += ch["delta"]["content"]
    return first, time.time() - t0, text.strip(), usage
def reset():
    try:
        r = urllib.request.urlopen(urllib.request.Request("http://localhost:8001/reset_prefix_cache", b"", method="POST", headers=AUTH), timeout=60)
        return r.status
    except Exception as e:
        return repr(e)[:100]
f1, t1, a1, u1 = ask(); print(f"cold:            ttft {f1:.2f}s total {t1:.1f}s prompt={u1['prompt_tokens']} answer={a1[:20]!r} (needle {'ok' if words[0] in a1 else 'MISSING'})", flush=True)
time.sleep(8)
f2, t2, a2, u2 = ask(); print(f"GPU-cache hit:   ttft {f2:.2f}s total {t2:.1f}s answer={a2[:20]!r}", flush=True)
print("reset_prefix_cache ->", reset(), flush=True); time.sleep(3)
f3, t3, a3, u3 = ask(); print(f"after reset:     ttft {f3:.2f}s total {t3:.1f}s answer={a3[:20]!r}  (identical to cold: {a3 == a1})", flush=True)
