"""How much does one huge cold prefill slow other sessions' decode?  3 decode streams alone, then the same 3 while a ~300k-token
cold prefill runs.   usage: interference.py [seed=77]"""
import json, os, random, sys, threading, time, urllib.request
def _api_key():
    k = _os.environ.get("API_KEY") or _os.environ.get("DSV41_API_KEY")
    if not k:
        raise SystemExit("set API_KEY to the key the server was started with (port 8001 requires it)")
    return k
AUTH = {"Authorization": "Bearer " + _api_key()}
URL = "http://localhost:8001/v1/chat/completions"; seed = int(sys.argv[1]) if len(sys.argv) > 1 else 77
funcs = "\n".join(f"def func_{i}(x, y):\n    result = x * {i} + y\n    if result > {i * 3}:\n        return result - {i}\n    return result\n" for i in range(300))
def decode_stream(k, res):
    body = {"model": "DeepSeek V4.1 Flash", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 1500, "temperature": 0,
            "messages": [{"role": "user", "content": f"Python module:\n\n{funcs}\n\nRewrite func_{20+40*k} through func_{60+40*k} with type hints and docstrings. Output only code."}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", **AUTH})
    first = None; usage = None
    with urllib.request.urlopen(req, timeout=3000) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:]); usage = d.get("usage") or usage
            ch = d.get("choices") or []
            if ch and (ch[0]["delta"].get("content") or ch[0]["delta"].get("reasoning") or ch[0]["delta"].get("reasoning_content")) and first is None: first = time.time()
    res[k] = (usage or {}).get("completion_tokens", 0) / (time.time() - first)
def big_prefill():
    random.seed(seed); words = [f"{random.choice(['alpha','beta','gamma'])}{random.randint(0,99999)}" for _ in range(95000)]
    body = {"model": "DeepSeek V4.1 Flash", "max_tokens": 50, "temperature": 0, "messages": [{"role": "user", "content": "First code only.\n" + " ".join(words)}]}
    t0 = time.time(); urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", **AUTH}), timeout=3000).read()
    print(f"  (the ~295k-token cold prefill took {time.time()-t0:.0f}s)", flush=True)
def run(label, with_prefill):
    res = {}; ths = [threading.Thread(target=decode_stream, args=(k, res)) for k in range(3)]
    bp = threading.Thread(target=big_prefill) if with_prefill else None
    if bp: bp.start(); time.sleep(5)
    [t.start() for t in ths]; [t.join() for t in ths]
    if bp: bp.join()
    print(f"{label}: per-stream decode tok/s {[round(res[k]) for k in sorted(res)]}", flush=True)
for k in range(2): decode_stream(k, {})  # warm
run("3 decode streams alone          ", False)
run("3 decode streams + 295k prefill ", True)
