"""6 concurrent requests: one ~1M-token prefill + five ~100k-token requests, with VRAM sampling.  usage: mixed.py [seed_base]"""
import os as _os
def _api_key():
    k = _os.environ.get("API_KEY") or _os.environ.get("DSV41_API_KEY")
    if not k:
        raise SystemExit("set API_KEY to the key the server was started with (port 8001 requires it)")
    return k
AUTH = {"Authorization": "Bearer " + _api_key()}
import json, random, subprocess, sys, threading, time, urllib.request
base = int(sys.argv[1]) if len(sys.argv) > 1 else 40
peak = {1: 0, 2: 0}; stop = False
def sample():
    while not stop:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits", "-i", "1,2"], capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            i, m = line.split(","); peak[int(i)] = max(peak[int(i)], int(m))
        time.sleep(0.5)
threading.Thread(target=sample, daemon=True).start()
res = {}
def req(k, n_words):
    random.seed(base + k)
    words = [f"{random.choice(['alpha','beta','gamma','delta','omega'])}{random.randint(0,99999)}" for _ in range(n_words)]
    prompt = "Here is a list of codes. Reply with only the FIRST code, then write 200 words about oceans.\n" + " ".join(words)
    body = {"model": "DeepSeek V4.1 Flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 700, "temperature": 0}
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8001/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", **AUTH}), timeout=3600))
        ans = (r["choices"][0]["message"].get("content") or "")
        res[k] = (round(time.time() - t0, 1), r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"], words[0] in ans)
    except Exception as e:
        res[k] = ("FAILED", repr(e)[:120])
sizes = [320000] + [32000] * 5
ths = [threading.Thread(target=req, args=(k, n)) for k, n in enumerate(sizes)]
t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; stop = True
print(f"wall {time.time()-t0:.0f}s; peak VRAM MiB gpu1={peak[1]} gpu2={peak[2]} (of 97887)")
for k in sorted(res): print(k, res[k])
