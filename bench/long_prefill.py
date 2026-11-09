"""One huge prefill (needle at the start) while sampling GPU1/GPU2 VRAM once a second.  usage: long_prefill.py <words> [seed]"""
import os as _os
def _api_key():
    k = _os.environ.get("API_KEY") or _os.environ.get("DSV41_API_KEY")
    if not k:
        raise SystemExit("set API_KEY to the key the server was started with (port 8001 requires it)")
    return k
AUTH = {"Authorization": "Bearer " + _api_key()}
import json, random, subprocess, sys, threading, time, urllib.request
words_n = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
random.seed(seed)
words = [f"{random.choice(['alpha','beta','gamma','delta','omega','kappa','sigma','theta'])}{random.randint(0,99999)}" for _ in range(words_n)]
prompt = "Here is a very long list of codes. Reply with only the FIRST code, nothing else.\n" + " ".join(words)
peak = {1: 0, 2: 0}; stop = False
def sample():
    while not stop:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits", "-i", "1,2"], capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            i, m = line.split(","); peak[int(i)] = max(peak[int(i)], int(m))
        time.sleep(1)
th = threading.Thread(target=sample, daemon=True); th.start()
body = {"model": "DeepSeek V4.1 Flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 600, "temperature": 0}
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8001/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", **AUTH}), timeout=3600))
    dt = time.time() - t0; pt = r["usage"]["prompt_tokens"]; ans = (r["choices"][0]["message"].get("content") or "").strip()
    print(f"{pt} prompt tokens in {dt:.1f}s = {pt/dt:.0f} tok/s; needle {'FOUND' if words[0] in ans else 'MISSING'} ({ans[:30]!r}); peak VRAM MiB gpu1={peak[1]} gpu2={peak[2]} (of 97887)")
except Exception as e:
    print("REQUEST FAILED:", repr(e)[:300], f"after {time.time()-t0:.0f}s; peak VRAM MiB gpu1={peak[1]} gpu2={peak[2]}")
stop = True
