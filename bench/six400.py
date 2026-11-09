"""Six concurrent ~400k-token requests (the compaction limit), needle at the start of each; run twice (2nd pass = same prompts,
shows GPU-prefix/LMCache reuse).  usage: six400.py [seed_base] [words_each]"""
import os as _os
def _api_key():
    k = _os.environ.get("API_KEY") or _os.environ.get("DSV41_API_KEY")
    if not k:
        raise SystemExit("set API_KEY to the key the server was started with (port 8001 requires it)")
    return k
AUTH = {"Authorization": "Bearer " + _api_key()}
import json, random, subprocess, sys, threading, time, urllib.request
base = int(sys.argv[1]) if len(sys.argv) > 1 else 700
words_each = int(sys.argv[2]) if len(sys.argv) > 2 else 128000
peak = {1: 0, 2: 0}; stop = False
def sample():
    while not stop:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits", "-i", "1,2"], capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            i, m = line.split(","); peak[int(i)] = max(peak[int(i)], int(m))
        time.sleep(0.5)
threading.Thread(target=sample, daemon=True).start()
prompts = []
for k in range(6):
    random.seed(base + k)
    words = [f"{random.choice(['alpha','beta','gamma','delta','omega','kappa'])}{random.randint(0,99999)}" for _ in range(words_each)]
    prompts.append((words[0], "Here is a very long list of codes. Reply with only the FIRST code, nothing else.\n" + " ".join(words)))
def run_pass(label):
    res = {}
    def req(k):
        needle, prompt = prompts[k]
        body = {"model": "DeepSeek V4.1 Flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 600, "temperature": 0}
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8001/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", **AUTH}), timeout=7200))
            ans = (r["choices"][0]["message"].get("content") or "")
            res[k] = (round(time.time() - t0, 1), r["usage"]["prompt_tokens"], needle in ans)
        except Exception as e:
            res[k] = ("FAILED", repr(e)[:100])
    ths = [threading.Thread(target=req, args=(k,)) for k in range(6)]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]
    pt = sum(v[1] for v in res.values() if v[0] != "FAILED")
    print(f"{label}: wall {time.time()-t0:.0f}s, {pt} prompt tokens total; peak VRAM MiB gpu1={peak[1]} gpu2={peak[2]} (of 97887)")
    for k in sorted(res): print("  ", k, res[k], flush=True)
run_pass("pass 1 (cold)")
run_pass("pass 2 (same prompts again)")
stop = True
