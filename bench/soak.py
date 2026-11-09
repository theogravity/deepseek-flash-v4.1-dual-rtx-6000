"""Soak test: 6 workers hammer the server for N minutes with a mix of request sizes (2k / 30k / 200k / 400k tokens, greedy
needle or code-edit prompts, repeated prefixes so LMCache and the GPU prefix cache churn). Samples VRAM / host RAM / LMCache
disk size every 30 s and reports errors, latency, drift and container health.   usage: soak.py [minutes=20] [seed=1000]"""
import json, os, random, subprocess, sys, threading, time, urllib.request
def _api_key():
    k = _os.environ.get("API_KEY") or _os.environ.get("DSV41_API_KEY")
    if not k:
        raise SystemExit("set API_KEY to the key the server was started with (port 8001 requires it)")
    return k
AUTH = {"Authorization": "Bearer " + _api_key()}
minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 20
seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
URL = "http://localhost:8001/v1/chat/completions"
deadline = time.time() + minutes * 60
SIZES = [(2000, 0.35), (30000, 0.3), (130000, 0.2), (128000 * 1, 0.1), (140000, 0.05)]  # (words ~ 3.1 tokens each)
lock = threading.Lock(); stats = {"ok": 0, "bad_answer": 0, "errors": 0}; lats = []; errs = []
def prompt_for(rng):
    words_n = rng.choices([s for s, _ in SIZES], [w for _, w in SIZES])[0]
    s = rng.randrange(40)                      # only 40 distinct prompts per size -> repeats hit GPU/LMCache caches
    r = random.Random(seed0 * 1000 + words_n + s)
    words = [f"{r.choice(['alpha','beta','gamma','delta','omega'])}{r.randint(0,99999)}" for _ in range(words_n)]
    return words[0], "Here is a list of codes. Reply with only the FIRST code, nothing else.\n" + " ".join(words), words_n
def worker(i):
    rng = random.Random(seed0 + i)
    while time.time() < deadline:
        needle, prompt, n = prompt_for(rng)
        body = {"model": "DeepSeek V4.1 Flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 800, "temperature": 0}
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", **AUTH}), timeout=3600))
            ans = r["choices"][0]["message"].get("content") or ""
            with lock:
                stats["ok" if needle in ans else "bad_answer"] += 1; lats.append((n, time.time() - t0))
        except Exception as e:
            with lock: stats["errors"] += 1; errs.append(repr(e)[:150])
samples = []
def sampler():
    while time.time() < deadline + 5:
        v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "1,2"], capture_output=True, text=True).stdout.split()
        avail = int(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()[1].split()[6])
        disk = int(subprocess.run(["du", "-sm", os.path.expanduser("~/.cache/dsv41-flash-exl3/lmcache")], capture_output=True, text=True).stdout.split()[0])
        samples.append((time.time(), max(map(int, v)), avail, disk)); time.sleep(30)
th = [threading.Thread(target=worker, args=(i,)) for i in range(6)] + [threading.Thread(target=sampler)]
t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
print(f"soak {minutes:.0f} min, 6 workers: {stats}")
if errs: print("first errors:", errs[:3])
by = {}
for n, l in lats: by.setdefault(n, []).append(l)
for n in sorted(by): print(f"  ~{n*3//1000}k-token prompts: {len(by[n])} done, median {sorted(by[n])[len(by[n])//2]:.1f}s, max {max(by[n]):.1f}s")
if samples:
    print(f"VRAM max {max(s[1] for s in samples)} MiB (first {samples[0][1]}, last {samples[-1][1]}); host RAM available min {min(s[2] for s in samples)//1024} GiB (first {samples[0][2]//1024}, last {samples[-1][2]//1024}); LMCache disk {samples[0][3]//1024} -> {samples[-1][3]//1024} GiB")
