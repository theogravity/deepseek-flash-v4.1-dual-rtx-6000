"""Shows that the caches are PREFIX caches: time to first token for (A) a cold prompt, (B) the same prompt with text appended,
(C) one word changed in the middle, (D) one word changed at the very start.   usage: prefix_demo.py [words=19000] [seed]
Use a new seed each time (cached blocks and the OS page cache make repeats of A look different)."""
import json, os, random, sys, time, urllib.request
key = os.environ.get("API_KEY") or os.environ.get("DSV41_API_KEY")
if not key:
    raise SystemExit("set API_KEY to the key the server was started with (port 8001 requires it)")
n_words = int(sys.argv[1]) if len(sys.argv) > 1 else 19000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else random.randrange(10**6)
random.seed(seed)
words = [f"{random.choice(['alpha','beta','gamma','delta','omega','kappa'])}{random.randint(0,99999)}" for _ in range(n_words)]
def ttft(text):
    body = {"model": "DeepSeek V4.1 Flash", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 20, "temperature": 0,
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request("http://localhost:8001/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:]); usage = d.get("usage") or usage
            for ch in d.get("choices") or []:
                if first is None and (ch["delta"].get("content") or ch["delta"].get("reasoning") or ch["delta"].get("reasoning_content")):
                    first = time.time() - t0
    return first, usage["prompt_tokens"]
head = "Here is a list of codes. Reply with only the FIRST code, nothing else.\n"
mid = list(words); mid[len(mid) // 2] = "CHANGEDWORD"
start = list(words); start[0] = "CHANGEDWORD"
cases = [("A  cold prompt", head + " ".join(words)),
         ("B  same prompt + a question appended", head + " ".join(words) + "\n\nAlso, what is the last code?"),
         ("C  one word changed at 50% of the prompt", head + " ".join(mid)),
         ("D  one word changed at the very start", head + " ".join(start))]
print(f"seed {seed}, {n_words} words")
for name, text in cases:
    f, n = ttft(text)
    print(f"{name:44s} {n:7d} tokens  TTFT {f:6.2f} s")
