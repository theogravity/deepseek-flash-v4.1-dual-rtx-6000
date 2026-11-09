"""Why Claude Code sessions can miss the prefix cache, and the fix, reproduced without Claude Code.

Claude Code puts `role: "system"` entries INSIDE `messages` (hook context, a per-request <total_tokens> counter). On vLLM's Anthropic endpoint (/v1/messages,
what LiteLLM's anthropic/ provider calls) those change the rendered start of the prompt on every request, so a growing conversation stops reusing its prefix.
Each case sends request 1, then request 2 = request 1 + one assistant turn + one new user message, and reads vLLM's own counters for the second request.
  A  plain messages (no inline system)             -> the cache works
  B  inline system messages, as Claude Code sends  -> reuse drops to the part of the prompt before the first inline message
  C  the same content folded into user <system-reminder> text (what examples/litellm_fold_inline_system.py does) -> the cache works again
usage: API_KEY=<key> python3 -I inline_system_test.py [host:port=localhost:8001]"""
import copy, json, os, random, sys, uuid, urllib.request
KEY = os.environ.get("API_KEY") or os.environ.get("DSV41_API_KEY")
if not KEY: raise SystemExit("set API_KEY to the key the server was started with")
HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost:8001"
MODEL = "DeepSeek V4.1 Flash"
def metrics():
    t = urllib.request.urlopen(f"http://{HOST}/metrics", timeout=30).read().decode()   # /metrics is not key-protected
    g = lambda n, l=None: sum(float(x.rsplit(" ", 1)[1]) for x in t.splitlines() if x.startswith(n + "{") and (l is None or l in x))
    return g("vllm:prompt_tokens_total"), g("vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"')
def send(body):
    b = copy.deepcopy(body); b["max_tokens"] = 16
    r = urllib.request.Request(f"http://{HOST}/v1/messages", json.dumps(b).encode(), {"Content-Type": "application/json", "x-api-key": KEY, "anthropic-version": "2023-06-01"})
    p0, h0 = metrics(); urllib.request.urlopen(r, timeout=600).read(); p1, h1 = metrics(); return p1 - p0, h1 - h0
random.seed(9)
words = lambda n: " ".join(f"{random.choice(['alpha','beta','gamma'])}{random.randint(0,99999)}" for _ in range(n))
TOOLS = [{"name": f"tool_{i}", "description": "A tool. " * 10, "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}} for i in range(8)]
HOOK = words(1500)                                   # ~4.5k tokens of hook context (Claude Code sends a few thousand tokens of it)
RESULT = words(5000)                                 # ~15k tokens: a big tool result, the bulk of a real conversation
U = lambda t: {"role": "user", "content": t}
A = lambda t: {"role": "assistant", "content": t}
S = lambda t: {"role": "system", "content": t}
TU = {"role": "assistant", "content": [{"type": "text", "text": "I'll read it."}, {"type": "tool_use", "id": "toolu_1", "name": "tool_0", "input": {"x": "file"}}]}
TR = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": RESULT}]}
def fold(msgs):                                      # what the LiteLLM callback does: system -> <system-reminder> text merged into the adjacent user turn
    out = []
    for m in msgs:
        if m["role"] == "system":
            m = {"role": "user", "content": [{"type": "text", "text": f"<system-reminder>\n{m['content']}\n</system-reminder>"}]}
        if out and m["role"] == "user" and out[-1]["role"] == "user":
            prev = out[-1]["content"]; prev = [{"type": "text", "text": prev}] if isinstance(prev, str) else prev
            cur = m["content"]; cur = [{"type": "text", "text": cur}] if isinstance(cur, str) else cur
            out[-1] = {"role": "user", "content": prev + cur}
        else: out.append(m)
    return out
def body(msgs, nonce):
    return {"model": MODEL, "max_tokens": 16, "tools": TOOLS, "system": f"[nonce {nonce}] You are a helpful agent.", "messages": msgs}
# Claude Code's shape: user prompt, hook context as a system message, tool call + big result, then a per-request <total_tokens> system message
inline1 = [U("Do the task."), S(HOOK), TU, TR, S("<total_tokens>15000000 tokens left</total_tokens>")]
inline2 = inline1 + [A("Done."), U("Now the next thing."), S("<total_tokens>14999911 tokens left</total_tokens>")]
cases = [("A  no inline system messages", [U("Do the task."), TU, TR], [U("Do the task."), TU, TR, A("Done."), U("Now the next thing.")]),
         ("B  inline system messages (Claude Code)", inline1, inline2),
         ("C  inline system folded into user text", fold(inline1), fold(inline2))]
print(f"request 2 = request 1 + an assistant turn + a new user message, against {HOST}; 'possible' = share of request 2 that extends request 1")
for name, m1, m2 in cases:
    n = uuid.uuid4().hex[:8]
    pa, _ = send(body(m1, n)); pc, hc = send(body(m2, n))
    print(f"  {name:42s} request 2: {pc:6.0f} tokens, reused {hc:6.0f} ({100*hc/pc:5.1f}%) of a possible {100*pa/pc:5.1f}%")
