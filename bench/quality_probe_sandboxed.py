#!/usr/bin/env python3
"""Quick quality probe for an OpenAI-compatible server: GSM8K subset + HumanEval pass@1.

SANDBOXED copy of recipe/bench/quality-probe.py (see sandbox_run below).

Greedy, thinking OFF by default (most sensitive to weight damage). Same prompts for every model.
Env: EFFORT (e.g. max: thinking on at that reasoning_effort), TEMP (0), TOP_P (unset), PORT (8000), SERVED_MODEL, API_KEY (required), EVALDATA (bench/evaldata; see fetch-evaldata.sh), N_GSM (200), CONC (4),
THINKING (0/1), MAX_TOKENS (4096), OUT (jsonl path), SKIP_HE=1 to skip HumanEval.
Usage: API_KEY=... PORT=8000 python3 quality-probe.py
"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = os.environ.get("EVALDATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaldata"))
PORT = os.environ.get("PORT", "8000")
MODEL = os.environ.get("SERVED_MODEL", "deepseek-v4.1-flash")
KEY = os.environ["API_KEY"]
N_GSM = int(os.environ.get("N_GSM", "200"))
CONC = int(os.environ.get("CONC", "4"))
THINK = os.environ.get("THINKING", "0") == "1"
MAXT = int(os.environ.get("MAX_TOKENS", "4096"))
EFFORT = os.environ.get("EFFORT")  # set => thinking on at this reasoning_effort (overrides THINKING)
TEMP = float(os.environ.get("TEMP", "0"))
TOP_P = os.environ.get("TOP_P")
OUT = os.environ.get("OUT", f"/tmp/quality-{MODEL.replace('/', '_')}-{int(time.time())}.jsonl")

# --- sandbox: model-written code never runs on the host. Each program runs in a throwaway container: no network, read-only
# root, non-root user, no capabilities, memory/pid/cpu caps. (The upstream probe used sys.executable on the host.)
SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "dsv41-flash-1017-exl3-3bpw-sm120")


def sandbox_run(path):
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:size=64m",
                        "--user", "65534:65534", "--memory", "1g", "--pids-limit", "128", "--cpus", "1",
                        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-v", f"{path}:/t/prog.py:ro",
                        "--entrypoint", "python3", SANDBOX_IMAGE, "-I", "/t/prog.py"], capture_output=True, timeout=60)
    if r.returncode in (125, 126, 127):  # docker itself failed: do not count it as a wrong answer
        raise RuntimeError("sandbox failed: " + r.stderr.decode()[:300])
    return r.returncode


def sandbox_selftest():
    import tempfile as _t
    with _t.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import socket\nassert 1 + 1 == 2\ntry:\n    socket.create_connection(('1.1.1.1', 53), 2)\n    raise SystemExit(3)\nexcept OSError:\n    pass\n")
    os.chmod(f.name, 0o644)
    rc = sandbox_run(f.name)
    os.unlink(f.name)
    assert rc == 0, f"sandbox self-test failed (rc={rc}): program must run and have NO network"


sandbox_selftest()


def chat(content):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "temperature": TEMP, "max_tokens": MAXT,
            "chat_template_kwargs": {"enable_thinking": THINK, "thinking": THINK}}
    if TOP_P:
        body["top_p"] = float(TOP_P)
    if EFFORT:
        body["chat_template_kwargs"] = {"enable_thinking": True, "thinking": True}
        body["reasoning_effort"] = EFFORT
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.load(r)
            m = d["choices"][0]["message"]
            return (m.get("content") or ""), d["choices"][0].get("finish_reason"), d.get("usage", {})
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2)
    return f"<<error {err}>>", "error", {}


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"[Ff]inal answer[:\s]*\**\s*(-?\d+(?:\.\d+)?)", s) or re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None


def gsm_one(r):
    c, fin, u = chat(r["question"] + "\n\nSolve step by step, then end with a line 'Final answer: <number>'.")
    p = num(c)
    ok = p is not None and abs(float(p) - float(r["answer"])) < 1e-6
    return {"task": "gsm8k", "ok": ok, "pred": p, "ref": r["answer"], "finish": fin, "out_tokens": u.get("completion_tokens")}


def he_one(r):
    c, fin, u = chat("Complete this Python function. Reply with the complete function (including imports and the "
                     "signature) in one ```python code block, no explanation.\n\n```python\n" + r["prompt"] + "```")
    blocks = re.findall(r"```(?:python)?\n(.*?)```", c, re.S)
    code = blocks[0] if blocks else c
    if f"def {r['entry_point']}" not in code:
        code = r["prompt"] + code
    prog = code + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    os.chmod(f.name, 0o644)
    try:
        ok = sandbox_run(f.name) == 0
    except subprocess.TimeoutExpired:
        ok = False
    os.unlink(f.name)
    return {"task": "humaneval", "id": r["task_id"], "ok": ok, "finish": fin, "out_tokens": u.get("completion_tokens")}


def run(fn, rows, name):
    t = time.time()
    with ThreadPoolExecutor(CONC) as ex:
        res = list(ex.map(fn, rows))
    acc = sum(r["ok"] for r in res) / len(res)
    trunc = sum(r["finish"] == "length" for r in res); errs = sum(r["finish"] == "error" for r in res)
    print(f"{MODEL} {name}: {sum(r['ok'] for r in res)}/{len(res)} = {acc:.1%}  (truncated {trunc}, errors {errs}, {time.time()-t:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    gsm = [json.loads(l) for l in open(f"{D}/gsm8k-test.jsonl")][:N_GSM]
    res = run(gsm_one, gsm, f"GSM8K first {len(gsm)}")
    if os.environ.get("SKIP_HE") != "1":
        he = [json.loads(l) for l in open(f"{D}/humaneval.jsonl")]
        res += run(he_one, he, f"HumanEval {len(he)}")
    with open(OUT, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print("wrote", OUT, "| thinking", THINK or bool(EFFORT), "| effort", EFFORT, "| temperature", TEMP, "| top_p", TOP_P)
