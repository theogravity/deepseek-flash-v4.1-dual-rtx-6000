#!/usr/bin/env bash
# Max-effort benchmark: thinking on, reasoning_effort=max, temperature 1.0, top_p 0.95 (the model card's settings).
# Part 1: code-edit decode at 1/4/6 streams. Part 2: GSM8K-200 + HumanEval-164 (sandboxed) at 6 concurrent requests.
# Speculative-decoding acceptance is read from the server's counters around each part. Run: env -u SHELLOPTS bash maxeffort.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export API_KEY=${API_KEY:-$(grep '^LITELLM_MASTER_KEY=' ~/llm/.env | cut -d= -f2-)}
export PORT=${PORT:-8001} SERVED_MODEL="${SERVED_MODEL:-DeepSeek V4.1 Flash}" EVALDATA=${EVALDATA:-$HOME/dsv41-evaldata}
export EFFORT=max TEMP=1.0 TOP_P=0.95
acc() { curl -s localhost:$PORT/metrics | awk '/^vllm:spec_decode_num_(draft_tokens|accepted_tokens)_total/{split($1,a,"{"); v[a[1]]=$2} END{print v["vllm:spec_decode_num_draft_tokens_total"], v["vllm:spec_decode_num_accepted_tokens_total"]}'; }
rate() { awk -v a="$1" -v b="$2" 'BEGIN{split(a,x," ");split(b,y," ");d=y[1]-x[1];printf "draft tokens %d, accepted %d = %.1f%%\n", d, y[2]-x[2], d?100*(y[2]-x[2])/d:0}'; }
cd /tmp
echo "=== decode (effort max, temp 1.0, top_p 0.95, max_tokens 6000)"; a0=$(acc)
MAX_TOKENS=6000 python3 -I "$HERE/decode_effort.py" 2>&1 | tail -8
echo "spec-decode acceptance: $(rate "$a0" "$(acc)")"
echo "=== quality (effort max, temp 1.0, top_p 0.95, 6 concurrent, max_tokens 32000)"; a0=$(acc)
CONC=6 MAX_TOKENS=32000 python3 -I "$HERE/quality_probe_sandboxed.py" 2>&1 | grep -E "GSM8K|HumanEval|Error|sandbox|wrote"
echo "spec-decode acceptance: $(rate "$a0" "$(acc)")"
