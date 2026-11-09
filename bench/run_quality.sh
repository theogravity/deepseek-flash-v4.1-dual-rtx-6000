#!/usr/bin/env bash
# Quality/validation suite from the recipe (HTTP clients only) against the running dsv41 server on :8001.
# Run as: API_KEY=<key> bash run_quality.sh      (needs the key the server was started with).
# GSM8K/HumanEval data is fetched once into $EVALDATA (outside the pack). HumanEval programs run in a throwaway
# no-network container (quality_probe_sandboxed.py), never on the host. Recipe reference (2x, GPU-only / CPU runtime):
# GSM8K-200 98.5 / 99.0 %, HumanEval-164 96.3 / 96.3 %, short sweep 11/11, tool calls OK, image 4/4, needles found.
set -u
R=${RECIPE:-/mnt/models/dsv41-flash-exl3/recipe}; B=$R/bench; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export API_KEY=${API_KEY:?set API_KEY to the key the server was started with}
export PORT=${PORT:-8001} SERVED_MODEL="${SERVED_MODEL:-DeepSeek V4.1 Flash}" EVALDATA=${EVALDATA:-$HOME/dsv41-evaldata}
cd /tmp
[ -f "$EVALDATA/gsm8k-test.jsonl" ] || bash "$B/fetch-evaldata.sh"
echo "--- short-context sweep:";  python3 -I "$B/len_sweep.py" 2>&1 | tail -1
echo "--- tool calls:";           python3 -I "$B/test-toolcall.py" 2>&1 | tail -2
echo "--- image input:";          python3 -I "$B/test-image.py" 2>&1 | tail -5
for r in 6300 15600 28000; do echo "--- needle, $r repeats (~$((r*16/1000))k tokens):"; python3 -I "$B/needle-test.py" $r 2>&1 | tail -2; done
echo "--- GSM8K-200 + HumanEval-164 (greedy, thinking off, sandboxed):"
python3 -I "$HERE/quality_probe_sandboxed.py" 2>&1 | grep -E "GSM8K|HumanEval|Error|sandbox"
