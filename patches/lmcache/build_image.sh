#!/usr/bin/env bash
# Derive the LMCache-enabled serving image from the recipe image (docker/build.sh output). Installs lmcache with
# --no-deps so the recipe's torch/NCCL/vLLM stack is untouched; run with `env -u SHELLOPTS bash build_image.sh`.
set -euo pipefail
BASE=${BASE:-dsv41-flash-1017-exl3-3bpw-sm120}
TAG=${TAG:-$BASE-lmcache}
C=dsv41-lmc-build
docker rm -f "$C" >/dev/null 2>&1 || true
docker run --name "$C" --entrypoint bash "$BASE" -c 'pip install -q --no-deps lmcache==0.5.7 && python3 -c "import lmcache; print(\"lmcache ok\")"'
docker commit --change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD []' "$C" "$TAG" >/dev/null
docker rm "$C" >/dev/null
docker images "$TAG"
