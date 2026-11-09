#!/usr/bin/env bash
# Entry point for the dsv41 container when LMCache is enabled: start the LMCache multiprocess server
# beside vLLM (they share GPU memory through CUDA IPC, so it must live in the same container), wait
# for its port, then exec vLLM with the compose `command` arguments.
#   DSV41_LMCACHE_L1_GB   pinned host-RAM tier (fallback 16 if unset; docker-compose.yml sets 48)        DSV41_LMCACHE_L2_GB   disk tier (default 128)
#   DSV41_LMCACHE_DIR     disk tier directory (default /root/.cache/lmcache = ~/.cache/dsv41-flash-exl3/lmcache)
set -u
export LMCACHE_DISABLE_BANNER=1
L1=${DSV41_LMCACHE_L1_GB:-16}
L2=${DSV41_LMCACHE_L2_GB:-128}
DIR=${DSV41_LMCACHE_DIR:-/root/.cache/lmcache}
mkdir -p "$DIR"
lmcache server --host 127.0.0.1 --port 5555 --chunk-size 256 --separate-object-groups \
  --max-workers "${DSV41_LMCACHE_WORKERS:-8}" --l1-size-gb "$L1" --l1-use-lazy --l1-init-size-gb 2 \
  --eviction-policy LRU \
  --l2-adapter "{\"type\":\"fs_native\",\"base_path\":\"$DIR\",\"num_workers\":16,\"use_odirect\":false,\"max_capacity_gb\":$L2,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.2}}" \
  > /root/.cache/lmcache-server.log 2>&1 &
for _ in $(seq 1 120); do
  python3 -c "import socket; socket.create_connection(('127.0.0.1', 5555), 2)" 2>/dev/null && break
  sleep 1
done
exec vllm "$@"
