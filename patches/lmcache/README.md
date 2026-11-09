# LMCache for DeepSeek V4.1 Flash

- `build_image.sh` — derives `dsv41-flash-1017-exl3-3bpw-sm120-lmcache` from the recipe image (`pip install --no-deps lmcache==0.5.7`
  so the recipe's torch/NCCL/vLLM stay untouched). Run with `env -u SHELLOPTS bash build_image.sh`.
- `start.sh` — container entrypoint: starts `lmcache server` (127.0.0.1:5555) beside vLLM, waits for the port, then `exec vllm "$@"`.
  Knobs via compose env: `DSV41_LMCACHE_L1_GB` (RAM tier, 48), `DSV41_LMCACHE_L2_GB` (disk tier, 128), `DSV41_LMCACHE_DIR`
  (default `/root/.cache/lmcache` = `~/.cache/dsv41-flash-exl3/lmcache`), `DSV41_LMCACHE_WORKERS`. The disk adapter carries an
  LRU `eviction` block; without it `max_capacity_gb` is not enforced.

The compose service must also: pass `--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'` to vLLM, leave
`PYTORCH_CUDA_ALLOC_CONF` unset (CUDA IPC does not work on expandable-segment/cumem memory; costs ~1.5 GiB), and grant `SYS_PTRACE`
+ `seccomp=unconfined` (`pidfd_getfd`). The server must live in the same container as vLLM (CUDA IPC).

Verify a restore: send a long prompt, push another ~2M tokens through (or restart; the disk tier persists), resend it, and look for
a low TTFT, `External prefix cache hit rate` > 0 in vLLM's stats and `Retrieved N tokens` in `/root/.cache/lmcache-server.log`.
(vLLM's `/reset_prefix_cache` did not evict in our tests.) Details and measured numbers: `docs/TUNING.md`.
