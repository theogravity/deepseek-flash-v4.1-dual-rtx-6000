# Install and build guide

Step by step, from a bare Linux box to a running, API-key-protected server on `localhost:8001`. Budget about **an hour or more of mostly unattended time**
(the 322 GiB download dominates; ~25 min at 250 MB/s) and **~500 GiB of disk**. The shorter version is the Quick start in the [README](../README.md); the reasoning behind the
settings is in [TUNING.md](TUNING.md).

Tested on: Ubuntu 26.04 (kernel 7.0), NVIDIA driver 615.71.09 (CUDA 13), Docker 29.9, Docker Compose v5.7, NVIDIA Container Toolkit 1.20.1, `hf` CLI 1.18.

## 1. Prerequisites (check each one)

| Need | Check | Expect |
|---|---|---|
| Two RTX PRO 6000 Blackwell Workstation cards (sm_120, 97,887 MiB each) | `nvidia-smi --query-gpu=index,name,memory.total --format=csv` | two rows, `97887 MiB`. Note their **indices**; you will put them in `GPU_A` / `GPU_B`. |
| Peer-to-peer between the two cards | `nvidia-smi topo -p2p r` | `OK` between the two RTX PRO 6000s (any other GPU may show `NS`; that is fine) |
| A CUDA 13-capable driver | `nvidia-smi` header | `CUDA Version: 13.x` (we ran 615.71.09) |
| Docker + Compose v2 + NVIDIA Container Toolkit | `docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L` | lists your GPUs; your user can run `docker` without sudo |
| `hf` CLI | `hf --version` | any recent version (`pip install -U "huggingface_hub[cli]"`) |
| Host RAM ≈ 246 GiB | `free -g` | `total` ≥ ~240. The container uses ~90 GiB plus the LMCache RAM tier (48 GiB by default); keep ≥48 GiB free for the OS. With less RAM, lower `DSV41_LMCACHE_L1_GB` in the compose file. |
| AVX-512 CPU (for the optional CPU host-expert runtime) | `grep -o -E 'avx512(f\|bw\|vnni\|vbmi\|vbmi2)' /proc/cpuinfo \| sort -u` | all five. Without them set `EXL3_CPU_EXPERTS: "0"` (decode ~12–17% slower). |
| Disk | `df -h` for the model dir, Docker's data dir and the cache dir | model **322 GiB**; Docker **~36 GB** (the recipe image is built on the pulled base image and shares its layers; the LMCache image adds ~0.2 GB; plus some build cache); caches up to **128 GiB** (the LMCache disk tier's cap; a few hundred MB of kernel caches). Put the model and cache directories on fast NVMe. |
| Network | — | GitHub and PyPI/apt (the recipe's image build clones three GitHub repos and installs packages), Docker Hub (base image), Hugging Face (model). The kernel and CPU-runtime builds run offline. |

## 2. Download the model and the recipe (~322 GiB; ~25 min at 250 MB/s)

```bash
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir /mnt/models/dsv41-flash-exl3
```

The directory holds the checkpoint (48 safetensors shards) **and** the recipe (`recipe/`: Dockerfile-less build scripts, kernels, patches, plans).
Check: `ls /mnt/models/dsv41-flash-exl3/model-000*-of-00048.safetensors | wc -l` → `48`, `du -sh` ≈ 322 GiB, and `ls /mnt/models/dsv41-flash-exl3/recipe`
shows `docker/ kernels/ cpuexp/ patches/ plans/ serve/ bench/`. The download resumes if interrupted; if throughput drops to ~0 for minutes, stop it and re-run the same command.

Use the path `/mnt/models/dsv41-flash-exl3` or set `MODEL_DIR` in step 7.

## 3. Build the recipe image (~10 min)

Run the recipe's scripts **by path from a directory that is not the download** (interpreters load code from the current directory):

```bash
cd ~
R=/mnt/models/dsv41-flash-exl3/recipe
BUILD_MEM=24g BUILD_CPUS=12 MAX_JOBS=6 bash $R/docker/build.sh
```

The script pulls the pinned vLLM nightly, applies the recipe's patch series, builds ExLlamaV3 for sm_120 and commits the image
`dsv41-flash-1017-exl3-3bpw-sm120`. `BUILD_MEM` defaults to 64g; lower values are fine if RAM is tight but too low gets the build container
OOM-killed (raise it). The build log should contain `vllm patches ok`, `vllm_exl3 ok` and `exllamav3 ok`, and end with `docker images` listing the new image (~35 GB).

If the script prints nothing and exits 0, check `echo $SHELLOPTS`: an exported `onecmd` option makes every child `bash` stop after its first line; run `env -u SHELLOPTS bash $R/docker/build.sh`.

## 4. Build the kernels and the CPU runtime (~3 min, no GPU needed)

```bash
MAX_JOBS=6 bash $R/kernels/build.sh   # expect: built xmoe, xmoe_do, xmoe_do_i8, fp8ar_ext, rowtopk
bash $R/cpuexp/build.sh               # expect libcpuexp_rt.so, libcpuexp_gpu.so, bwprobe in $R/cpuexp/build/
```

Both build inside the image, offline. The CPU runtime is compiled with `-march=native`: **build it on the machine that will run it**.
Then check the host (stop any running server first, the probe loads the cores for a few seconds):

```bash
bash $R/cpuexp/check_host.sh --bandwidth   # expect "OK"; we measured 251.8 GB/s, the recipe's minimum is 180
```

On a different CPU the pools are derived from its L3 layout (ours: 9965WX → `0-4,6-10;12-16,18-22`). If the check fails, set `EXL3_CPU_EXPERTS: "0"`.

## 5. Build the LMCache image (~1 min)

```bash
bash patches/lmcache/build_image.sh   # from this repo; creates dsv41-flash-1017-exl3-3bpw-sm120-lmcache
```

It only adds `lmcache==0.5.7` (with `--no-deps`, so the recipe's torch/NCCL/vLLM stay untouched).

## 6. Check the patches match your image

The mounted patches were generated against the image the recipe built on 2026-11-08. If your build used a newer recipe commit, regenerate them:
[TUNING.md §2](TUNING.md#2-rebuild-from-scratch) has the three `docker run ... cat` commands and the two `apply_patch.py` invocations. Each patcher
refuses to run if the code it patches has moved, so a stale patch cannot silently apply. Optional but recommended: run the model-free test
in [`patches/engram-nvme/README.md`](../patches/engram-nvme/README.md) (needs one GPU, no model load).

## 7. Configure and start

```bash
cp compose/.env.example compose/.env
$EDITOR compose/.env          # API_KEY (required); MODEL_DIR, CACHE_DIR, GPU_A, GPU_B if they differ from the defaults
(cd compose && docker compose up -d)
docker logs -f dsv41-flash-exl3
```

`GPU_A` / `GPU_B` are the host indices from step 1 (default 0 and 1). If your cards have a different amount of VRAM, `--kv-cache-memory` in the compose
file must be re-derived (TUNING.md §3): it was sized so a 500k-token prefill peaks ~2 GiB under the limit.

**What a healthy boot looks like** (about 5 minutes; the very first boot ~8 because kernels compile and nothing is cached):

1. `Engram table served from NVMe (layer 1 ...)` four times (two layers × two ranks), then `Model loading took ~77 GiB ... seconds`.
2. `EXL3 CPU host experts ACTIVE for decode-sized calls` (if the CPU runtime is on).
3. `GPU KV cache size: 2,769,546 tokens, Maximum concurrency for 524,288 tokens per request: 5.28x`.
4. The LMCache server log (`$CACHE_DIR/lmcache-server.log` on the host, `/root/.cache/lmcache-server.log` in the container) shows `Registered KV cache for GPU ID ... with 54 layers`.
5. `docker ps` reports `(healthy)`.

## 8. Verify

```bash
export API_KEY=<the key from compose/.env>
code() { curl -s -o /dev/null -w '%{http_code}\n' "$@"; }
code localhost:8001/v1/models                                         # 401  (no key)
code -H "Authorization: Bearer $API_KEY" localhost:8001/v1/models     # 200
code -H "x-api-key: $API_KEY" localhost:8001/v1/models                # 200  (the auth patch)
code localhost:8001/health                                            # 200  (open, used by the healthcheck)
curl -s -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' localhost:8001/v1/chat/completions \
  -d '{"model":"DeepSeek V4.1 Flash","max_tokens":400,"messages":[{"role":"user","content":"What is 17*23? Just the number."}]}'
bash bench/run_quality.sh      # the recipe's suite: expect sweep 11/11, tools OK, image OK, needles found, GSM8K ~98.5 %, HumanEval ~96 %
```

The first requests after boot are slow while the remaining kernels compile. Benchmarks and what to expect: [BENCHMARKS.md](BENCHMARKS.md).

## 9. Day-to-day

```bash
(cd compose && docker compose down)          # stop (allow up to 4 minutes; stop_grace_period is 240 s)
(cd compose && docker compose up -d)         # start again: ~5 min, kernel caches and the LMCache disk tier persist in CACHE_DIR
docker logs --since 10m dsv41-flash-exl3     # recent logs (use --since: a restarted container keeps older runs' logs)
```

Removing everything: `docker compose down`, `docker rmi dsv41-flash-1017-exl3-3bpw-sm120-lmcache dsv41-flash-1017-exl3-3bpw-sm120`, then delete the model
directory and `CACHE_DIR`.

## 10. If a step fails

| Where | Symptom | Fix |
|---|---|---|
| any script | prints nothing, exit 0 | `env -u SHELLOPTS bash script.sh` (see step 3) |
| step 3 | build container killed / `Killed` | raise `BUILD_MEM`; free RAM |
| step 4 | `check_host.sh` exits 4 (bandwidth) or 1 (no AVX-512) | set `EXL3_CPU_EXPERTS: "0"` in the compose file |
| step 7 | `required variable API_KEY is missing a value` | create `compose/.env` (step 7) |
| step 7 | container dies ~1 min into loading, `journalctl -k` shows `Out of memory: Killed process ... VLLM::Worker_TP` | the Engram patch is not applied (look for `Engram table served from NVMe` in the log) or the host has too little RAM; check the three patch mounts exist and match your image |
| step 7 | `CUDA error: invalid argument` / `pidfd_getfd: Operation not permitted` / `incompatible with PYTORCH_CUDA_ALLOC_CONF` | LMCache requirements: `SYS_PTRACE` + `seccomp=unconfined`, and no `expandable_segments` (the compose file already does this; do not add it back) |
| step 8 | clients using `x-api-key` get 401 | the `patches/api-auth` mount is missing or stale |

More symptoms and causes: [TUNING.md §5](TUNING.md#5-troubleshooting-every-one-of-these-was-hit).
