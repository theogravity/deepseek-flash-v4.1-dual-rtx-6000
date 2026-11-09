# NVMe-backed Engram tables for DeepSeek V4.1 Flash (EXL3)

Why: the stock recipe pins the two int4 Engram tables (103 GiB, checkpoint shards 47/48) in host RAM; with the rest of the
recipe that is ~230 GiB of this host's 246 GiB and the load is host-OOM-killed. Enabled by `DSV41_ENGRAM_NVME=1` in the compose
template; with the variable unset the patched files behave like the stock ones.

What it does: the loader skips the two tables (`ep_weight_filter.py`), and `ParallelEngramEmbedding.lookup` (`engram.py`) serves
rows straight from the mmapped shards: D2H ids -> pinned buffer, a CUDA host-function node gathers the needed rows (32 threads,
`posix_fadvise` hints for small batches) into pinned staging, H2D, then the *stock* int4 Triton kernel runs on that compact table.
Everything is shape/argument-static per token count, so it is captured inside the FULL decode CUDA graph. Output is bit-exact
with the stock kernel (checked by the test).

Files: `apply_patch.py` (generates `patched/*` from the image's stock files; fails if an anchor moved), `patched/engram.py`,
`patched/ep_weight_filter.py` (mounted over the image's copies by the compose service), `engram-nvme.diff` (review aid),
`test_nvme_lookup.py` (model-free GPU test, see below).

Regenerate after the image changes:

    docker run --rm --entrypoint cat IMAGE /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common/engram.py > engram.py
    docker run --rm --entrypoint cat IMAGE /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/ep_weight_filter.py > ep_weight_filter.py
    python3 -I apply_patch.py engram.py ep_weight_filter.py patched

Test (GPU 0, no model load; `SEED=<n>` changes the random rows, use a new one for cold timings):

    docker run --rm --gpus device=0 -e DSV41_ENGRAM_INT4=1 -e DSV41_ENGRAM_NVME=1 -e SEED=1 \
      -v $PWD/patched/engram.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common/engram.py:ro \
      -v /mnt/models/dsv41-flash-exl3:/mnt/models/dsv41-flash-exl3:ro -v $PWD:/t:ro \
      --entrypoint python3 IMAGE -I /t/test_nvme_lookup.py

It checks staged bytes vs `pread`, lookup semantics vs a torch reference, bit-equality with the stock kernel, and graph capture +
replay with fresh ids. Pitfalls: vLLM sets a CUDA default device during model init (CPU buffers need `device="cpu"`) and drops
torch intra-op threads to 1 (the gather has its own pool). Full history: `docs/TUNING.md`.
