#!/usr/bin/env python3
"""Generate the NVMe-backed Engram patch for the DeepSeek V4.1 Flash EXL3 image.

Reads the image's stock vllm files and writes patched copies that compose mounts
over them (like the other files under patches/). Gated by DSV41_ENGRAM_NVME=1;
with the variable unset the patched files behave exactly like the stock ones.

    python3 -I apply_patch.py <stock engram.py> <stock ep_weight_filter.py> <out dir>

Why: stock vLLM pins both ~51.5 GiB int4 Engram tables in host RAM (103 GiB total,
unswappable). Their rows are only ever addressed by token-id hashes, so with this
patch the tables stay in the checkpoint shards on NVMe and each step reads just
the rows it needs (mmap + page cache), stages them on the GPU as a compact table
and runs the *unchanged* int4 Triton kernel on it. Same bytes, same dequant math.
"""
from __future__ import annotations

import sys
from pathlib import Path

NVME_BLOCK = '''
# DSV41_ENGRAM_NVME=1: serve the int4 Engram tables straight from the checkpoint
# shards (mmap, never pinned) instead of holding them in pinned host RAM.
#
# CUDA-graph friendly: decode runs as one FULL graph, so nothing here may sync the host.
# A lookup is  D2H ids -> pinned | host node: gather rows from NVMe into pinned staging |
# H2D staging -> GPU | the stock int4 kernel on that compact table.  All shapes and kernel
# arguments depend only on the token count, so graphs capture and replay it unchanged.
_ENGRAM_NVME = os.environ.get("DSV41_ENGRAM_NVME", "0") == "1"
# Up to this many unique rows per lookup get one WILLNEED hint each so the NVMe sees a deep
# queue; bigger batches are gathered by _NVME_THREADS threads faulting in parallel.
_NVME_WILLNEED_MAX_ROWS = int(os.environ.get("DSV41_ENGRAM_NVME_WILLNEED_MAX", "1024"))
_NVME_THREADS = int(os.environ.get("DSV41_ENGRAM_NVME_THREADS", "32"))

_HOSTFN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_HOST_SLOTS: list = [None]  # slot 0 unused: ctypes hands a NULL void* to Python as None
_CUDART = None


def _host_trampoline(slot):
    try:
        _HOST_SLOTS[slot]()
    except BaseException:
        import sys
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        # Stale staging rows would silently corrupt generations; die loudly instead.
        os._exit(70)


_HOST_TRAMPOLINE = _HOSTFN(_host_trampoline)


def _launch_host_slot(slot: int) -> None:
    """Enqueue (or, under stream capture, add a graph host node for) _HOST_SLOTS[slot]."""
    global _CUDART
    if _CUDART is None:
        lib = None
        for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("libcudart not found (needed for cudaLaunchHostFunc)")
        lib.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, _HOSTFN, ctypes.c_void_p]
        lib.cudaLaunchHostFunc.restype = ctypes.c_int
        _CUDART = lib
    err = _CUDART.cudaLaunchHostFunc(
        torch.cuda.current_stream().cuda_stream, _HOST_TRAMPOLINE, slot
    )
    if err != 0:
        raise RuntimeError(f"cudaLaunchHostFunc failed with CUDA error {err}")


class _NvmeEngramTable:
    """One layer's int4 Engram table, read row-wise from its safetensors shards."""

    def __init__(
        self,
        model_dir: str,
        layer_id: int,
        dim: int,
        block_size: int,
        *,
        max_tokens: int,
        local_heads: int,
        total_heads: int,
        device: torch.device,
    ) -> None:
        import json

        with open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        self.dim = dim
        self.sb = dim // block_size
        self._keep: list = []
        self.w_fd, self.w_base, self.w = self._map(
            model_dir, weight_map, f"layers.{layer_id}.engram.embed.weight", np.uint8, (dim // 2,)
        )
        self.s_fd, self.s_base, self.s = self._map(
            model_dir, weight_map, f"layers.{layer_id}.engram.embed.scale", np.float16, (self.sb,)
        )
        assert self.w.shape[0] == self.s.shape[0], "engram weight/scale row counts differ"
        self.num_rows = self.w.shape[0]
        # Fixed-address staging, so captured graphs keep reading/writing the same memory.
        # vLLM sets a CUDA default device during model init, so CPU buffers must say so.
        cap = max_tokens * local_heads
        self.max_tokens = max_tokens
        self.ids_pin = torch.zeros((max_tokens, local_heads), dtype=torch.int64, device="cpu").pin_memory()
        self.cid_pin = torch.full((max_tokens, total_heads), -1, dtype=torch.int64, device="cpu").pin_memory()
        self.w_pin = torch.zeros((cap, dim // 2), dtype=torch.uint8, device="cpu").pin_memory()
        self.s_pin = torch.zeros((cap, self.sb), dtype=torch.float16, device="cpu").pin_memory()
        self.cid_gpu = torch.full((max_tokens, total_heads), -1, dtype=torch.int64, device=device)
        self.w_gpu = torch.zeros((cap, dim // 2), dtype=torch.uint8, device=device)
        self.s_gpu = torch.zeros((cap, self.sb), dtype=torch.float16, device=device)
        self._slots: dict = {}
        self._pool = None

    def _map(self, model_dir, weight_map, tname, np_dtype, row_shape):
        import json
        import struct
        import warnings

        path = os.path.join(model_dir, weight_map[tname])
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            meta = json.loads(fh.read(n))[tname]
        shape = tuple(meta["shape"])
        assert shape[1:] == row_shape, f"{tname}: row shape {shape[1:]} != {row_shape}"
        base = 8 + n + meta["data_offsets"][0]
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        mapping.madvise(mmap.MADV_RANDOM)
        arr = np.frombuffer(mapping, dtype=np_dtype, count=int(np.prod(shape)), offset=base).reshape(shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only mapping; we only ever read
            tensor = torch.from_numpy(arr)
        self._keep.append(mapping)
        return fd, base, tensor

    def _gather_slice(self, rows: torch.Tensor, lo: int, hi: int) -> None:
        part = rows[lo:hi]
        torch.index_select(self.w, 0, part, out=self.w_pin[lo:hi])
        torch.index_select(self.s, 0, part, out=self.s_pin[lo:hi])

    def _gather_into(self, rows: torch.Tensor) -> None:
        """rows: [U] int64 global row ids -> w_pin[:U], s_pin[:U] (host thread, no CUDA calls).

        vLLM drops torch's intra-op threads to 1 for serving, which would leave this a
        queue-depth-1 page-fault loop, so big gathers are split over our own threads
        (torch releases the GIL inside index_select)."""
        n = rows.numel()
        if n <= _NVME_WILLNEED_MAX_ROWS:
            wb, sb = self.dim // 2, self.sb * 2
            for r in rows.tolist():
                os.posix_fadvise(self.w_fd, self.w_base + r * wb, wb, os.POSIX_FADV_WILLNEED)
                os.posix_fadvise(self.s_fd, self.s_base + r * sb, sb, os.POSIX_FADV_WILLNEED)
            self._gather_slice(rows, 0, n)
            return
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=_NVME_THREADS, thread_name_prefix="engram-nvme")
        step = -(-n // _NVME_THREADS)
        futures = [self._pool.submit(self._gather_slice, rows, lo, min(lo + step, n)) for lo in range(0, n, step)]
        for future in futures:
            future.result()

    def _host_gather(self, t, head_start, end, local_heads, vocab_start, vocab_end) -> None:
        """Runs on a CUDA host-function thread: ids_pin[:t] -> cid_pin[:t], w_pin, s_pin."""
        flat = self.ids_pin[:t].reshape(-1)
        owned = (flat >= vocab_start) & (flat < vocab_end)
        uniq, inv = torch.unique(flat[owned], return_inverse=True)
        comp = torch.full_like(flat, -1)
        comp[owned] = inv
        cid = self.cid_pin[:t]
        cid.fill_(-1)
        cid[:, head_start:end] = comp.view(t, local_heads)[:, : end - head_start]
        if uniq.numel():
            self._gather_into(uniq)

    def _slot(self, key) -> int:
        slot = self._slots.get(key)
        if slot is None:
            _HOST_SLOTS.append(lambda: self._host_gather(*key))
            slot = self._slots[key] = len(_HOST_SLOTS) - 1
        return slot

    def lookup(
        self,
        indices: torch.Tensor,
        out: torch.Tensor,
        *,
        head_start: int,
        local_heads: int,
        total_heads: int,
        vocab_start: int,
        vocab_end: int,
        quant_block: int,
        num_sms: int,
        background: bool = False,
    ) -> None:
        """Same contract as ParallelEngramEmbedding.lookup for the int4 kernel:
        out[token * local_heads + h] = dequant(row ids[token, head_start + h]), or zeros
        when that id is padding or belongs to another rank's rows."""
        t = indices.shape[0]
        assert t <= self.max_tokens, f"engram lookup of {t} tokens > staging {self.max_tokens}"
        end = min(head_start + local_heads, total_heads)
        ids = indices[:, head_start:end].to(torch.int64)
        if ids.shape[1] < local_heads:
            ids = nn.functional.pad(ids, (0, local_heads - ids.shape[1]), value=-1)
        self.ids_pin[:t].copy_(ids, non_blocking=True)
        _launch_host_slot(self._slot((t, head_start, end, local_heads, vocab_start, vocab_end)))
        n_rows = t * local_heads
        self.w_gpu[:n_rows].copy_(self.w_pin[:n_rows], non_blocking=True)
        self.s_gpu[:n_rows].copy_(self.s_pin[:n_rows], non_blocking=True)
        self.cid_gpu[:t].copy_(self.cid_pin[:t], non_blocking=True)
        grid = min(triton.cdiv(n_rows, 16), num_sms // 2 if background else num_sms)
        # Compact ids are -1 (unowned / padding) or < n_rows, so (0, n_rows) is a static bound.
        _engram_lookup_int4_kernel[(grid,)](
            self.w_gpu, self.s_gpu, self.cid_gpu, out, 0, n_rows, n_rows,
            self.cid_gpu.stride(0), self.cid_gpu.stride(1),
            HEAD_START=head_start, LOCAL_HEADS=local_heads, TOTAL_HEADS=total_heads,
            DIM=self.dim, QUANT_BLOCK=quant_block, BLOCK_R=16, GRID=grid,
        )


'''

FILTER_BLOCK = '''import os as _os_nvme

# DSV41_ENGRAM_NVME=1: the Engram tables are read from the shards on demand
# (see models/deepseek_v41/common/engram.py), so never load them into host RAM.
_ENGRAM_NVME_SKIP = _os_nvme.environ.get("DSV41_ENGRAM_NVME", "0") == "1"


def should_skip_weight('''


def replace_once(src: str, old: str, new: str, label: str) -> str:
    count = src.count(old)
    if count != 1:
        raise SystemExit(f"apply_patch: anchor '{label}' found {count} times (expected 1)")
    return src.replace(old, new, 1)


def patch_engram(src: str) -> str:
    src = replace_once(
        src,
        "\nclass ParallelEngramEmbedding(nn.Module):\n",
        NVME_BLOCK + "class ParallelEngramEmbedding(nn.Module):\n",
        "class ParallelEngramEmbedding",
    )
    src = replace_once(
        src,
        "    _shared_memory: DPSharedEngramStorage | None = None\n    _packed: torch.Tensor | None = None\n",
        "    _shared_memory: DPSharedEngramStorage | None = None\n    _packed: torch.Tensor | None = None\n"
        "    _nvme: \"_NvmeEngramTable | None\" = None\n",
        "class attrs",
    )
    src = replace_once(
        src,
        "        dp_shared_memory: bool = False,\n        use_thp: bool = False,\n    ) -> None:\n"
        "        super().__init__()\n        assert head_sizes",
        "        dp_shared_memory: bool = False,\n        use_thp: bool = False,\n"
        "        nvme_source: tuple[str, int] | None = None,\n    ) -> None:\n"
        "        super().__init__()\n        assert head_sizes",
        "__init__ signature",
    )
    src = replace_once(
        src,
        "        self.int4 = _ENGRAM_INT4   # packed int4 rows + fp16 per-block scales (convert_engram_int4.py)\n"
        "        weight, scales = self._allocate_weights()\n",
        "        self.int4 = _ENGRAM_INT4   # packed int4 rows + fp16 per-block scales (convert_engram_int4.py)\n"
        "        if nvme_source is not None:\n"
        "            assert self.int4 and cpu_offload and self.dp_size == 1, (\n"
        "                'DSV41_ENGRAM_NVME needs DSV41_ENGRAM_INT4=1, Engram cpu_offload and no Engram DP'\n"
        "            )\n"
        "            self._nvme = _NvmeEngramTable(\n"
        "                nvme_source[0], nvme_source[1], dim, block_size,\n"
        "                max_tokens=get_current_vllm_config().scheduler_config.max_num_batched_tokens,\n"
        "                local_heads=self.part_n_hash_cols, total_heads=self.n_hash_cols,\n"
        "                device=torch.device('cuda', torch.cuda.current_device()),\n"
        "            )\n"
        "            assert self.vocab_end_idx <= self._nvme.num_rows, (\n"
        "                f'engram rows {self.vocab_end_idx} > checkpoint rows {self._nvme.num_rows}'\n"
        "            )\n"
        "            # Placeholders as buffers: nothing for the weight loader to fill or to report missing.\n"
        "            self.register_buffer('weight', torch.zeros(1, dim // 2, dtype=torch.uint8), persistent=False)\n"
        "            self.register_buffer(\n"
        "                'weight_scale_inv', torch.zeros(1, dim // block_size, dtype=torch.float16), persistent=False\n"
        "            )\n"
        "            logger.info(\n"
        "                'Engram table served from NVMe (layer %d, %d rows x %d, %.2f GiB stays on disk; rows %d..%d are ours)',\n"
        "                nvme_source[1], self._nvme.num_rows, dim,\n"
        "                self._nvme.num_rows * (dim // 2 + 2 * (dim // block_size)) / 1024**3,\n"
        "                self.vocab_start_idx, self.vocab_end_idx,\n"
        "            )\n"
        "            return\n"
        "        weight, scales = self._allocate_weights()\n",
        "__init__ nvme branch",
    )
    src = replace_once(
        src,
        "        weight, scales = self._storage()\n        ids_stride_t, ids_stride_h = indices.stride()\n",
        "        if self._nvme is not None:\n"
        "            self._nvme.lookup(\n"
        "                indices, out, head_start=self.head_start, local_heads=self.part_n_hash_cols,\n"
        "                total_heads=self.n_hash_cols, vocab_start=self.vocab_start_idx,\n"
        "                vocab_end=self.vocab_end_idx, quant_block=self.block_size,\n"
        "                num_sms=self._num_sms, background=background,\n"
        "            )\n"
        "            return\n"
        "        weight, scales = self._storage()\n        ids_stride_t, ids_stride_h = indices.stride()\n",
        "lookup hook",
    )
    src = replace_once(
        src,
        "            dp_shared_memory=bool(engram_config.dp_shared_memory),\n            use_thp=engram_config.use_thp,\n        )\n",
        "            dp_shared_memory=bool(engram_config.dp_shared_memory),\n            use_thp=engram_config.use_thp,\n"
        "            nvme_source=(\n"
        "                get_current_vllm_config().model_config.model,\n"
        "                layout.layer_ids[layer_hash_index],\n"
        "            ) if _ENGRAM_NVME else None,\n"
        "        )\n",
        "_create_embedding",
    )
    return src


def patch_filter(src: str) -> str:
    src = replace_once(src, "def should_skip_weight(", FILTER_BLOCK, "should_skip_weight def")
    return replace_once(
        src,
        '    belong to the local rank and should be skipped during loading."""\n    if local_expert_ids is None:\n',
        '    belong to the local rank and should be skipped during loading."""\n'
        '    if _ENGRAM_NVME_SKIP and weight_name.endswith(\n'
        '        (".engram.embed.weight", ".engram.embed.scale")\n'
        '    ):\n'
        '        return True\n'
        '    if local_expert_ids is None:\n',
        "should_skip_weight body",
    )


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    engram_in, filter_in, out_dir = (Path(a) for a in sys.argv[1:])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "engram.py").write_text(patch_engram(engram_in.read_text()))
    (out_dir / "ep_weight_filter.py").write_text(patch_filter(filter_in.read_text()))
    print(f"wrote {out_dir}/engram.py and {out_dir}/ep_weight_filter.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
