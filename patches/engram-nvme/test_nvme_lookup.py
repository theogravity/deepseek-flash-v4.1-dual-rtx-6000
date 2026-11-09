#!/usr/bin/env python3
"""Model-free GPU test for the NVMe-backed Engram lookup (run in the serving image).

    docker run --rm --gpus device=0 -e DSV41_ENGRAM_INT4=1 -e DSV41_ENGRAM_NVME=1 \
      -v <patched>/engram.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common/engram.py:ro \
      -v /mnt/models/dsv41-flash-exl3:/mnt/models/dsv41-flash-exl3:ro \
      -v <this dir>:/t:ro --entrypoint python3 <image> -I /t/test_nvme_lookup.py

Checks (1) staged bytes == independent pread of the shard file, (2) lookup semantics
(owned / foreign / padding ids, padded tail heads) against a pure-torch reference,
(3) bit-equality with the STOCK int4 kernel on the same bytes, (4) the lookup captured in
a CUDA graph and replayed with fresh ids (how decode runs), (5) timing.
"""
import json
import os
import struct
import sys
import time

import torch
import triton

from vllm.models.deepseek_v41.common import engram as E

MODEL = os.environ.get("MODEL_DIR", "/mnt/models/dsv41-flash-exl3")
DIM, QB, TOTAL = 256, 32, 24
LAYER = 1
torch.manual_seed(int(os.environ.get("SEED", "0")))
torch.set_num_threads(1)  # vLLM does this for serving; the gather must not depend on torch's intra-op threads
dev = torch.device("cuda")
sms = torch.cuda.get_device_properties(0).multi_processor_count


def mk(local, max_tokens=16384):
    # vLLM runs model init under a CUDA default device; build the table the same way.
    with torch.device("cuda"):
        return E._NvmeEngramTable(MODEL, LAYER, DIM, QB, max_tokens=max_tokens, local_heads=local,
                                  total_heads=TOTAL, device=dev)


table = mk(12)
N = table.num_rows
print(f"table layer {LAYER}: {N} rows, w{tuple(table.w.shape)} s{tuple(table.s.shape)}")

# --- independent reader: header offsets + os.pread, no mmap, no torch indexing ---
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]


def raw_reader(tname, row_bytes):
    path = f"{MODEL}/{idx[tname]}"
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        start = json.loads(fh.read(n))[tname]["data_offsets"][0]
    fd = os.open(path, os.O_RDONLY)
    return lambda r: os.pread(fd, row_bytes, 8 + n + start + r * row_bytes)


read_w = raw_reader(f"layers.{LAYER}.engram.embed.weight", DIM // 2)
read_s = raw_reader(f"layers.{LAYER}.engram.embed.scale", (DIM // QB) * 2)


def ref_row(r):
    w = torch.frombuffer(bytearray(read_w(r)), dtype=torch.uint8)
    s = torch.frombuffer(bytearray(read_s(r)), dtype=torch.float16).float()
    cols = torch.arange(DIM)
    nib = ((w[cols // 2].to(torch.int32) >> ((cols % 2) * 4)) & 0xF)
    return ((nib - 8).float() * s[cols // QB]).to(torch.bfloat16)


fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (" " + extra if extra else ""))
    fails += 0 if ok else 1


vs, ve = 100_000_000, 250_000_000


def make_ids(t):
    ids = torch.randint(vs, ve, (t, TOTAL))                       # owned
    foreign = torch.rand(t, TOTAL) < 0.15
    ids[foreign] = torch.randint(ve, N, (int(foreign.sum()),))      # above our range
    low = torch.rand(t, TOTAL) < 0.10
    ids[low] = torch.randint(0, vs, (int(low.sum()),))              # below our range
    pad = torch.rand(t, TOTAL) < 0.05
    ids[pad] = -1                                                   # padding
    ids[:, 0] = ids[:, 1]                                           # force repeated rows (dedup path)
    return ids


def ref_lookup(ids, head_start, local):
    t = ids.shape[0]
    ref = torch.zeros((t, local, DIM), dtype=torch.bfloat16)
    for tok in range(t):
        for h in range(local):
            head = head_start + h
            if head >= TOTAL:
                continue
            r = int(ids[tok, head])
            if vs <= r < ve:
                ref[tok, h] = ref_row(r)
    return ref


def run_lookup(tbl, ids_dev, out, head_start, local, vstart=vs, vend=ve):
    tbl.lookup(ids_dev, out, head_start=head_start, local_heads=local, total_heads=TOTAL,
               vocab_start=vstart, vocab_end=vend, quant_block=QB, num_sms=sms)


def same(a, b):
    return torch.equal(a.cpu().view(torch.int16), b.view(torch.int16))


# 1. staged bytes -----------------------------------------------------------------
rows = torch.unique(torch.cat([torch.tensor([0, 1, N - 2, N - 1]), torch.randint(0, N, (3000,))]))
table._gather_into(rows)
ok = all(
    table.w_pin[i].numpy().tobytes() == read_w(int(r)) and table.s_pin[i].numpy().tobytes() == read_s(int(r))
    for i, r in enumerate(rows.tolist())
)
check(f"staged bytes == pread ({rows.numel()} rows incl. first/last)", ok)

# 2. lookup semantics vs torch reference (eager) ----------------------------------
for head_start, local in ((0, 12), (12, 12), (18, 12), (0, 24)):
    tbl = table if local == 12 else mk(local, 4096)
    t = 37
    ids = make_ids(t)
    out = torch.full((t, local, DIM), float("nan"), dtype=torch.bfloat16, device=dev)
    run_lookup(tbl, ids.to(dev), out, head_start, local)
    torch.cuda.synchronize()
    ref = ref_lookup(ids, head_start, local)
    check(f"lookup semantics vs reference (head_start={head_start}, local={local})",
          same(out, ref), f"max|diff|={(out.cpu().float() - ref.float()).abs().max().item():.3g}")

# 3. bit-equality with the stock kernel on the same bytes -------------------------
S0, M = 150_000_000, 1_000_000
w_slice = table.w[S0 : S0 + M].to(dev)
s_slice = table.s[S0 : S0 + M].to(dev)
t = 4096
ids = torch.randint(S0 - 5000, S0 + M + 5000, (t, TOTAL))
ids[torch.rand(t, TOTAL) < 0.05] = -1
local_heads, head_start = 12, 12
rows_n = t * local_heads
ids_dev = ids.to(dev)
out_stock = torch.empty((t, local_heads, DIM), dtype=torch.bfloat16, device=dev)
grid = min(triton.cdiv(rows_n, 16), sms)
E._engram_lookup_int4_kernel[(grid,)](
    w_slice, s_slice, ids_dev, out_stock, S0, S0 + M, rows_n, ids_dev.stride(0), ids_dev.stride(1),
    HEAD_START=head_start, LOCAL_HEADS=local_heads, TOTAL_HEADS=TOTAL, DIM=DIM, QUANT_BLOCK=QB,
    BLOCK_R=16, GRID=grid,
)
out_nvme = torch.empty_like(out_stock)
run_lookup(table, ids_dev, out_nvme, head_start, local_heads, S0, S0 + M)
torch.cuda.synchronize()
check("NVMe lookup == stock kernel, bit-exact (4096 tokens x 12 heads)",
      torch.equal(out_stock.view(torch.int16), out_nvme.view(torch.int16)))
del w_slice, s_slice

# 4. CUDA graph capture + replay with fresh ids (decode path) ---------------------
for t in (48, 6):
    static_ids = make_ids(t).to(dev)
    out_g = torch.empty((t, 12, DIM), dtype=torch.bfloat16, device=dev)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):  # warm up eagerly first, as vLLM does
        run_lookup(table, static_ids, out_g, 12, 12)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_lookup(table, static_ids, out_g, 12, 12)
    good = True
    times = []
    for rep in range(4):
        ids = make_ids(t)
        static_ids.copy_(ids.to(dev))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        good &= same(out_g, ref_lookup(ids, 12, 12))
    check(f"graph capture+replay, fresh ids each replay (t={t}, 4 replays)", good,
          "replay ms: " + " ".join(f"{x:.2f}" for x in times))

# 5. timing + spot-verification on a table nothing has touched yet (layer 14), ids over its whole range
L14 = 14
with torch.device("cuda"):
    tbl14 = E._NvmeEngramTable(MODEL, L14, DIM, QB, max_tokens=16384, local_heads=12, total_heads=TOTAL, device=dev)
N14 = tbl14.num_rows
rw14 = raw_reader(f"layers.{L14}.engram.embed.weight", DIM // 2)
rs14 = raw_reader(f"layers.{L14}.engram.embed.scale", (DIM // QB) * 2)


def ref_row14(r):
    w = torch.frombuffer(bytearray(rw14(r)), dtype=torch.uint8)
    s = torch.frombuffer(bytearray(rs14(r)), dtype=torch.float16).float()
    cols = torch.arange(DIM)
    nib = ((w[cols // 2].to(torch.int32) >> ((cols % 2) * 4)) & 0xF)
    return ((nib - 8).float() * s[cols // QB]).to(torch.bfloat16)


for t, label in ((48, "decode step, 48 tokens"), (6, "decode step, 6 tokens"), (16384, "prefill chunk, 16384 tokens")):
    for rep in range(2):
        ids_cpu = torch.randint(0, N14, (t, TOTAL))
        ids = ids_cpu.to(dev)
        out = torch.empty((t, 12, DIM), dtype=torch.bfloat16, device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_lookup(tbl14, ids, out, 12, 12, 0, N14)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        out_cpu = out.cpu()
        good = all(
            torch.equal(out_cpu[tok, h].view(torch.int16), ref_row14(int(ids_cpu[tok, 12 + h])).view(torch.int16))
            for tok, h in [(int(torch.randint(0, t, ())), int(torch.randint(0, 12, ()))) for _ in range(24)]
        )
        print(f"TIME {label}: {ms:8.2f} ms  ({t * 12} rows over whole {N14}-row table, rep {rep}, 24 spot-checks {'ok' if good else 'MISMATCH'})")
        fails += 0 if good else 1

print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
