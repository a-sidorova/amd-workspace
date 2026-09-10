# GPU Communication Overview — AMD (XGMI / RDMA) vs NVIDIA (NVLink)

> Scope: how GPUs talk to each other, at two scales:
> - **Intra-node (scale-up)** — GPU↔GPU *within a box*: AMD **XGMI / Infinity Fabric** vs NVIDIA **NVLink 5 + NVSwitch**.
> - **Inter-node (scale-out)** — GPU↔GPU *across nodes/racks*: **RDMA (RoCE / InfiniBand)** on both sides.
>
> Systems covered:
> - **AMD:** 8× Instinct **MI355X** (gfx950, CDNA4) + 8× Pensando 400 GbE RoCE NICs (numbers verified on this node).
> - **NVIDIA:** **DGX/HGX B200** (8 GPUs) and **GB200 NVL72** (72 GPUs), from NVIDIA docs.

---

## Units

All bandwidth figures are **GB/s (gigabytes/second)** unless a table says otherwise, so every number is directly
comparable. NIC line rates are natively marketed in **Gb/s (bits)**; converted here via `Gb/s ÷ 8 = GB/s`.

Two conventions to keep straight when comparing vendors:
- **Bidirectional** = both directions summed. **Single direction** = one way (half of bidirectional on a full-duplex link).
- NVIDIA quotes NVLink two different ways for two different things, so don't equate them:
  - **Per-GPU:** 1.8 TB/s — this is **bidirectional** and covers **all 18 links** of the GPU.
  - **Per-link:** the networking side calls one link "400 Gbps **unidirectional**" (= 50 GB/s each way = 100 GB/s bidi).
  These are the same physical wires counted at different granularity (whole GPU vs. one link) and in different units
  (bytes bidi vs. bits per direction) — not two names for the same number.

---
---

# PART A — AMD: XGMI vs RDMA

## A1. What each one is

| | **XGMI (Infinity Fabric)** | **RDMA (RoCE v2)** |
|---|---|---|
| Purpose | **Scale-up** — GPU↔GPU *within a node* | **Scale-out** — GPU↔GPU *across nodes* |
| Medium | On-baseboard (UBB) Infinity Fabric mesh | Ethernet (RoCE v2) via NIC |
| Hardware here | MI355X OAM links | Pensando DSC (`ionic_0..7`) |
| Software path | HIP P2P / RCCL (direct, no NIC) | libibverbs → UCX/NIXL → NIC |
| CPU involvement | None (hardware fabric) | None on data path (kernel bypass, zero-copy) |
| Typical use | Tensor/expert parallel, RCCL collectives | PD-disaggregation KV transfer, multi-node TP/PP |

## A2. XGMI (Infinity Fabric) — intra-node

Bidirectional link. **Direct all-to-all mesh** (no switch): each GPU has one dedicated link to each peer.

| Metric | GB/s |
|---|---|
| Per link (GPU↔GPU pair), bidirectional | **153.6 GB/s** |
| Per link, single direction | **~76.8 GB/s** |
| Per-GPU aggregate (all 7 links), bidirectional | **~1075.2 GB/s ≈ 1.075 TB/s** |
| Per-GPU aggregate, single direction | ~537 GB/s |
| Node total (8 GPUs), bidirectional | ≈ 8601.6 GB/s ≈ 8.6 TB/s |

| Property | Value |
|---|---|
| Links per GPU | **7** (one to each peer) |
| Topology | 8-GPU **fully-connected all-to-all mesh**, **1 hop** |

**P2P reality (single GPU→GPU copy):**
- One-way `hipMemcpyPeer(A→B)` is capped at the **one dedicated link** → **~76.8 GB/s** theoretical (~65–70 GB/s achievable, ~85–90%).
- Both directions at once → 153.6 GB/s total on the link (full duplex).
- The ~1.075 TB/s figure only appears when a GPU talks to **all 7 peers simultaneously** (e.g. RCCL ring/tree collectives), **not** on a single pair.

**Verified on this node (amdgpu sysfs):**
- 8 physical cards (card1/9/17/25/33/41/49/57), `xgmi_num_links = 1` to each of 7 peers, `xgmi_num_hops = 1`.
- `rocm-smi --showtopo`: every GPU pair = `XGMI`, weight 15, 1 hop.

## A3. RDMA / RoCE — inter-node

Full-duplex port.

| Metric | GB/s |
|---|---|
| Per NIC port, line rate (per direction) | **50 GB/s** |
| Per NIC port, bidirectional (full-duplex) | 100 GB/s |
| Per GPU (1:1 NIC), per direction | ~50 GB/s |
| Node total (8 NICs), per direction | 400 GB/s |
| Node total, bidirectional | 800 GB/s |
| Achievable (`ib_write_bw`), per direction | ~42–47 GB/s |

| Property | Value |
|---|---|
| Lanes / signaling | **4X NDR** (4 lanes × ~12.5 GB/s/lane) |
| MTU | 9000 (jumbo) |
| NIC : GPU ratio | **1:1** (rail-optimized) |
| NICs on node | **8×** Pensando DSC (`ionic_0..7`) |

**Transport properties:** kernel bypass, zero-copy, CPU offload, GPUDirect RDMA (NIC DMAs straight into GPU HBM).

**Addressing here:** RoCE v2 over IPv6 ULA (`fd93:16d3:59b6:XXXX::/64`), one `/64` rail per NIC; management is separate (`eth0`, 1.25 GB/s, do **not** route KV traffic over it).

## A4. AMD side-by-side

| Path | Per direction (GB/s) | Bidirectional | Scope |
|---|:---:|:---:|---|
| HBM3E (GPU↔VRAM) | ~8000 | — | inside GPU |
| XGMI, all 7 links (per GPU) | ~537 | ~1.075 TB/s | intra-node |
| XGMI, single link (GPU↔GPU) | **~76.8** | 153.6 GB/s | intra-node |
| RDMA/RoCE, 1 NIC (per GPU) | **~50** | 100 GB/s | inter-node |

**Hierarchy (per GPU):**
`HBM (8 TB/s) ≫ XGMI aggregate (~1 TB/s bidi) > single XGMI link (153.6 GB/s bidi) > RDMA NIC (100 GB/s bidi)`

---
---

# PART B — NVIDIA: NVLink 5 & NVLink Switch

## B1. The three layers

| | **NVLink 5 (link)** | **NVSwitch / NVLink Switch** | **RDMA (InfiniBand NDR / RoCE)** |
|---|---|---|---|
| Purpose | GPU↔GPU **point-to-point** SerDes | **Switched fabric** — all-to-all GPU mesh | **Scale-out** — GPU↔GPU *across nodes/racks* |
| Scope | one GPU's ports | one node (B200) or one rack (NVL72) | beyond the NVLink domain |
| Medium | 224G PAM4 SerDes (copper) | NVSwitch ASIC + copper backplane | ConnectX-7 NIC → IB/Ethernet switch |
| Software path | CUDA P2P / NCCL (direct, no NIC) | CUDA P2P / NCCL (SHARP in-network reduce) | libibverbs → NCCL/UCX → NIC (GPUDirect RDMA) |
| CPU on data path | none | none | none (kernel bypass, zero-copy) |
| Typical use | tensor/expert parallel, collectives | node-wide / rack-wide collectives | multi-node TP/PP, KV transfer, SuperPOD |

**Mental model:** NVLink is the *link* (the wire). NVSwitch is the *switch* that lets every GPU use its full NVLink
bandwidth to **any** peer at once (non-blocking). InfiniBand is the fallback once you leave the NVLink domain.

> **Key difference vs. AMD XGMI:** AMD uses a **direct mesh** (links = peers, 1 wire per peer, no switch). NVIDIA's
> 18 links per GPU **all feed NVSwitch chips**, not peers — so one GPU→GPU pair can use *all 18* links at once.

## B2. NVLink 5 — the per-GPU link (Blackwell B200)

Every Blackwell (B200) GPU exposes **18 NVLink 5 links** built from 224 Gbps PAM4 SerDes.

| Metric | Value |
|---|---|
| Per link, bidirectional | **100 GB/s** (50 GB/s each direction) |
| Per link, single direction | **50 GB/s** (= 400 Gbps in networking terms) |
| Links per GPU | **18** (all wired to switches, **not** to peers) |
| Per-GPU aggregate, bidirectional | **1,800 GB/s = 1.8 TB/s** (18 × 100) |
| Per-GPU aggregate, single direction | **900 GB/s** |
| Generation lift | **2× NVLink 4 (Hopper, 900 GB/s)** — same 18 links, per-link speed doubled (25 → 50 GB/s/dir) |

## B3. Intra-node: DGX / HGX B200 (8 GPUs)

The 18 links of each of the 8 GPUs land on **2× fifth-gen NVSwitch chips** → **non-blocking all-to-all fabric**:
any GPU hits any other at full NVLink speed, in **one switch hop**. (8 GPUs × 18 links = 144 = exactly the 2 × 72
switch ports.)

| Metric | Value |
|---|---|
| GPUs per node | **8× B200** (1,440 GB total HBM3E) |
| NVSwitch chips on baseboard | **2** (5th gen), 7.2 TB/s each |
| Any GPU↔GPU pair, bidirectional | **1.8 TB/s** (900 GB/s per direction) |
| Any GPU↔GPU pair, single direction | **900 GB/s** |
| **Node total NVLink, bidirectional** | **14.4 TB/s** (8 × 1.8, = 2 × 7.2 TB/s switch) |
| Topology | 8-GPU **all-to-all via NVSwitch**, **1 hop**, non-blocking |

**P2P reality:** the switch does **not** cap one pair to a single link — a GPU↔GPU pair can use **all 18 links →
the full 1.8 TB/s bidi (900 GB/s each way)**. The 14.4 TB/s node figure only appears with all 8 GPUs active.

**NVSwitch chip (5th gen):** 72 ports/chip @ 100 GB/s → **7.2 TB/s**; SHARP engines do in-network reductions/multicast.

## B4. Rack-scale: GB200 NVL72 (72 GPUs, one NVLink domain)

The **same 1.8 TB/s NVLink 5** stretched across a whole rack, so **72 Blackwell GPUs act as one giant GPU**. Every
GPU keeps its full 1.8 TB/s; the fabric is non-blocking.

**Physical build:** 36× GB200 superchips (36 Grace + **72 Blackwell**) · **18 compute trays** (4 GPUs each) ·
**9 NVLink Switch trays** (2 chips each = **18 switch chips**) · rear **copper-cable backplane** · **NVLink-C2C**
bridges each Grace CPU ↔ its GPUs at **900 GB/s**.

| Metric | Value |
|---|---|
| GPUs in one NVLink domain | **72** (largest NVLink domain NVIDIA ships) |
| Any GPU↔GPU pair, bidirectional | **1.8 TB/s** (900 GB/s per direction), **1 switch hop** |
| **Rack total NVLink, bidirectional** | **130 TB/s** (72 × 1.8 ≈ 129.6) |
| Bisection bandwidth | ~65 TB/s |
| NVSwitch trays / chips | **9 trays / 18 chips** (7.2 TB/s per chip) |
| Grace↔Blackwell (NVLink-C2C) | 900 GB/s coherent |
| Scaling vs. 8-GPU node | ~**9× the GPU throughput** of a single 8-GPU system |

**Why it matters:** a 72-GPU all-reduce stays within **~8–12%** of intra-baseboard line rate — the whole rack
behaves like one shared-memory machine, and the "8-GPU wall" is gone inside the rack.

## B5. NVIDIA inter-node / scale-out

Once you exceed the NVLink domain (9th GPU on B200, or a second rack on NVL72), traffic drops onto
**InfiniBand/Ethernet via ConnectX NICs** — ~an order of magnitude slower per GPU than NVLink.

| Metric | DGX B200 | GB200 NVL72 |
|---|---|---|
| Compute NICs | **8× ConnectX-7** (1:1 GPU) | **4× ConnectX-7 per tray → 72 total** (1:1 GPU) |
| Per NIC, per direction | **50 GB/s** (400 Gbps NDR) | **50 GB/s** (400 Gbps NDR) |
| Per NIC, bidirectional | 100 GB/s | 100 GB/s |
| **Total scale-out, per direction** | **400 GB/s** (8 × 50) | **~3,600 GB/s** (72 × 50) |
| Storage / in-band mgmt | 2× BlueField-3 DPU | 2× BlueField-3 per tray |
| Beyond the domain | 9th GPU → IB | NVL576 / SuperPOD → IB between racks (~10× slower than in-rack NVLink) |

**Transport:** GPUDirect RDMA, kernel bypass, zero-copy.

## B6. NVIDIA side-by-side (per GPU unless noted)

| Path | Per direction (GB/s) | Bidirectional | Scope |
|---|:---:|:---:|---|
| HBM3E (GPU↔VRAM), B200 | ~8,000 | — | inside GPU |
| NVLink 5, full per-GPU (via NVSwitch) | **900** | **1.8 TB/s** | intra-node (B200) & rack (NVL72) |
| NVLink 5, single link | 50 | 100 GB/s | one wire |
| DGX B200 node total (8 GPUs) | 7,200 | **14.4 TB/s** | intra-node |
| GB200 NVL72 rack total (72 GPUs) | 65,000 | **130 TB/s** | rack (scale-up) |
| RDMA / ConnectX-7 NDR (1 NIC per GPU) | **50** | 100 GB/s | inter-node / rack↔rack |

---
---

# PART C — AMD vs NVIDIA comparison

## C1. Intra-node (scale-up) — GPU↔GPU within a box

| | **AMD MI355X** | **NVIDIA B200 (DGX/HGX)** | **NVIDIA GB200 NVL72** |
|---|---|---|---|
| Fabric | XGMI / Infinity Fabric | NVLink 5 + NVSwitch | NVLink 5 + NVLink Switch System |
| Topology | **Direct all-to-all mesh** (no switch) | **Switched** all-to-all | **Switched** all-to-all (rack-wide) |
| GPUs in domain | 8 | 8 | **72** |
| Links per GPU | 7 (one per peer) | 18 (all to switches) | 18 (all to switches) |
| Hops GPU→GPU | 1 (direct wire) | 1 (through switch) | 1 (through switch) |
| **Single pair, per direction** | **~76.8 GB/s** | **~900 GB/s** | **~900 GB/s** |
| **Single pair, bidirectional** | **153.6 GB/s** | **1,800 GB/s** | **1,800 GB/s** |
| Per-GPU aggregate, bidirectional | ~1.075 TB/s | 1.8 TB/s | 1.8 TB/s |
| **Domain total, bidirectional** | ~8.6 TB/s (8 GPUs) | **14.4 TB/s** (8 GPUs) | **130 TB/s** (72 GPUs) |
| Single-pair cap? | **Yes** — 1 link only | No — uses all 18 links | No — uses all 18 links |

**Takeaways:**
- Per **link**, AMD XGMI (76.8 GB/s/dir) is actually *faster* than one NVLink 5 link (50 GB/s/dir).
- On a **single GPU→GPU pair**, NVIDIA wins **~11–12×** (900 vs 76.8 GB/s/dir) — purely because the switch lets a
  pair use all 18 links, while XGMI's mesh caps a pair to its one dedicated link.
- NVIDIA's real structural advantage is **domain size**: NVL72 keeps **72 GPUs** at full 1.8 TB/s each, vs. AMD's
  8-GPU mesh.

## C2. Inter-node (scale-out) — GPU↔GPU across nodes/racks

| | **AMD MI355X node** | **NVIDIA DGX B200** | **NVIDIA GB200 NVL72** |
|---|---|---|---|
| Transport | RoCE v2 (Ethernet) | InfiniBand NDR (or Ethernet) | InfiniBand NDR (or Ethernet) |
| NIC | 8× Pensando DSC | 8× ConnectX-7 | 72× ConnectX-7 (4/tray) |
| NIC : GPU ratio | 1:1 | 1:1 | 1:1 |
| **Per GPU, per direction** | **~50 GB/s** | **~50 GB/s** | **~50 GB/s** |
| Per GPU, bidirectional | 100 GB/s | 100 GB/s | 100 GB/s |
| Node/rack total, per direction | 400 GB/s | 400 GB/s | ~3,600 GB/s |
| Achievable, per direction | ~42–47 GB/s | ~42–47 GB/s | ~42–47 GB/s |

**Takeaway:** inter-node per-GPU bandwidth is **~50 GB/s/dir on both vendors** — the scale-out NIC is the great
equalizer (and the bottleneck). The vendor difference lives entirely in the **intra-node/scale-up** fabric.

## C3. Full hierarchy, both vendors (per GPU, single direction)

| Tier | AMD MI355X | NVIDIA B200 / GB200 |
|---|---|---|
| HBM (inside GPU) | ~8,000 GB/s | ~8,000 GB/s |
| Scale-up per-GPU aggregate | ~537 GB/s (XGMI, 7 links) | **900 GB/s** (NVLink, 18 links) |
| Scale-up single pair | ~76.8 GB/s (1 XGMI link) | **900 GB/s** (all 18 links via switch) |
| Scale-out per GPU | ~50 GB/s (RoCE NIC) | ~50 GB/s (ConnectX-7 NIC) |

## C4. Why it matters for placement (both vendors)

- **Keep collectives inside the scale-up domain** — AMD: 8 GPUs (XGMI); NVIDIA B200: 8 GPUs; **NVIDIA NVL72: 72 GPUs**.
- **Crossing nodes over RDMA is the slowest hop** (~50 GB/s/dir per GPU on both) — send only compact data
  (KV-cache, post-reduction gradients) across the network, never a domain-wide all-reduce.
- **AMD single-pair caveat:** a single GPU→GPU copy on MI355X is capped to one link (~76.8 GB/s/dir); you only get
  the ~1 TB/s aggregate when talking to all 7 peers at once. NVIDIA has no such single-pair cap.
- **NVIDIA's structural win is domain size**, not per-link speed: NVL72 turns the "8-GPU node" boundary into a
  "72-GPU rack" boundary, so large models fit in one NVLink domain instead of sharding across the slow IB fabric.

---

**Sources:** AMD numbers verified on-node (amdgpu sysfs, `rocm-smi --showtopo`). NVIDIA:
[DGX B200 User Guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html),
[NVLink & NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/),
[GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/),
[Blackwell architecture](https://www.nvidia.com/en-gb/data-center/technologies/blackwell-architecture/),
[GB200 NVL multi-node tuning guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html).
