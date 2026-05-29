# KV Cache Implementation in llama.cpp

## 1. KV Cache Format and Data Structures

The KV cache is implemented in `src/llama-kv-cache.h` and `src/llama-kv-cache.cpp`. The core class is **`llama_kv_cache`** which implements the `llama_memory_i` interface.

### Per-Layer Storage

Each cached model layer gets a `kv_layer` struct (`src/llama-kv-cache.h`):

```cpp
struct kv_layer {
    uint32_t il;                          // model layer index
    ggml_tensor * k;                      // 3D backing tensor for keys
    ggml_tensor * v;                      // 3D backing tensor for values
    std::vector<ggml_tensor *> k_stream;  // 2D views per stream
    std::vector<ggml_tensor *> v_stream;  // 2D views per stream
};
```

The backing tensors are allocated as 3D GGML tensors (`src/llama-kv-cache.cpp`):

```cpp
ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
```

| Dimension | K tensor | V tensor |
|-----------|----------|----------|
| `ne[0]` | `n_embd_k_gqa` (all KV heads × head dim) | `n_embd_v_gqa` |
| `ne[1]` | `kv_size` (number of cache slots) | `kv_size` |
| `ne[2]` | `n_stream` (1 if unified, else n_seq_max) | `n_stream` |

The cache is organized as a **ring buffer of slots per layer per stream**. Each slot holds one token's full K (or V) embedding across all KV heads.

### V Layout Depends on Flash Attention

- **Flash attention ON** → `v_trans = false` → V stored as `[n_embd_head_v, n_head_kv, n_kv]` (head-major, same layout as K)
- **Flash attention OFF** → `v_trans = true` → V stored **transposed** for classic matmul: `[n_kv, n_head_kv, n_embd_head_v]`

### Cell Metadata (separate from tensors)

`llama_kv_cells` (in `src/llama-kv-cells.h`) tracks per-slot metadata — position, sequence IDs, pending RoPE shifts — independently from the tensor data. This drives slot allocation, multi-sequence sharing, and SWA eviction.

### Variants

- **`llama_kv_cache_iswa`**: wraps two `llama_kv_cache` instances — one for base (non-SWA) layers and one for sliding-window layers
- **MLA models**: only K cache is allocated; V is derived from K

---

## 2. Supported Data Types

The user-facing allowlist is in `common/arg.cpp`:

```cpp
const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};
```

| Type | Category | Block size | Bytes/block | Bits/element (approx) |
|------|----------|------------|-------------|----------------------|
| `f32` | Float | 1 | 4 | 32 |
| `f16` | Float | 1 | 2 | 16 |
| `bf16` | Float | 1 | 2 | 16 |
| `q8_0` | Quantized | 32 | 34 | 8.5 |
| `q4_0` | Quantized | 32 | 18 | 4.5 |
| `q4_1` | Quantized | 32 | 20 | 5 |
| `q5_0` | Quantized | 32 | 22 | 5.5 |
| `q5_1` | Quantized | 32 | 24 | 6 |
| `iq4_nl` | Quantized | 32 | 18 | 4.5 |

### Block Layouts (from `ggml/src/ggml-common.h`)

All quantized types used for KV cache share a block size of **32 elements** (defined as `QK4_0`, `QK4_1`, `QK5_0`, `QK5_1`, `QK8_0`, `QK4_NL` — all equal to 32). This is important because `n_embd_head_k` and `n_embd_head_v` must be divisible by this block size when using quantized cache with flash attention.

#### `block_q4_0` — 18 bytes per 32 elements (4.5 bpw)

```cpp
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta (2 bytes, scale factor)
    uint8_t qs[QK4_0 / 2]; // nibbles / quants (16 bytes, two 4-bit values packed per byte)
} block_q4_0;
```

Asymmetric quantization with a single FP16 scale per block. Each weight is stored as a 4-bit unsigned integer; dequantized as `x[i] = (qs[i] - 8) * d`.

#### `block_q4_1` — 20 bytes per 32 elements (5 bpw)

```cpp
#define QK4_1 32
typedef struct {
    ggml_half d; // delta (2 bytes, scale)
    ggml_half m; // min   (2 bytes, offset)
    uint8_t qs[QK4_1 / 2]; // nibbles / quants (16 bytes)
} block_q4_1;
```

Affine quantization with both scale and minimum. Dequantized as `x[i] = qs[i] * d + m`. The extra 2 bytes for the minimum give slightly better accuracy than `q4_0`.

#### `block_q5_0` — 22 bytes per 32 elements (5.5 bpw)

```cpp
#define QK5_0 32
typedef struct {
    ggml_half d;           // delta (2 bytes)
    uint8_t qh[4];         // 5th bit of quants (4 bytes, packed as a 32-bit mask)
    uint8_t qs[QK5_0 / 2]; // nibbles / quants (16 bytes, lower 4 bits)
} block_q5_0;
```

5-bit quantization. The lower 4 bits are packed in `qs` (same as q4_0), the 5th high bit for each element is stored in `qh` as a bitmask.

#### `block_q5_1` — 24 bytes per 32 elements (6 bpw)

```cpp
#define QK5_1 32
typedef struct {
    ggml_half d; // delta (2 bytes)
    ggml_half m; // min   (2 bytes)
    uint8_t qh[4];         // 5th bit of quants (4 bytes)
    uint8_t qs[QK5_1 / 2]; // nibbles / quants (16 bytes)
} block_q5_1;
```

5-bit affine quantization with scale + minimum. Like `q5_0` but with an additional min offset for better accuracy.

#### `block_q8_0` — 34 bytes per 32 elements (8.5 bpw)

```cpp
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta (2 bytes)
    int8_t  qs[QK8_0]; // quants (32 bytes, one signed byte per element)
} block_q8_0;
```

8-bit quantization with a single FP16 scale. Each element is a full signed byte — the highest quality quantized type for KV cache, and the fastest for dot products (simple int8 multiply-accumulate).

#### `block_iq4_nl` — 18 bytes per 32 elements (4.5 bpw)

```cpp
#define QK4_NL 32
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL/2]; // nibbles / quants (16 bytes)
} block_iq4_nl;
```

Same size as `q4_0` but uses a **non-linear** lookup table for dequantization instead of simple `(qs - 8) * d`. The 4-bit indices map to 16 optimally chosen values, giving better accuracy than `q4_0` at the same memory cost. The "i" prefix stands for "importance matrix" quantization.

**K and V can have different types.** They are configured independently via:
- C API: `llama_context_params.type_k` / `llama_context_params.type_v` (default: both `GGML_TYPE_F16`)
- CLI: `--cache-type-k TYPE` / `--cache-type-v TYPE`

**Important**: while the KV cache data structure and allocation logic are identical across all backends (a single `llama_kv_cache` class), the **op support** for writing quantized data (`set_rows`) and reading it during flash attention varies by backend. See section 4 for the per-backend breakdown.

---

## 3. Quantization and Dequantization: Where They Happen

**Quantization happens on WRITE. Dequantization happens INSIDE the attention kernel (fused). There are no separate dequant graph nodes.**

### Write Path (Quantize on Store)

The flow in `src/llama-graph.cpp` `build_attn()`:

```
Q/K/V linear projections (F32) → RoPE on K,Q → [optional Hadamard rotation] → ggml_set_rows → KV cache
```

1. `k_cur` and `v_cur` come out of linear projections + RoPE as **F32**
2. Optional **Walsh-Hadamard rotation** is applied when the cache type is quantized and head dim is divisible by 64 (reduces quantization error)
3. `cpy_k()` / `cpy_v()` call **`ggml_set_rows`** which scatter-writes F32 data into the cache tensor, **quantizing on the fly**:
   - **CPU backend** (`ggml/src/ggml-cpu/ops.cpp`): uses `from_float` type trait to quantize each row
   - **CUDA/HIP backend** (`ggml/src/ggml-cuda/set-rows.cu`): dedicated GPU kernels like `quantize_f32_q4_0_block` etc.

There is **no separate quantization node** in the computation graph. The `set_rows` op fuses the scatter + type conversion.

### Read Path (Dequant Inside Attention)

`get_k()` / `get_v()` return **views** into the cache tensors — the quantized type is preserved, no dequantization here.

Then these quantized views go directly into the attention op:

#### Flash Attention Path

```cpp
ggml_flash_attn_ext(q, k_quantized_view, v_quantized_view, mask, scale)
```

Inside the flash attention kernel:
- **K**: quantized dot products are computed **directly** using `vec_dot` functions (e.g., `dp4a`-based for Q4_0). Q is converted to Q8_1 in registers for the dot product. **No full K dequantization buffer.**
- **V**: dequantized **on the fly per row** during the weighted accumulation step (unless V is F16, which is accumulated natively)

#### Non-Flash (Classic) Attention Path

```cpp
kq = ggml_mul_mat(k, q)        // scores
kq = softmax(kq)
kqv = ggml_mul_mat(v, kq)      // weighted sum
```

`ggml_mul_mat` handles quantized operands internally via its own vec_dot/dequant routines. **Quantized V is NOT allowed** on this path — it requires flash attention.

### Summary Table

| Stage | Operation | What happens |
|-------|-----------|-------------|
| **Store K/V** | `ggml_set_rows` | F32 → quantize into cache type (fused) |
| **Read K/V** | `ggml_view_4d` | Returns quantized view (no dequant) |
| **Flash Attn K** | Inside kernel | vec_dot on quantized K directly |
| **Flash Attn V** | Inside kernel | On-the-fly dequant per row during accumulation |
| **Classic Attn K** | `ggml_mul_mat` | Internal dequant in matmul |
| **Classic Attn V** | Not allowed quantized | Must be float with non-flash path |
| **K-shift (RoPE)** | Explicit in graph | dequant → rotate → re-quantize |

---

## 4. Per-Backend KV Cache Support

The KV cache data structure and allocation logic are **identical across all backends** — there is a single `llama_kv_cache` class. What differs is **op support**: which quantized types each backend can handle for writing (`set_rows`) and reading during flash attention (`flash_attn_ext`).

### 4.1 set_rows (KV Cache Write) Support

All backends follow the same pattern: F32 source → quantize/copy into destination rows indexed by I32/I64.

| Backend | Implementation | Supported dst types |
|---------|---------------|-------------------|
| **CPU** | `ggml-cpu/ops.cpp` via `from_float` trait | All types: f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl, and more |
| **CUDA/HIP** | `ggml-cuda/set-rows.cu` per-type kernels | f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl |
| **Metal** | `ggml-metal.metal` `kernel_set_rows_q32` | f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl |
| **Vulkan** | `copy_to_quant.comp` (SET_ROWS=1) | f32, f16, bf16, q1_0, q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl |
| **SYCL** | `ggml-sycl/set_rows.cpp` | f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl |
| **CANN** (Ascend) | `aclnn_ops.cpp` cast + InplaceIndexCopy | **f32, f16, bf16 only — no quantized write** |

### 4.2 Flash Attention (KV Cache Read) Support

| Backend | FA implementation | K/V quant types supported | K/V type pairing | Build flags |
|---------|-------------------|--------------------------|------------------|-------------|
| **CPU** | `ggml-cpu/ops.cpp` generic | Any type with `vec_dot` (K) / `to_float` (V): all quants | **Mixed K/V allowed** | — |
| **CUDA** | fattn-vec, fattn-tile, fattn-mma-f16, fattn-wmma-f16 | Default: f16, f32, bf16, q4_0, q8_0. With `GGML_CUDA_FA_ALL_QUANTS`: also q4_1, q5_0, q5_1 | Default: **K.type == V.type**. With `GGML_CUDA_FA_ALL_QUANTS`: mixed allowed | `GGML_CUDA_FA_ALL_QUANTS` |
| **HIP (AMD)** | Same CUDA code compiled with `GGML_USE_HIP` | Same as CUDA | Same as CUDA | `GGML_CUDA_FA_ALL_QUANTS`, `GGML_HIP_ROCWMMA_FATTN` |
| **Metal** | `kernel_flash_attn_ext` templates | f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0. **No iq4_nl** | **K.type == V.type** | — |
| **Vulkan** | scalar, coopmat1, coopmat2 shaders | f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q1_0 (cm2 only). **No iq4_nl, no bf16** | **Mixed K/V allowed** | — |
| **SYCL** | tile (f16/f32) + vec (quant) | Default: f16, f32, q4_0, q8_0. With `GGML_SYCL_FA_ALL_QUANTS`: q4_1, q5_0, q5_1. **No bf16** | Default: **K.type == V.type**. With flag: mixed allowed | `GGML_SYCL_FA_ALL_QUANTS` |
| **CANN** (Ascend) | ACL `FusedInferAttentionScoreV2` | **f16 only** | f16 == f16 only | — |

### 4.3 AMD GPU (HIP/ROCm) Details

The HIP backend compiles the **same CUDA source files** with `GGML_USE_HIP` defined. The shim layer in `ggml/src/ggml-cuda/vendors/hip.h` maps CUDA APIs to HIP equivalents.

Three FA kernel families exist, with different quantized KV support:

| Kernel | Quantized KV? | Used on AMD when |
|--------|--------------|------------------|
| **`fattn-vec`** | Yes (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16, F16) | Small Q batch (decode, ≤2 columns) — **primary quant path** |
| **`fattn-tile`** | No (F16 `half2` only) | Larger batches, F16 KV cache |
| **`fattn-mma-f16`** | No (F16, uses MFMA on CDNA / WMMA on RDNA4) | Large batches on MI300X, RDNA4 |

For typical **decode** workloads (generating one token at a time), kernel selection favors the **vec kernel**, which handles quantized KV natively with AMD-specific tuning:

```cpp
// fattn-vec.cuh
#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;  // CDNA
#endif
#endif
```

| AMD Arch | FA for F16 KV | FA for quantized KV |
|----------|--------------|-------------------|
| **CDNA** (MI100/MI210/MI300) | MFMA (mma-f16) for large batch; tile otherwise | **vec** kernel |
| **RDNA4** | MMA/WMMA or tile | **vec** kernel |
| **RDNA3** | Tile (WMMA disabled by default) | **vec** kernel |
| **RDNA1/2** | Tile | **vec** kernel |

AMD-relevant build flags:
- `GGML_CUDA_FA_ALL_QUANTS=ON`: compiles all K/V quant combinations (default: only 4 matching pairs)
- `GGML_HIP_ROCWMMA_FATTN=ON`: enables rocWMMA-based WMMA flash attention on RDNA3/4/CDNA
- Requires ROCm/HIP >= 6.1

### 4.4 Practical Implications

1. **Same cache object, different op coverage** — switching GPU backend does not change the cache structure; it changes which ops the backend's scheduler can execute. Unsupported ops fall back to CPU.

2. **CANN is the outlier** — Ascend NPUs cannot use quantized KV cache at all (no quantized `set_rows`, F16-only FA).

3. **`iq4_nl` has a gap** — writable on all GPU backends, but **not readable via FA on Metal or Vulkan**. Only CPU, CUDA/HIP, and SYCL support `iq4_nl` in flash attention.

4. **Mixed K/V types** — natively supported on CPU and Vulkan. On CUDA/HIP/SYCL requires a build flag. Metal always requires matching types.

5. **Layout is FA-driven, not backend-driven** — `v_trans` (V transposition) is set by `!cparams.flash_attn`, not by the backend. All backends use the same tensor shapes.

---

## 5. End-to-End Flow Diagram

```
Token embedding + transformer layers
  │
  ▼
Q/K/V linear projections + RoPE (F32 on device)
  │
  ├── [optional] Hadamard rotation (when quant cache + head_dim % 64 == 0)
  │
  ├──► ggml_set_rows ──► K cache (e.g. Q4_0 on GPU)     ← QUANTIZE HERE
  ├──► ggml_set_rows ──► V cache (e.g. Q4_0 on GPU)     ← QUANTIZE HERE
  │
  ▼
get_k() / get_v() ──► ggml_view_4d (quantized views, no copy)
  │
  ▼
ggml_flash_attn_ext(Q_f16, K_q4_0, V_q4_0, mask)
  │
  ├─ K: vec_dot on quantized data directly (dp4a-style)
  ├─ V: dequant per-row on the fly during weighted accumulation
  │
  ▼
Attention output (F32) → output projection → next layer
```

---

## 6. Is KV Cache Paged?

**No, llama.cpp does not use paged KV cache** (unlike vLLM's PagedAttention). It uses a flat ring buffer with scatter writes.

### What llama.cpp uses: a flat ring buffer with scatter writes

The KV cache is a **single contiguous pre-allocated tensor per layer** — dimensions `[n_embd_k_gqa, kv_size, n_stream]`. There is no concept of "pages" or "blocks" in the memory management sense.

Slot allocation (`find_slot()` in `src/llama-kv-cache.cpp`) works as a **linear scan** from a head pointer through a flat array of cells. Each cell corresponds to one row (one token) in the pre-allocated tensor. The scan looks for empty (or evictable) cells one at a time:

```cpp
// src/llama-kv-cache.cpp — find_slot()
while (true) {
    // ... wrap around if head reaches end ...
    for (uint32_t i = 0; i < n_test; i++) {
        const auto idx = head_cur;
        // ...
        bool can_use = cells.is_empty(idx);
        // ... or SWA-evictable ...
        if (can_use) {
            res.idxs[s].push_back(idx);
        }
    }
}
```

The found cell indices (which can be **non-contiguous**) are then passed to `ggml_set_rows`, which scatter-writes the new K/V data into those specific rows of the flat tensor. This is why `set_rows` takes an index array — it handles arbitrary placement into the buffer.

### Comparison with paged KV cache (vLLM-style)

| Aspect | vLLM PagedAttention | llama.cpp |
|--------|---------------------|-----------|
| **Memory unit** | Fixed-size pages (e.g. 16 tokens each) | Individual token slots (1 token each) |
| **Allocation** | Page table maps logical → physical blocks | Flat ring buffer, cell index array |
| **Fragmentation** | No internal fragmentation (page-aligned) | Can have scattered empty cells |
| **Sharing** | Pages shared across sequences via refcount | Cells track sequence membership via bitmask |
| **Indirection** | Page table lookup in attention kernel | Direct index into flat tensor via `set_rows` |
| **Pre-allocation** | Allocates pages on demand | Entire `kv_size` tensor allocated upfront |

### Why no paging?

llama.cpp's approach is simpler and avoids the complexity of page tables and block management. The `ggml_set_rows` scatter-write pattern gives it the flexibility to place tokens in arbitrary positions without needing contiguous allocation. The tradeoff is that the entire cache must be pre-allocated (you set `kv_size` = context length upfront), whereas paged systems can grow incrementally and share memory more efficiently across many concurrent sequences.

---

## 7. Key Files Reference

| File | What to look at |
|------|----------------|
| `src/llama-kv-cache.h` | Class definitions, `kv_layer`, APIs |
| `src/llama-kv-cache.cpp` | Allocation, `cpy_k`/`cpy_v`, `get_k`/`get_v`, state I/O |
| `src/llama-kv-cells.h` | Cell metadata (positions, sequences, ring buffer) |
| `src/llama-graph.cpp` | `build_attn()`, `build_attn_mha()` — the graph construction |
| `include/llama.h` | `type_k`, `type_v` in `llama_context_params` |
| `common/arg.cpp` | Allowed KV cache types |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Vec flash attn — quantized K/V kernel (also used on HIP) |
| `ggml/src/ggml-cuda/fattn.cu` | Kernel dispatch and selection logic |
| `ggml/src/ggml-cuda/set-rows.cu` | GPU-side quantization on KV store |
| `ggml/src/ggml-cpu/ops.cpp` | CPU flash attn with quantized KV (~line 8214+) |
| `ggml/src/ggml-hip/CMakeLists.txt` | HIP build setup (compiles CUDA sources) |
