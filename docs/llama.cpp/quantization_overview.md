# Quantization Schemes in llama.cpp

## Overview

llama.cpp supports a wide range of quantization schemes for compressing model weights, spanning from
1-bit ternary encodings to full 32-bit floating point. These schemes trade off model size and inference
speed against accuracy (measured as perplexity degradation).

Quantization types fall into several families:

| Family | Types | Description |
|--------|-------|-------------|
| **Legacy** | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | Original fixed-bit quantization with simple per-block scaling |
| **K-quants** | Q2_K … Q6_K | Two-level (super-block + sub-block) quantization with quantized scales |
| **I-quants (IQ)** | IQ1_S … IQ4_XS | Importance-matrix-aware quantization using grid codebooks or non-linear mappings |
| **Ternary** | TQ1_0, TQ2_0 | Three-level {-1, 0, +1} weight encoding |
| **FP4** | MXFP4, NVFP4 | 4-bit floating-point microscaling formats |
| **Minimal** | Q1_0 | 1-bit binary quantization |
| **Non-quantized** | F16, BF16, F32 | Full-precision storage formats |
| **Intermediate** | Q8_1, Q8_K | Used internally for dot-product computation, not for model storage |

All definitions live in [`ggml/src/ggml-common.h`](ggml/src/ggml-common.h), with type metadata in
[`ggml/src/ggml.c`](ggml/src/ggml.c) and model-level quantization logic in
[`src/llama-quant.cpp`](src/llama-quant.cpp).

---

## Master Reference Table

The **Bits/Weight** (bpw) column is the effective storage cost per weight, computed as:

```
bpw = (Block Bytes × 8) ÷ Block Size
```

For quantized types, bpw is always higher than the raw quant bit-width because each block carries
scale metadata (scale factors, mins, grid indices) in addition to the packed weight values. For example,
Q4_K stores 256 weights as 4-bit integers (128 bytes) plus 16 bytes of scales, totaling 144 bytes:

```
Q4_K:  (144 bytes × 8 bits) ÷ 256 weights = 4.5 bpw
        ─────────────────     ───────────
        128 B quants          pure 4-bit data = 4.0 bpw
      +   2 B super-block scale (d)
      +   2 B super-block min (dmin)    ← scale overhead = +0.5 bpw
      +  12 B sub-block scales/mins
```

### Non-Quantized Types

| Type | Primary Use | Bits/Weight | Block Size | Block Bytes | Scale DType | Notes |
|------|-------------|:-----------:|:----------:|:-----------:|:-----------:|-------|
| F32 | Full precision | 32.0 | 1 | 4 | — | Reference precision, largest |
| F16 | High precision | 16.0 | 1 | 2 | — | IEEE 754 half-precision |
| BF16 | High precision | 16.0 | 1 | 2 | — | Brain float; better dynamic range than F16 |

### Legacy Quantization (Simple Block Scaling)

| Type | Primary Use | Bits/Weight | Quant Levels | Block Size | Block Bytes | Packing | Scale DType | Scale Direction | Scales/Block |
|------|-------------|:-----------:|:------------:|:----------:|:-----------:|---------|:-----------:|:---------------:|:------------:|
| Q1_0 | Ultra-low bit | 1.125 | 2 | 128 | 18 | 1 bit/element, 8 per byte | F16 | Symmetric | 1 |
| Q4_0 | General weight quant | 4.5 | 16 | 32 | 18 | 4-bit nibbles, 2 per byte | F16 | Symmetric | 1 |
| Q4_1 | General weight quant | 5.0 | 16 | 32 | 20 | 4-bit nibbles, 2 per byte | F16 | Asymmetric (d + min) | 1 + 1 min |
| Q5_0 | Higher quality | 5.5 | 32 | 32 | 22 | 4-bit nibbles + 1-bit high array | F16 | Symmetric | 1 |
| Q5_1 | Higher quality | 6.0 | 32 | 32 | 24 | 4-bit nibbles + 1-bit high array | F16 | Asymmetric (d + min) | 1 + 1 min |
| Q8_0 | Near-lossless | 8.5 | 256 | 32 | 34 | 1 byte per element (int8) | F16 | Symmetric | 1 |

**Dequantization formulas:**
- Symmetric: `x = d × (q − offset)`  (e.g. Q4_0: offset = 8, Q5_0: offset = 16)
- Asymmetric: `x = d × q + m`  (Q4_1, Q5_1)
- Q8_0: `x = d × q`  (no offset, signed int8)

### K-Quants (Super-Block Quantization)

All K-quants use a **256-element super-block** (`QK_K = 256`) divided into sub-blocks. Each sub-block
has its own quantized scale (and optionally min), which are themselves scaled by a shared super-block
scale factor stored in F16.

| Type | Primary Use | Bits/Weight | Quant Levels | Block Size | Sub-Blocks | Sub-Block Size | Block Bytes | Packing | Scale DType | Sub-Block Scale Bits | Scale Direction | Scales/Block |
|------|-------------|:-----------:|:------------:|:----------:|:----------:|:--------------:|:-----------:|---------|:-----------:|:--------------------:|:---------------:|:------------:|
| Q2_K | Aggressive compression | 2.625 | 4 | 256 | 16 | 16 | 84 | 2 bits/element, 4 per byte | F16 (d + dmin) | 4-bit | Asymmetric | 16 scales + 16 mins |
| Q3_K | Low-bit quality | 3.4375 | 8 | 256 | 16 | 16 | 110 | 2-bit low + 1-bit high array | F16 (d) | 6-bit | Symmetric | 16 |
| Q4_K | Balanced quality/size | 4.5 | 16 | 256 | 8 | 32 | 144 | 4-bit nibbles, 2 per byte | F16 (d + dmin) | 6-bit | Asymmetric | 8 scales + 8 mins |
| Q5_K | High quality | 5.5 | 32 | 256 | 8 | 32 | 176 | 4-bit nibbles + 1-bit high array | F16 (d + dmin) | 6-bit | Asymmetric | 8 scales + 8 mins |
| Q6_K | Very high quality | 6.5625 | 64 | 256 | 16 | 16 | 210 | 4-bit low + 2-bit high arrays | F16 (d) | 8-bit (int8) | Symmetric | 16 |

**Dequantization formulas:**
- Asymmetric (Q2_K, Q4_K, Q5_K): `x = d × scale × q − dmin × min`
- Symmetric (Q3_K, Q6_K): `x = d × scale × (q − offset)`

**Scale packing:** Q4_K and Q5_K pack 8 scales + 8 mins into 12 bytes (`K_SCALE_SIZE = 12`) using 6-bit
encoding. Q3_K packs 16 scales into 12 bytes at 6 bits each. Q2_K packs 16 scales + 16 mins into
16 bytes at 4 bits each. Q6_K uses full 8-bit signed integers (16 bytes for 16 scales).

### I-Quants (Importance-Matrix Quantization)

IQ types use lookup-table (grid/codebook) based quantization, where weight values are encoded as indices
into precomputed optimal value grids. The grids are designed to minimize quantization error weighted by
activation importance (imatrix). Most IQ types require or strongly benefit from an importance matrix.

| Type | Primary Use | Bits/Weight | Block Size | Sub-Blocks | Block Bytes | Encoding Method | Scale DType | imatrix | Scales/Block |
|------|-------------|:-----------:|:----------:|:----------:|:-----------:|-----------------|:-----------:|:-------:|:------------:|
| IQ1_S | Extreme compression | 1.5625 | 256 | 32 × 8 | 50 | Grid index (iq1s_grid, 2048 entries) + high bits | F16 | **Required** | 1 super-block + encoded in qh |
| IQ1_M | Extreme compression | 1.75 | 256 | 32 × 8 | 56 | Grid index + shift bit + 3-bit block scales | Packed in scales[] | **Required** | 8 (3-bit packed) |
| IQ2_XXS | Ultra-low bit | 2.0625 | 256 | 8 × 32 | 66 | Grid index (iq2xxs_grid) + sign bits | F16 | **Required** | 1 |
| IQ2_XS | Low-bit | 2.3125 | 256 | 8 × 32 | 74 | Grid index (iq2xs_grid) + 4-bit sub-block scales | F16 | **Required** | 8 (4-bit) |
| IQ2_S | Low-bit | 2.5625 | 256 | 8 × 32 | 82 | Grid index (iq2s_grid) + high bits + scales | F16 | **Required** | 8 |
| IQ3_XXS | Low-bit | 3.0625 | 256 | 8 × 32 | 98 | Dual grid lookup + packed signs/scales | F16 | **Required** | Packed in qs[] |
| IQ3_S | Moderate compression | 3.4375 | 256 | 4 × 64 | 110 | Grid index + explicit signs + 4 sub-block scales | F16 | Recommended | 4 (8-bit) |
| IQ4_NL | Quality 4-bit | 4.5 | 32 | 1 × 32 | 18 | Non-linear LUT (kvalues_iq4nl, 16 entries) | F16 | Recommended | 1 |
| IQ4_XS | Efficient 4-bit | 4.25 | 256 | 8 × 32 | 136 | Non-linear LUT + 6-bit sub-block scales | F16 + 6-bit | Recommended | 8 (6-bit) |

**Grid-based encoding (IQ1–IQ3):** Each group of elements is quantized by finding the closest match
in a precomputed grid (codebook). The grid index is stored instead of individual quant values. Sign bits
are stored separately. Grids are initialized at runtime via `ggml_quantize_init()`.

**Non-linear encoding (IQ4_NL, IQ4_XS):** Instead of uniform spacing, 4-bit indices map through a
non-uniform 16-entry lookup table optimized for weight distributions:

```c
kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
```

This gives better accuracy than linear Q4_0 at the same bit-width.

### Ternary Quantization

Ternary types encode weights as one of three values {-1, 0, +1}, scaled by a per-block factor.

| Type | Primary Use | Bits/Weight | Block Size | Block Bytes | Packing | Scale DType | Notes |
|------|-------------|:-----------:|:----------:|:-----------:|---------|:-----------:|-------|
| TQ1_0 | Ultra-low bit | 1.6875 | 256 | 54 | Base-3: 5 trits/byte (3⁵ = 243 < 256) + 4 trits/byte remainder | F16 | Efficient ternary packing |
| TQ2_0 | Low-bit ternary | 2.0625 | 256 | 66 | 2 bits per element, 4 per byte | F16 | Simpler but less compact |

### FP4 (4-bit Floating Point)

Hardware-oriented microscaling formats using floating-point representations at the element level.

| Type | Primary Use | Bits/Weight | Block Size | Sub-Blocks | Block Bytes | Element Format | Scale DType | Scale Direction | Scales/Block |
|------|-------------|:-----------:|:----------:|:----------:|:-----------:|----------------|:-----------:|:---------------:|:------------:|
| MXFP4 | MoE models | 4.25 | 32 | 1 × 32 | 17 | E2M1 (4-bit float) | E8M0 (uint8 shared exponent) | Shared exponent | 1 |
| NVFP4 | GPU-optimized | 4.5 | 64 | 4 × 16 | 36 | E2M1 (4-bit float) | UE4M3 (uint8 per sub-block) | Per sub-block | 4 |

### Intermediate Types (Not for Model Storage)

| Type | Primary Use | Bits/Weight | Block Size | Block Bytes | Scale DType | Notes |
|------|-------------|:-----------:|:----------:|:-----------:|:-----------:|-------|
| Q8_1 | Dot-product intermediate | 9.0 | 32 | 36 | F16 (d + sum) | Accumulates `d × Σqs[i]` for faster dot products |
| Q8_K | K-quant dot products | 9.125 | 256 | 292 | F32 | Stores per-16-element sums (`bsums`) for K-quant matmul |

### Integer Types

| Type | Primary Use | Bits/Element | Block Size | Bytes/Element |
|------|-------------|:------------:|:----------:|:-------------:|
| I8 | Token indices, metadata | 8 | 1 | 1 |
| I16 | Token indices, metadata | 16 | 1 | 2 |
| I32 | Token indices, metadata | 32 | 1 | 4 |
| I64 | Token indices, metadata | 64 | 1 | 8 |

---

## Comprehensive Comparison (Sorted by Bits/Weight)

| # | Type | Family | bpw | Block Size | Block Bytes | Quant Method | Scale Overhead | imatrix |
|---|------|--------|:---:|:----------:|:-----------:|--------------|:--------------:|:-------:|
| 1 | Q1_0 | Legacy | 1.125 | 128 | 18 | 1-bit binary | 1 × F16 | No |
| 2 | IQ1_S | I-quant | 1.5625 | 256 | 50 | Grid codebook (1-bit) | 1 × F16 + encoded | **Required** |
| 3 | TQ1_0 | Ternary | 1.6875 | 256 | 54 | Ternary base-3 packed | 1 × F16 | No |
| 4 | IQ1_M | I-quant | 1.75 | 256 | 56 | Grid codebook + shift | 8 × 3-bit packed | **Required** |
| 5 | IQ2_XXS | I-quant | 2.0625 | 256 | 66 | Grid codebook (2-bit) | 1 × F16 | **Required** |
| 6 | TQ2_0 | Ternary | 2.0625 | 256 | 66 | Ternary 2-bit packed | 1 × F16 | No |
| 7 | IQ2_XS | I-quant | 2.3125 | 256 | 74 | Grid codebook + sub-scales | 1 × F16 + 8 × 4-bit | **Required** |
| 8 | IQ2_S | I-quant | 2.5625 | 256 | 82 | Grid codebook + scales | 1 × F16 + 8 × 8-bit | **Required** |
| 9 | Q2_K | K-quant | 2.625 | 256 | 84 | 2-bit uniform + quantized scales | 2 × F16 + 16 × 4-bit | Recommended |
| 10 | IQ3_XXS | I-quant | 3.0625 | 256 | 98 | Dual grid codebook | 1 × F16 + packed | **Required** |
| 11 | Q3_K | K-quant | 3.4375 | 256 | 110 | 3-bit uniform + quantized scales | 1 × F16 + 16 × 6-bit | Recommended |
| 12 | IQ3_S | I-quant | 3.4375 | 256 | 110 | Grid codebook + signs + scales | 1 × F16 + 4 × 8-bit | Recommended |
| 13 | MXFP4 | FP4 | 4.25 | 32 | 17 | E2M1 micro-float | 1 × E8M0 | No |
| 14 | IQ4_XS | I-quant | 4.25 | 256 | 136 | Non-linear LUT + sub-scales | 1 × F16 + 8 × 6-bit | Recommended |
| 15 | Q4_0 | Legacy | 4.5 | 32 | 18 | 4-bit uniform symmetric | 1 × F16 | Recommended |
| 16 | Q4_K | K-quant | 4.5 | 256 | 144 | 4-bit uniform + quantized scales | 2 × F16 + 8 × 6-bit | Recommended |
| 17 | IQ4_NL | I-quant | 4.5 | 32 | 18 | Non-linear 16-entry LUT | 1 × F16 | Recommended |
| 18 | NVFP4 | FP4 | 4.5 | 64 | 36 | E2M1 micro-float | 4 × UE4M3 | No |
| 19 | Q4_1 | Legacy | 5.0 | 32 | 20 | 4-bit uniform asymmetric | 1 × F16 + 1 × F16 min | Recommended |
| 20 | Q5_0 | Legacy | 5.5 | 32 | 22 | 5-bit uniform symmetric | 1 × F16 | Recommended |
| 21 | Q5_K | K-quant | 5.5 | 256 | 176 | 5-bit uniform + quantized scales | 2 × F16 + 8 × 6-bit | Recommended |
| 22 | Q5_1 | Legacy | 6.0 | 32 | 24 | 5-bit uniform asymmetric | 1 × F16 + 1 × F16 min | Recommended |
| 23 | Q6_K | K-quant | 6.5625 | 256 | 210 | 6-bit uniform + int8 scales | 1 × F16 + 16 × 8-bit | Recommended |
| 24 | Q8_0 | Legacy | 8.5 | 32 | 34 | 8-bit uniform symmetric | 1 × F16 | No |
| 25 | BF16 | Float | 16.0 | 1 | 2 | Brain floating point | — | — |
| 26 | F16 | Float | 16.0 | 1 | 2 | IEEE 754 half-precision | — | — |
| 27 | F32 | Float | 32.0 | 1 | 4 | IEEE 754 single-precision | — | — |

---

## K-Quant S/M/L Mixing Strategy

The S (Small), M (Medium), and L (Large) variants of K-quants share the same underlying `ggml_type`
but differ in **per-tensor type assignment**. Sensitive tensors (attention value projections, FFN down
projections, output layer) are selectively upgraded to higher-precision types.

### Tensor Sensitivity Hierarchy

Tensors are ranked by quantization sensitivity (most sensitive first):

1. **output.weight** — usually kept at Q6_K or higher
2. **attn_v.weight** — attention value projection; most sensitive in attention
3. **ffn_down.weight** — FFN down-projection
4. **attn_output.weight** — attention output
5. **attn_q.weight / attn_k.weight** — less sensitive
6. **ffn_up.weight / ffn_gate.weight** — least sensitive, use base type

### Per-Tensor Mixing Examples (Q4_K variants)

| Tensor Category | Q4_K_S | Q4_K_M |
|-----------------|--------|--------|
| Base type | Q4_K | Q4_K |
| output.weight | Q6_K | Q6_K |
| token_embd.weight | Q4_K | Q4_K |
| attn_v (sensitive layers) | Q5_K | Q6_K |
| attn_v (other layers) | Q4_K | Q4_K |
| ffn_down (sensitive layers) | Q5_K | Q6_K |
| ffn_down (other layers) | Q4_K | Q4_K |
| 1D tensors (norms, biases) | F32 | F32 |

"Sensitive layers" are determined by a heuristic: first ⅛ of layers, last ⅛ of layers, and every 3rd
layer in the middle section.

The `--pure` flag disables all mixing and quantizes every 2D tensor to the base type.

---

## Importance Matrix (imatrix)

The importance matrix captures per-weight activation magnitudes from a calibration dataset. During
quantization, it provides per-element weights that guide the optimizer to preserve more important
weights at higher precision.

### imatrix Requirements by Type

| Requirement | Types |
|-------------|-------|
| **Hard required** (quantization will fail without it) | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, Q2_K_S |
| **Strongly recommended** | All other quantized types |
| **Not applicable** | F16, BF16, F32, integer types |

Generate an importance matrix with:

```bash
./llama-imatrix -m model-f16.gguf -f calibration_data.txt -o imatrix.dat
./llama-quantize --imatrix imatrix.dat model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

---

## Block Structure Details

### Legacy Blocks (32-element)

```
┌─────────────────────────────────────────────┐
│ block_q4_0 (18 bytes, 32 elements)          │
├──────────┬──────────────────────────────────┤
│  d (F16) │  qs[16] — packed 4-bit nibbles   │
│  2 bytes │  16 bytes (2 per byte)           │
└──────────┴──────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ block_q5_0 (22 bytes, 32 elements)                   │
├──────────┬────────────┬──────────────────────────────┤
│  d (F16) │  qh[4]     │  qs[16] — 4-bit nibbles      │
│  2 bytes │  high bits │  16 bytes                     │
└──────────┴────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────┐
│ block_q8_0 (34 bytes, 32 elements)             │
├──────────┬─────────────────────────────────────┤
│  d (F16) │  qs[32] — int8 values               │
│  2 bytes │  32 bytes (1 per element)            │
└──────────┴─────────────────────────────────────┘
```

### K-Quant Super-Block (256-element)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ block_q4_K (144 bytes, 256 elements = 8 sub-blocks × 32)                    │
├──────────┬───────────┬──────────────┬────────────────────────────────────────┤
│  d (F16) │ dmin (F16)│ scales[12]   │  qs[128] — packed 4-bit nibbles       │
│  2 bytes │  2 bytes  │ 6-bit packed │  128 bytes (2 per byte)               │
│          │           │ 8 scales +   │                                        │
│          │           │ 8 mins       │                                        │
└──────────┴───────────┴──────────────┴────────────────────────────────────────┘
  ↑ super-block scale    ↑ sub-block scales/mins (quantized)

  Dequant for sub-block j: x[i] = d × scale[j] × q[i] − dmin × min[j]
```

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ block_q6_K (210 bytes, 256 elements = 16 sub-blocks × 16)                   │
├────────────────┬───────────────┬───────────────────┬─────────┐              │
│  ql[128]       │  qh[64]       │  scales[16]       │  d (F16)│              │
│  lower 4 bits  │  upper 2 bits │  int8 per sub-blk │  2 bytes│              │
│  128 bytes     │  64 bytes     │  16 bytes         │         │              │
└────────────────┴───────────────┴───────────────────┴─────────┘

  Dequant: x[i] = d × scale[j] × (reconstruct_6bit(ql, qh) − 32)
```

### IQ Grid-Based Block

```
┌──────────────────────────────────────────────────────────────────────────┐
│ block_iq2_xxs (66 bytes, 256 elements)                                   │
├──────────┬───────────────────────────────────────────────────────────────┤
│  d (F16) │  qs[32] — uint16 grid indices (8 groups of 32 elements)      │
│  2 bytes │  64 bytes — each uint16 encodes grid index + sign bits       │
└──────────┴───────────────────────────────────────────────────────────────┘

  Dequant: grid = iq2xxs_grid[index]; x[i] = d × grid[i] × sign[i]
```

### Ternary Block

```
┌─────────────────────────────────────────────────────────────────────┐
│ block_tq1_0 (54 bytes, 256 elements)                                │
├────────────────────────┬──────────┬─────────┐                       │
│  qs[48] — base-3 pack  │  qh[4]   │  d (F16)│                       │
│  5 trits per byte      │  4/byte  │  2 bytes│                       │
│  (3⁵ = 243 < 256)      │          │         │                       │
└────────────────────────┴──────────┴─────────┘

  Values: {-1, 0, +1} × d
```

---

## Choosing a Quantization Type

### Quick Guide

| Priority | Recommended Type | bpw | Typical Llama-3-8B Size |
|----------|-----------------|:---:|:-----------------------:|
| Maximum quality | F16 / BF16 | 16.0 | ~15 GiB |
| Near-lossless | Q8_0 | 8.5 | ~8 GiB |
| High quality | Q6_K | 6.6 | ~6.1 GiB |
| **Balanced (default)** | **Q4_K_M** | **4.9** | **~4.6 GiB** |
| Good quality, smaller | Q4_K_S | 4.7 | ~4.4 GiB |
| Compact | Q3_K_M | 4.0 | ~3.7 GiB |
| Very compact (with imatrix) | IQ3_M | 3.8 | ~3.5 GiB |
| Aggressive (with imatrix) | IQ2_M | 2.7 | ~2.7 GiB |
| Extreme (with imatrix) | IQ1_M | 1.8 | ~2.0 GiB |

