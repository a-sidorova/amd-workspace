# Quantization in Deep Learning

Quantization is the process of representing model weights, activations, or both with fewer bits than the usual `FP32` format.

Reducing a model’s precision offers several significant benefits:

- Smaller model size: Lower-precision data types require less storage space. An int8 model, for example, is roughly 4 times smaller than its float32 counterpart.
- Faster inference: Operations on lower-precision data types, especially integers, can be significantly faster on compatible hardware (CPUs and GPUs often have specialized instructions for int8 operations). This leads to lower latency.
- Reduced energy consumption: Faster computations and smaller memory transfers often translate to lower power usage.

## 1. Purpose of Quantization

### Why quantization is useful

Neural networks are often limited not only by arithmetic throughput, but also by:

- model size in memory
- memory bandwidth
- cache capacity
- latency of moving weights and activations to compute units

Quantization helps because fewer bits are used per value. For example:

- `FP32` uses 32 bits per value
- `INT8` uses 8 bits per value. Can reduce raw storage by about `4x` compared with `FP32`.
- `INT4` uses 4 bits per value. Can reduce raw storage by about `8x` compared with `FP32`.

This often leads to faster inference, especially for large language models and vision models where moving data is expensive.

### Pros

- Smaller model checkpoints.
- Lower memory bandwidth usage.
- Better cache efficiency.
- Higher throughput on hardware with quantized kernels.
- Lower serving cost and power usage.

### Cons

- Quantization introduces approximation error.
- Some layers are more sensitive than others.
- Accuracy may drop if calibration or scales are poor.
- Lower-bit formats often need special kernels.
- Real speedup depends on hardware and software support.

## 2. Quantization Schemes

The central idea is to map a real value `x` to a lower-precision stored value `q`, then reconstruct an approximation `x_hat` when the value is used.

### General affine quantization

The most common formula is:

```markdown
q = clip(round(x / s) + z, q_min, q_max)
x_hat = s * (q - z)
```

Where:

- `s` is the scale
- `z` is the zero-point
- `q` is the quantized integer value
- `x_hat` is the dequantized approximation

### Symmetric quantization

In symmetric quantization, zero is represented exactly and the integer range is centered around zero. Usually:

```markdown
z = 0
q = clip(round(x / s), q_min, q_max)
x_hat = s * q
```

Example signed ranges:

- `INT8`: `q in [-128, 127]`
- `INT4`: `q in [-8, 7]`

#### Why use symmetric quantization

- Simpler math.
- Common for weights.
- Efficient in matrix multiplication kernels.

#### Drawback

If data is not centered around zero, part of the integer range may be wasted.

### Asymmetric quantization

In asymmetric quantization, the integer range is shifted by a zero-point:

```markdown
q = clip(round(x / s) + z, q_min, q_max)
x_hat = s * (q - z)
```

This is useful when values are not symmetric around zero, for example activations after `ReLU`.

#### Why use asymmetric quantization

- Better use of the available code range when data is skewed.
- Common for activations in post-training quantization.

#### Drawback

- Extra zero-point handling can make kernels more complicated.

## 3. Static and Dynamic Quantization

These terms describe how quantization parameters are chosen at runtime.

### Static quantization

In static quantization, scales and zero-points are determined before inference, usually by calibration on representative data.

At inference time weights are already quantized and activation quantization parameters are already known.

#### Pros

- Usually better runtime efficiency.
- Good for deployment on fixed hardware.

#### Cons

- Needs calibration data.
- Bad calibration can hurt accuracy.

### Dynamic quantization

In dynamic quantization, some quantization parameters, usually for activations, are computed from the current input during inference.

Typical flow:

- weights are pre-quantized
- activation range is measured on the fly
- activation scale is computed at runtime

#### Pros

- Easy to apply.
- No calibration dataset is needed in the simplest case.

#### Cons

- Runtime overhead for computing scales.
- Usually less efficient than fully static quantization.

### Quick summary

- Static quantization: precompute activation parameters.
- Dynamic quantization: compute some activation parameters during inference.

## 4. Quantization Granularity

Granularity describes how broadly a single quantization parameter is shared.

The most common choices are:

- per-tensor
- per-channel
- per-group
- per-token in some activation quantization methods

### Per-tensor quantization

Per-tensor quantization uses one scale and one zero-point for the whole tensor:

```markdown
q_i = clip(round(x_i / s) + z, q_min, q_max)
x_hat_i = s * (q_i - z)
```

This is often acceptable for some activations, but can be too coarse for weights.

### Per-channel quantization

Per-channel quantization uses a different scale for each channel. For a weight matrix, this is often done per output channel:

```markdown
q_{c,i} = clip(round(x_{c,i} / s_c) + z_c, q_min, q_max)
x_hat_{c,i} = s_c * (q_{c,i} - z_c)
```

Where `c` is the channel index.

### Per-group quantization

Per-group quantization sits between per-channel and per-element quantization. A small group of values shares one scale:

```markdown
q_i = clip(round(x_i / s_g) + z_g, q_min, q_max)
x_hat_i = s_g * (q_i - z_g)
```

Per-group quantization is especially popular for low-bit LLM weights because it gives a good balance between: ccuracy, metadata cost and kernel complexity.

### Per-token quantization

In some activation quantization systems scales are computed separately for each token or micro-batch item:

```markdown
q_{t,i} = clip(round(x_{t,i} / s_t) + z_t, q_min, q_max)
x_hat_{t,i} = s_t * (q_{t,i} - z_t)
```

Where `t` is the token or sample index.

This can better adapt to changing activation distributions, but it adds runtime work.

### Quick intuition

- per-tensor: cheapest, least accurate
- per-channel: common high-quality choice for weights
- per-group: strong compromise for low-bit models
- per-token: adaptive for activations, but more expensive at runtime

## 5. Quantization Techniques

There are two main types of quantization techniques.

- Post-Training Quantization (`PTQ`): Quantization is applied after the model is fully trained. This is cheaper and faster to apply
- Quantization-Aware Training (`QAT`): Quantization effects are simulated during training by inserting “fake quantization” ops that simulate the rounding errors of quantization. This lets the model adapt to quantization, and usually results in better accuracy, especially at lower bit-widths. This is usually more accurate, but more expensive

## 6. Quantized and Reduced-Precision Data Types

Not all low-precision formats are quantization formats in the strictest sense. Some are true integer quantization formats, and some are reduced-precision floating-point formats. In practice, both are discussed together because both reduce memory and compute cost.

Common formats include:

- `INT8`
- `UINT8`
- `INT6`
- `INT4`
- `INT3`
- `INT2`
- `FP8`
- `MXFP8`
- `MXFP6`
- `MXFP4`
- `FP6` in some research or vendor-specific flows
- `FP4`
- `NF4`
- `BF16`
- `FP16`

#### Important note

- `BF16` and `FP16` are reduced-precision floating-point formats, not usually called quantized integer formats.
- `MXFP*`, `FP4`, `FP6`, and `NF4` are not always standardized the same way across papers and hardware vendors.

## Integer Quantization Formats

### `INT8`

- 8 bits per value
- 1 value per byte
- Usually stored as signed integer in two's complement

To read `INT8` value and dequantize back to approximate `FP32`:

```markdown
x_hat = s * (q - z)
```

### `UINT8`

- 8 bits per value
- 1 value per byte
- Unsigned range usually `0..255`

To read `UINT8` value and dequantize back to approximate `FP32`:

```markdown
x_hat = s * (q - z)
```

Usually `z != 0 `.

### `INT6`

- 6 bits per value
- Values are usually bit-packed
- 4 values take `4 * 6 = 24` bits = 3 bytes

Typical runtime steps:

1. Load the bytes containing the 6-bit field.
2. Extract the correct 6 bits.
3. Sign-extend to a wider integer type.
4. Apply scale and zero-point if needed.

Formula:

```markdown
x_hat = s_g * (q - z_g)
```

or for symmetric group quantization:

```markdown
x_hat = s_g * q
```

### `INT4`

- 4 bits per value
- Usually 2 values per byte

Typical runtime steps:

1. Load one byte.
2. Extract high nibble or low nibble.
3. Sign-extend from 4 bits if signed.
4. Apply scale.

Formula:

```markdown
x_hat = s_g * (q - z_g)
```

This is one of the most common weight-only quantization formats in LLM inference.

## Floating Low-Precision Formats

Floating low-precision formats do not usually use the integer affine formula above. Instead, the stored bits directly represent:

- sign
- exponent
- mantissa (fraction)

For a normal floating-point number:

```markdown
x = (-1)^s * 2^(e - bias) * (1 + m / 2^p)
```

Where:

- `s` is the sign bit
- `e` is the stored exponent
- `bias` is the exponent bias
- `m` is the mantissa integer field
- `p` is the number of mantissa bits

### `FP16`

- 16 bits per value
- layout: `1` sign bit, `5` exponent bits, `10` mantissa bits
- 2 bytes per value

`FP16` is reduced precision, not usually called quantization, but it is still a major low-precision format in deep learning.

### `BF16`

- 16 bits per value
- layout: `1` sign bit, `8` exponent bits, `7` mantissa bits
- 2 bytes per value

`BF16` keeps the wide exponent range of `FP32`, which is why it is popular for training.

### `FP8`

`FP8` usually refers to one of two common encodings:

- `E4M3`: 1 sign, 4 exponent, 3 mantissa. Offers higher precision (more mantissa bits) but a smaller dynamic range (fewer exponent bits).
- `E5M2`: 1 sign, 5 exponent, 2 mantissa. Offers a wider dynamic range but lower precision.

Memory consumption:

- 8 bits per value
- 1 byte per value

FP8 is used in the `A8W8` quantization scheme, which quantizes both activations (`A`) and weights (`W`) to 8-bit precision.

While `INT8` has broad support, efficient `FP8` computation requires specific hardware capabilities found in newer GPUs like NVIDIA H100/H200/B100 and AMD Instinct MI300 series. Without native hardware acceleration, the benefits of `FP8` might not be fully realized.

Compared with integer quantization, `FP8` gives a larger dynamic range but fewer exact levels near any specific value.

### `FP6`

`FP6` is not a single universal standard. It usually means a 6-bit floating-point style format proposed in research or used in specialized hardware flows.

- 6 bits per value
- bit-packed

Because the exact split between exponent and mantissa can vary, always check the format specification used by the hardware or paper.

### `FP4`

Like `FP6`, `FP4` is usually format-family terminology, not one single universal encoding.

- 4 bits per value
- usually 2 values per byte

To reade these value, unpack the 4-bit code. Then decode it using the specific 4-bit floating-point table or bit layout.

Many practical implementations treat very low-bit floating formats through lookup tables rather than general arithmetic decoding.

### `NF4`

`NF4` means NormalFloat4, a 4-bit format designed so its 16 code points better match normally distributed weight values.

Storage in memory:

- 4 bits per value
- usually 2 values per byte
- the 4-bit code is an index into a fixed set of reconstruction values

For a code `c`:

```markdown
x_hat = s_g * table[c]
```

Where:

- `table[c]` is the fixed normalized reconstruction value
- `s_g` is usually a per-group or per-block scale

Read and reconstruct:

1. Load and unpack the 4-bit code.
2. Look up the normalized value in a table.
3. Multiply by the group scale.

This is popular in some weight-only LLM quantization methods.

## Microscaling Formats

These formats combine:

- very small per-element floating-point values
- a shared scale for a block of elements

This idea is often called microscaling or block scaling.

### General storage idea

For a block of `B` elements:

- each element is stored in a low-bit floating format
- one shared block scale is stored separately

Conceptually:

```markdown
block = {scale_block, m_0, m_1, ..., m_(B-1)}
```

where `scale_block` is the shared scale for the whole block of `B` values. It tells you the overall magnitude/range of that block. `m_i` is the stored low-precision value for element `i` inside the block. 

The reconstructed value is:

```markdown
x_hat_i = scale_block * decode(m_i)
```

If the scale is itself represented as a power of two:

```markdown
scale_block = 2^k
x_hat_i = 2^k * decode(m_i)
```

### Why microscaling is useful

- Better dynamic range than plain tiny floats alone.
- Less metadata than storing a separate scale per element.
- Good match for matrix multiply hardware that operates on blocks.

### `MXFP8`

- 8-bit float-like element per value
- plus one shared scale per block

For block size `B`, approximate storage per element is:

```markdown
bits_per_element ~= 8 + scale_bits / B
```

### `MXFP6`


- 6-bit float-like element per value
- bit-packed payload
- one shared scale per block

Approximate cost:

```markdown
bits_per_element ~= 6 + scale_bits / B
```

The main extra work compared with `MXFP8` is unpacking the 6-bit fields.

### `MXFP4`

- 4-bit float-like element per value
- usually 2 packed values per byte
- one shared scale per block

Approximate cost:

```markdown
bits_per_element ~= 4 + scale_bits / B
```

This gives excellent compression, but the quality depends heavily on:

- block size
- scale choice
- the exact 4-bit element encoding

## Practical Intuition

In real systems, reading a quantized value usually means one of these three paths:

### Integer path

```markdown
stored bits -> unpack integer q -> apply scale/zero-point -> x_hat
```

### Floating path

```markdown
stored bits -> decode sign/exponent/mantissa -> x_hat
```

### Microscaling path

```markdown
stored bits + shared block scale -> decode element -> multiply by block scale -> x_hat
```
