import warnings
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import numpy as np

from fla.ops.chunk import chunk_gated_delta_rule, profile_chunk_gated_delta_rule

from vllm.triton_utils import tl, triton
import triton.profiler as proton

def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
):

    if value.shape[2] // key.shape[2] > 1:
        query = query.repeat_interleave(value.shape[2] // key.shape[2], dim=2)
        key = key.repeat_interleave(value.shape[2] // key.shape[2], dim=2)

    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

def _run_triton_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    fusion: bool = False,
):
    return chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        use_fusion=fusion,
    )

def profile_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    warmup,
    iters,
    fusion: bool = False,
):
    return profile_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        warmup=warmup,
        iters=iters,
        use_fusion=fusion,
    )

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of benchmark iterations for the Triton path.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate Triton output against torch reference implementation.",
    )
    parser.add_argument(
        "--profile-kernel",
        action="store_true",
        help="Profile the Triton kernel.",
        dest="profile_kernel"
    )
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Use fused matrix-inverse kernel (chunk_size=64).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run chunk_gated_delta_rule_fwd.")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    tp = 8

    # Qwen3.5-397B-A17B text config:
    # linear_num_value_heads=64, linear_key_head_dim=128, linear_value_head_dim=128
    # max_position_embeddings=262144 (too large for a lightweight correctness run).
    bsz = 64
    seq_len = 8580
    num_k_heads = 16 // tp
    num_v_heads = 64 // tp
    k_dim = 128
    v_dim = 128
    q = torch.randn(bsz, seq_len, num_k_heads, k_dim, device=device, dtype=dtype)
    k = F.normalize(
        torch.randn(bsz, seq_len, num_k_heads, k_dim, device=device, dtype=dtype),
        p=2,
        dim=-1,
    )
    v = torch.randn(bsz, seq_len, num_v_heads, v_dim, device=device, dtype=dtype)
    beta = torch.rand(bsz, seq_len, num_v_heads, device=device, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.randn(bsz, seq_len, num_v_heads, device=device, dtype=dtype))
    scale = k_dim ** -0.5

    if args.profile_kernel:
        initial_state = torch.zeros(
            bsz, num_v_heads, v_dim, k_dim, device=device, dtype=dtype
        )
        o_triton, final_state_triton = profile_kernel(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            args.warmup_iters,
            args.iters,
            fusion=args.fusion,
        )
        return

    for _ in range(args.warmup_iters):
        initial_state = torch.zeros(
            bsz, num_v_heads, v_dim, k_dim, device=device, dtype=dtype
        )
        _run_triton_once(q, k, v, g, beta, scale, initial_state, fusion=args.fusion)
    torch.cuda.synchronize()

    o_triton, final_state_triton = None, None
    timings_us = []
    proton.start("run_triton_once")
    for _ in range(args.iters):
        initial_state = torch.zeros(
            bsz, num_v_heads, v_dim, k_dim, device=device, dtype=dtype
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        o_triton, final_state_triton = _run_triton_once(
            q, k, v, g, beta, scale, initial_state, fusion=args.fusion
        )
        end.record()
        torch.cuda.synchronize()
        timings_us.append(start.elapsed_time(end) * 1000)
    proton.finalize()
    print(f"Triton mean time: {np.mean(timings_us):.6f} us")
    print(f"Triton std time: {np.std(timings_us):.6f} us")
    print(f"Triton min time: {np.min(timings_us):.6f} us")
    print(f"Triton max time: {np.max(timings_us):.6f} us")

    print(f"Triton output shape: {tuple(o_triton.shape)}")
    print(f"Triton final_state shape: {tuple(final_state_triton.shape)}")

    if args.validate:
        o_torch, final_state_torch = torch_chunk_gated_delta_rule(
            query=q,
            key=k,
            value=v,
            g=g,
            beta=beta,
            chunk_size=64,
        )
        max_abs_diff = (o_triton.float() - o_torch.float()).abs().max().item()
        is_close = torch.allclose(o_triton.float(), o_torch.float(), atol=2e-2, rtol=2e-2)

        print(f"Validation max abs diff: {max_abs_diff:.6f}")
        print(f"Outputs match torch_chunk_gated_delta_rule: {is_close}")
        print(f"Torch output shape:  {tuple(o_torch.shape)}")
        print(f"Torch final_state shape:  {tuple(final_state_torch.shape)}")

        if not is_close:
            raise AssertionError(
                "chunk_gated_delta_rule_fwd output does not match torch_chunk_gated_delta_rule."
            )


if __name__ == "__main__":
    main()

