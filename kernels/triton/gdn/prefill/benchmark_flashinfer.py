#!/usr/bin/env python3
"""Benchmark chunked Gated Delta Rule prefill kernels.

Supports:
- flashinfer backend via `fi_chunk_gated_delta_rule` wrapper
- vllm(triton/FLa) backend via `chunk_gated_delta_rule`
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

# Ensure local workspace import works when script is launched from any cwd.
WORKSPACE_ROOT = Path("/workspace")
VLLM_REPO_ROOT = WORKSPACE_ROOT / "vllm"
if str(VLLM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_REPO_ROOT))

from vllm.model_executor.models.qwen3_next import fi_chunk_gated_delta_rule  # noqa: E402
from vllm.model_executor.layers.fla.ops import (  # noqa: E402
    chunk_gated_delta_rule as vllm_chunk_gated_delta_rule,
)
from vllm.triton_utils.allocation import set_triton_allocator  # noqa: E402

device = torch.device("cuda:0")


@dataclass(frozen=True)
class ModelLinearConfig:
    name: str
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int


MODEL_CONFIGS: dict[str, ModelLinearConfig] = {
    "qwen3.5-397b-a17b": ModelLinearConfig(
        name="Qwen/Qwen3.5-397B-A17B",
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    ),
    "qwen3.5-35b-a3b": ModelLinearConfig(
        name="Qwen/Qwen3.5-35B-A3B",
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    ),
}

# Hardcoded values requested by user.
DEFAULT_BATCHES = [1, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_SEQ_LENS = [128, 512, 1024, 2 * 1024, 8 * 1024, 16 * 1024]


def _parse_comma_ints(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    try:
        numbers = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Failed to parse comma-separated integers from '{value}'."
        ) from exc
    if any(n <= 0 for n in numbers):
        raise argparse.ArgumentTypeError("All values must be positive integers.")
    return numbers


def _parse_models(value: str) -> list[str]:
    keys = [k.strip().lower() for k in value.split(",") if k.strip()]
    if not keys:
        raise argparse.ArgumentTypeError("Expected at least one model key.")
    bad = [k for k in keys if k not in MODEL_CONFIGS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown model keys: {bad}. Allowed: {sorted(MODEL_CONFIGS.keys())}"
        )
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark chunked Gated Delta Rule prefill implementation "
            "(flashinfer or vllm triton)."
        )
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["flashinfer", "vllm"],
        help="Backend to benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of timed benchmark iterations per config.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=20,
        help="Number of warmup iterations before benchmarking.",
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=list(MODEL_CONFIGS.keys()),
        help=(
            "Comma-separated model keys. "
            f"Default: {','.join(MODEL_CONFIGS.keys())}"
        ),
    )
    parser.add_argument(
        "--batches",
        type=_parse_comma_ints,
        default=DEFAULT_BATCHES,
        help="Comma-separated batch sizes. Default: 1,8,16,32,64,128,256,512",
    )
    parser.add_argument(
        "--seq-lens",
        type=_parse_comma_ints,
        default=DEFAULT_SEQ_LENS,
        help="Comma-separated sequence lengths. Default: 128,512,1024,2048,8192,16384",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Input dtype for q/k/v/g/beta tensors.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output txt path. If omitted, a timestamped file is created.",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be >= 0")
    return args


def _make_backend_call(backend: str) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    if backend == "flashinfer":
        return fi_chunk_gated_delta_rule
    if backend == "vllm":
        return vllm_chunk_gated_delta_rule
    raise ValueError(f"Unsupported backend: {backend}")


def _prepare_backend_runtime(backend: str, device: torch.device) -> None:
    # vLLM Triton kernels can require runtime scratch allocations.
    # In standalone scripts (outside normal vLLM worker init), allocator setup is
    # not done automatically, so configure it explicitly.
    if backend == "vllm":
        set_triton_allocator(device)


def _dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def _build_inputs(
    model_cfg: ModelLinearConfig,
    batch: int,
    seq_len: int,
    dtype: torch.dtype,
    tp: int,
) -> dict[str, torch.Tensor]:
    total_tokens = batch * seq_len
    hk = model_cfg.linear_num_key_heads // tp
    hv = model_cfg.linear_num_value_heads // tp
    dk = model_cfg.linear_key_head_dim
    dv = model_cfg.linear_value_head_dim

    q = torch.randn((1, total_tokens, hk, dk), dtype=dtype, device=device)
    k = torch.randn((1, total_tokens, hk, dk), dtype=dtype, device=device)
    v = torch.randn((1, total_tokens, hv, dv), dtype=dtype, device=device)

    # g is in log-space and beta in [0, 1].
    g = torch.nn.functional.logsigmoid(
        torch.randn((1, total_tokens, hv), dtype=dtype, device=device)
    )
    beta = torch.sigmoid(torch.randn((1, total_tokens, hv), dtype=dtype, device=device))

    # "empty torch tensor with zeroes" as requested.
    initial_state = torch.empty((batch, hv, dv, dk), dtype=dtype, device=device).zero_()

    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        seq_len,
        dtype=torch.long,
        device=device,
    )

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
    }


def _benchmark_single_case(
    backend_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    tensors: dict[str, torch.Tensor],
    warmup_iterations: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup_iterations):
        backend_fn(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            g=tensors["g"],
            beta=tensors["beta"],
            initial_state=tensors["initial_state"],
            output_final_state=True,
            cu_seqlens=tensors["cu_seqlens"],
            use_qk_l2norm_in_kernel=True,
        )
    torch.cuda.synchronize()

    elapsed_us: list[float] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start_evt.record()
        backend_fn(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            g=tensors["g"],
            beta=tensors["beta"],
            initial_state=tensors["initial_state"],
            output_final_state=True,
            cu_seqlens=tensors["cu_seqlens"],
            use_qk_l2norm_in_kernel=True,
        )
        end_evt.record()
        torch.cuda.synchronize()
        elapsed_us.append(start_evt.elapsed_time(end_evt) * 1000.0)

    mean_us = statistics.mean(elapsed_us)
    return {
        "avg_us": mean_us,
        "mean_us": mean_us,
        "min_us": min(elapsed_us),
        "max_us": max(elapsed_us),
        "stdev_us": statistics.stdev(elapsed_us) if len(elapsed_us) > 1 else 0.0,
    }


def _build_case_row(
    model_cfg: ModelLinearConfig,
    backend: str,
    batch: int,
    tp: int,
    seq_len: int,
    iters: int,
    warmup: int,
    dtype_name: str,
    stats: dict[str, float],
) -> dict[str, str | int | float]:
    total_tokens = batch * seq_len
    tokens_per_sec = (
        total_tokens / (stats["mean_us"] / 1_000_000.0)
        if stats["mean_us"] > 0
        else math.inf
    )
    return {
        "model": model_cfg.name,
        "backend": backend,
        "dtype": dtype_name,
        "tp": tp,
        "linear_num_key_heads": model_cfg.linear_num_key_heads,
        "linear_num_value_heads": model_cfg.linear_num_value_heads,
        "linear_key_head_dim": model_cfg.linear_key_head_dim,
        "linear_value_head_dim": model_cfg.linear_value_head_dim,
        "batch": batch,
        "sequence_length": seq_len,
        "total_tokens": total_tokens,
        "warmup_iterations": warmup,
        "benchmark_iterations": iters,
        "average_time_us": f"{stats['avg_us']:.4f}",
        "mean_time_us": f"{stats['mean_us']:.4f}",
        "min_time_us": f"{stats['min_us']:.4f}",
        "max_time_us": f"{stats['max_us']:.4f}",
        "stdev_time_us": f"{stats['stdev_us']:.4f}",
        "throughput_tokens_per_sec": f"{tokens_per_sec:.2f}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    args = parse_args()

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested, but CUDA is not available.")
        torch.cuda.set_device(device)
    else:
        raise RuntimeError("This benchmark script currently supports CUDA device only.")

    dtype = _dtype_from_name(args.dtype)
    backend_fn = _make_backend_call(args.backend)
    _prepare_backend_runtime(args.backend, device)

    output_path = (
        Path(args.output)
        if args.output
        else WORKSPACE_ROOT
        / f"gated_delta_prefill_benchmark_{args.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []

    for model_key in args.models:
        model_cfg = MODEL_CONFIGS[model_key]
        for tp in [1, 2, 4, 8]:
            for batch in args.batches:
                for seq_len in args.seq_lens:
                    # out of memory for flashinfer
                    if args.backend == "flashinfer" and batch * seq_len >= 256 * 16384:
                        continue
                     # out of memory for triton (vllm)
                    if args.backend == "vllm" and batch * seq_len  32 * 16384:
                        continue
                    print(
                        f"[RUN] {model_cfg.name} | batch={batch} seq={seq_len} "
                        f"(backend={args.backend}, dtype={args.dtype}, tp={tp})"
                    )
                    tensors = _build_inputs(
                        model_cfg=model_cfg,
                        batch=batch,
                        seq_len=seq_len,
                        dtype=dtype,
                        tp=tp,
                    )

                    stats = _benchmark_single_case(
                        backend_fn=backend_fn,
                        tensors=tensors,
                        warmup_iterations=args.warmup_iterations,
                        iterations=args.iterations,
                    )

                    row = _build_case_row(
                        model_cfg=model_cfg,
                        backend=args.backend,
                        batch=batch,
                        tp=tp,
                        seq_len=seq_len,
                        iters=args.iterations,
                        warmup=args.warmup_iterations,
                        dtype_name=args.dtype,
                        stats=stats,
                    )
                    rows.append(row)

                    print(
                        f"[OK] {model_cfg.name} | batch={batch} seq={seq_len} "
                        f"mean={stats['mean_us']:.4f} us"
                    )

    if rows:
        fieldnames = list(rows[0].keys())
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        output_path.write_text("", encoding="utf-8")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
