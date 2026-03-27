import csv
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import triton

from fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.solve_tril import solve_tril, solve_tril_16x16_kernel
from fla.ops.chunk_cumsum_matrix_inv_a import chunk_cumsum_matrix_inv_a_fwd
from fla.ops.chunk_matrix_inv_a import chunk_matrix_inv_a_fwd
from fla.ops.cumsum import chunk_local_cumsum
from fla.ops.wy_fast import recompute_w_u_fwd
from fla.ops.index import prepare_chunk_indices
from fla.ops.utils import is_tma_supported
from fla.ops.fused_cumsum_kkt import fused_cumsum_kkt
from fla.ops.fused_merge_recompute import fused_merge_recompute
from fla.ops.fused_preprocessing import fused_preprocessing_fwd

BATCHES = [1, 4, 16, 64, 256, 512]
SEQ_LENS = [512, 1024, 4 * 1024, 8192]
NUM_V_HEADS_LIST = [4, 8, 16, 32, 64]
NUM_K_HEADS_LIST = [2, 4, 8, 16, 32, 64]
K_DIMS = [64, 128]
V_DIMS = [64, 128]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]

WARMUP = 3
ITERS = 20
CHUNK_SIZE = 64
CSV_FILE = "preprocess_benchmark.csv"

DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}

FLA_TRIL_PRECISION = "ieee"


def make_inputs(batch, seq_len, num_k_heads, num_v_heads, k_dim, v_dim, dtype, device="cuda"):
    k = F.normalize(
        torch.randn(batch, seq_len, num_k_heads, k_dim, device=device, dtype=dtype),
        p=2, dim=-1,
    )
    v = torch.randn(batch, seq_len, num_v_heads, v_dim, device=device, dtype=dtype)
    beta = torch.rand(batch, seq_len, num_v_heads, device=device, dtype=dtype).sigmoid()
    g_raw = F.logsigmoid(torch.randn(batch, seq_len, num_v_heads, device=device, dtype=dtype))
    return k, v, beta, g_raw


def estimate_memory_bytes(batch, seq_len, num_k_heads, num_v_heads, k_dim, v_dim, dtype):
    elem = torch.finfo(dtype).bits // 8
    k_bytes = batch * seq_len * num_v_heads * k_dim * elem
    v_bytes = batch * seq_len * num_v_heads * v_dim * elem
    beta_bytes = batch * seq_len * num_v_heads * elem
    g_raw_bytes = batch * seq_len * num_v_heads * elem
    g_bytes = batch * seq_len * num_v_heads * 4
    a_float32_bytes = batch * seq_len * num_v_heads * CHUNK_SIZE * 4
    a_dtype_bytes = batch * seq_len * num_v_heads * CHUNK_SIZE * elem
    ai16_bytes = batch * seq_len * num_v_heads * 16 * 4
    w_bytes = batch * seq_len * num_v_heads * k_dim * elem
    u_bytes = batch * seq_len * num_v_heads * v_dim * elem
    peak = (k_bytes + v_bytes + beta_bytes + g_raw_bytes + g_bytes
            + a_float32_bytes + a_dtype_bytes + ai16_bytes + w_bytes + u_bytes)
    return peak * 2


def bench_reference(k, v, beta, g_raw):
    """Scenario 1: cumsum + chunk_scaled_dot_kkt + solve_tril + recompute_w_u (all separate)."""
    print('bench_reference')
    times = []
    for _ in range(WARMUP):
        g = chunk_local_cumsum(g_raw, chunk_size=CHUNK_SIZE, cu_seqlens=None)
        A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g, output_dtype=torch.float32)
        A = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=None)
    del g, A, w, u
    torch.cuda.synchronize()

    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g = chunk_local_cumsum(g_raw, chunk_size=CHUNK_SIZE, cu_seqlens=None)
        A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g, output_dtype=torch.float32)
        A = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=None)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
        del g, A, w, u

    return np.mean(times), np.std(times)


def bench_fused_matrix_a(k, v, beta, g_raw, fusion=False):
    """Scenario 2: fused cumsum + chunk_matrix_inv_a + recompute_w_u."""
    print('bench_fused_matrix_a')
    times = []
    for _ in range(WARMUP):
        if fusion:
            g, A = chunk_cumsum_matrix_inv_a_fwd(
                k=k, beta=beta, g=g_raw, cu_seqlens=None, output_dtype=k.dtype
            )
        else:
            g = chunk_local_cumsum(g_raw, chunk_size=CHUNK_SIZE, cu_seqlens=None)
            A = chunk_matrix_inv_a_fwd(k=k, beta=beta, g=g, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=None)
    del g, A, w, u
    torch.cuda.synchronize()

    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if fusion:
            g, A = chunk_cumsum_matrix_inv_a_fwd(
                k=k, beta=beta, g=g_raw, cu_seqlens=None, output_dtype=k.dtype
            )
        else:
            g = chunk_local_cumsum(g_raw, chunk_size=CHUNK_SIZE, cu_seqlens=None)
            A = chunk_matrix_inv_a_fwd(k=k, beta=beta, g=g, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=None)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
        del g, A, w, u

    return np.mean(times), np.std(times)

def bench_sg_preprocessing(k, v, beta, g_raw, fusion=False):
    """Scenario 3: fused cumsum + kkt + solve_tril + merge_recompute (from SGLang commit)."""
    print('bench_sg_preprocessing')
    B, T, Hv = beta.shape
    times = []
    for _ in range(WARMUP):
        g, A = fused_cumsum_kkt(g_raw, k, beta, chunk_size=CHUNK_SIZE, cu_seqlens=None)
        chunk_indices_16 = None
        NT_16 = triton.cdiv(T, 16)
        Ai16 = torch.empty(B, T, Hv, 16, device=A.device, dtype=torch.float32)
        solve_tril_16x16_kernel[(NT_16, B * Hv)](
            A=A, Ai=Ai16, cu_seqlens=None, chunk_indices=chunk_indices_16,
            T=T, H=Hv, BT=CHUNK_SIZE,
            USE_TMA=is_tma_supported, DOT_PRECISION=FLA_TRIL_PRECISION,
        )
        w, u = fused_merge_recompute(k, v, beta, g, A, Ai16, chunk_size=CHUNK_SIZE, cu_seqlens=None)
    del g, A, Ai16, w, u
    torch.cuda.synchronize()

    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g, A = fused_cumsum_kkt(g_raw, k, beta, chunk_size=CHUNK_SIZE, cu_seqlens=None)
        NT_16 = triton.cdiv(T, 16)
        Ai16 = torch.empty(B, T, Hv, 16, device=A.device, dtype=torch.float32)
        solve_tril_16x16_kernel[(NT_16, B * Hv)](
            A=A, Ai=Ai16, cu_seqlens=None, chunk_indices=None,
            T=T, H=Hv, BT=CHUNK_SIZE,
            USE_TMA=is_tma_supported, DOT_PRECISION=FLA_TRIL_PRECISION,
        )
        w, u = fused_merge_recompute(k, v, beta, g, A, Ai16, chunk_size=CHUNK_SIZE, cu_seqlens=None)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
        del g, A, Ai16, w, u

    return np.mean(times), np.std(times)


def bench_fused_all(k, v, beta, g_raw):
    """Scenario 4: fused all preprocessing into one kernel."""
    B, T, Hv = beta.shape

    print('bench_fused_all')
    times = []
    for _ in range(WARMUP):
        g, w, u = fused_preprocessing_fwd(k=k, v=v, beta=beta, g=g_raw, cu_seqlens=None)
    del g, w, u
    torch.cuda.synchronize()

    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g, w, u = fused_preprocessing_fwd(k=k, v=v, beta=beta, g=g_raw, cu_seqlens=None)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
        del g, w, u

    return np.mean(times), np.std(times)


def main():
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA is required"

    combos = list(itertools.product(DTYPES, BATCHES, SEQ_LENS, NUM_K_HEADS_LIST, NUM_V_HEADS_LIST, K_DIMS, V_DIMS))
    total = len(combos)

    gpu_total = torch.cuda.get_device_properties(device).total_memory
    print(f"GPU total memory: {gpu_total / 2**30:.2f} GiB")

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "data_type", "batch", "seq_len", "num_k_heads", "num_v_heads",
            "k_dim", "v_dim",
            "reference avg time, us", "reference std",
            "fused_matrix_A avg time, us", "fused_matrix_A std",
            "fused_cumsum_matrix_inv_a avg time, us", "fused_cumsum_matrix_inv_a std",
            "fused_sg_preprocessing avg time, us", "fused_sg_preprocessing std",
            "fused_all avg time, us", "fused_all std",
        ])

        for idx, (dtype, batch, seq_len, num_k_heads, num_v_heads, k_dim, v_dim) in enumerate(combos):
            if num_k_heads > num_v_heads:
                continue

            label = (f"[{idx+1}/{total}] dtype={DTYPE_NAMES[dtype]} "
                     f"B={batch} T={seq_len} Hk={num_k_heads} Hv={num_v_heads} "
                     f"K={k_dim} V={v_dim}")
            print(f"{label}")

            if (batch * seq_len * num_v_heads * v_dim >= 2147483647 or batch * seq_len * num_k_heads * k_dim >= 2147483647):
                print(f"skipping")
                continue

            peak = estimate_memory_bytes(batch, seq_len, num_k_heads, num_v_heads, k_dim, v_dim, dtype)
            free_mem, _ = torch.cuda.mem_get_info(device)
            print(f"peak: {peak / 2**30:.1f} GiB, free: {free_mem / 2**30:.1f} GiB")
            if peak > free_mem:
                print(f"estimated {peak / 2**30:.1f} GiB > free {free_mem / 2**30:.1f} GiB, skipping")
                continue

            try:
                k, v, beta, g_raw = make_inputs(batch, seq_len, num_k_heads, num_v_heads, k_dim, v_dim, dtype, device)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM during allocation, skipping")
                torch.cuda.empty_cache()
                continue

            ref_avg, ref_std = float("nan"), float("nan")
            fused_a_avg, fused_a_std = float("nan"), float("nan")
            fused_cumsum_matrix_inv_a_avg, fused_cumsum_matrix_inv_a_std = float("nan"), float("nan")
            fused_sg_preprocessing_avg, fused_sg_preprocessing_std = float("nan"), float("nan"), float("nan")
            fused_all_avg, fused_all_std = float("nan"), float("nan")

            try:
                ref_avg, ref_std = bench_reference(k, v, beta, g_raw)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM in reference, skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"reference error: {e}, skipping")

            try:
                fused_a_avg, fused_a_std = bench_fused_matrix_a(k, v, beta, g_raw, fusion=False)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM in fused_matrix_A, skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"fused_matrix_A error: {e}, skipping")

            try:
                fused_cumsum_matrix_inv_a_avg, fused_cumsum_matrix_inv_a_std = bench_fused_matrix_a(k, v, beta, g_raw, fusion=True)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM in fused_cumsum_matrix_inv_a, skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"fused_cumsum_matrix_inv_a error: {e}, skipping")

            try:
                fused_sg_preprocessing_avg, fused_sg_preprocessing_std = bench_sg_preprocessing(k, v, beta, g_raw)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM in fused_all, skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"fused_all error: {e}, skipping")

            try:
                fused_all_avg, fused_all_std = bench_fused_all(k, v, beta, g_raw)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM in fused_all, skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"fused_all error: {e}, skipping")

            writer.writerow([
                DTYPE_NAMES[dtype], batch, seq_len, num_k_heads, num_v_heads,
                k_dim, v_dim,
                f"{ref_avg:.2f}", f"{ref_std:.2f}",
                f"{fused_a_avg:.2f}", f"{fused_a_std:.2f}",
                f"{fused_cumsum_matrix_inv_a_avg:.2f}", f"{fused_cumsum_matrix_inv_a_std:.2f}",
                f"{fused_sg_preprocessing_avg:.2f}", f"{fused_sg_preprocessing_std:.2f}",
                f"{fused_all_avg:.2f}", f"{fused_all_std:.2f}",
            ])
            f.flush()

            del k, v, beta, g_raw
            torch.cuda.empty_cache()

            print(f"ref: {ref_avg:.2f} ± {ref_std:.2f}  |  "
                  f"fused_A: {fused_a_avg:.2f} ± {fused_a_std:.2f}  |  "
                  f"fused_cumsum_matrix_inv_a: {fused_cumsum_matrix_inv_a_avg:.2f} ± {fused_cumsum_matrix_inv_a_std:.2f}  |  "
                  f"fused_sg_preprocessing: {fused_sg_preprocessing_avg:.2f} ± {fused_sg_preprocessing_std:.2f}  |  "
                  f"fused_all: {fused_all_avg:.2f} ± {fused_all_std:.2f} us")

    print(f"\nResults saved to {CSV_FILE}")


if __name__ == "__main__":
    main()
