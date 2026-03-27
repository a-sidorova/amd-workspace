# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton
import triton.profiler as proton
import triton.profiler.language as pl
import triton.profiler.viewer as viewer

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp
from .utils import use_cuda_graph


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": 64}, num_warps=4, num_stages=2)
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=use_cuda_graph,
)
# @triton.autotune(
#     configs=[
#         triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1, 2, 4, 8]
#         for num_stages in [1, 2, 3, 4]
#         for BV in [16, 32, 64]
#     ],
#     key=["H", "K", "V", "BT"],
#     use_cuda_graph=use_cuda_graph,
# )
@triton.jit(do_not_specialize=["T"])
def chunk_fused_h_o_kernel(
    q,
    k,
    v,
    w,
    g,
    o,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pl.enter_scope("kernel")
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # Keep the recurrent state in [K, BV] tiles so both the output path
    # and the state-update path can use it directly without per-iteration
    # transposes. Public h0/ht tensors keep their original [V, K] layout.
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    q += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    o += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_qk = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    pl.enter_scope("prologue")
    if USE_INITIAL_STATE:
        # View the external [V, K] tensor as logical [K, V] using strides
        # (1, K), preserving the public tensor layout while matching the
        # internal [K, BV] accumulator layout.
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (1, K), (128, i_v * BV), (64, BV), (0, 1))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (1, K), (192, i_v * BV), (64, BV), (0, 1))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
    pl.exit_scope("prologue")

    pl.enter_scope("loop")
    for i_t in range(NT):
        # ---- Phase 1: merged K-loop ----
        # Inter-chunk output (q @ h), intra-chunk attention (q @ k^T),
        # and value correction (w @ h) reuse the same state tile.
        # k blocks are kept in registers (b_k1..b_k4) and reused in Phase 4
        # for the state update, avoiding redundant GMEM reloads.
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1))
        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k1 = tl.load(p_k, boundary_check=(0, 1))
        b_w_blk = tl.load(p_w, boundary_check=(0, 1))
        b_h1_bf16 = b_h1.to(b_q.dtype)
        b_o += tl.dot(b_q, b_h1_bf16)
        b_A += tl.dot(b_q, b_k1)
        b_vc = tl.dot(b_w_blk, b_h1_bf16)

        if K > 64:
            p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k2 = tl.load(p_k, boundary_check=(0, 1))
            b_w_blk = tl.load(p_w, boundary_check=(0, 1))
            b_h2_bf16 = b_h2.to(b_q.dtype)
            b_o += tl.dot(b_q, b_h2_bf16)
            b_A += tl.dot(b_q, b_k2)
            b_vc += tl.dot(b_w_blk, b_h2_bf16)

        # ---- Phase 2: v_new = u - w @ h^T ----
        p_u = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_vc

        # ---- Phase 3: gating, causal mask, output ----
        if USE_G:
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_o *= exp(b_g)[:, None]
            b_A *= exp(b_g[:, None] - b_g[None, :])

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)

        b_v_out = b_v.to(k.dtype.element_ty)
        b_o = (b_o + tl.dot(b_A.to(b_v_out.dtype), b_v_out)) * scale

        p_o = tl.make_block_ptr(
            o, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # ---- Phase 4: state gating + update ----
        # Reuses b_k1..b_k4 from Phase 1 (no GMEM reload).
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last

        b_v = b_v.to(k.dtype.element_ty)

        b_h1 += tl.dot(b_k1, b_v)
        if K > 64:
            b_h2 += tl.dot(b_k2, b_v)
    pl.exit_scope("loop")

    pl.enter_scope("extracted")

    pl.exit_scope("extracted")

    pl.enter_scope("epilogue")
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    pl.exit_scope("epilogue")
    pl.exit_scope("kernel")


def chunk_fused_h_o_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT)
        if cu_seqlens is not None
        else None
    )
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    if scale is None:
        scale = K ** -0.5

    o = torch.empty_like(u)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_fused_h_o_kernel[grid](
        q=q,
        k=k,
        v=u,
        w=w,
        g=g,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return o, final_state
