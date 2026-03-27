import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .op import exp
from .utils import input_guard, gpu_timestamp


@triton.heuristics(
    {
        "STORE_A_INV": lambda args: args["A_out"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def fused_intra_chunk_fwd_kernel(
    k,
    v,
    beta,
    g,
    w,
    u,
    A_out,
    g_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_A_INV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    # --------------------------
    # 1) local cumsum on g
    # --------------------------
    p_g = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g = tl.load(p_g, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_g = tl.where(m_t, b_g, 0.0)
    b_g_cumsum = tl.cumsum(b_g, axis=0)
    p_go = tl.make_block_ptr(
        g_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    tl.store(p_go, b_g_cumsum.to(p_go.dtype.element_ty), boundary_check=(0,))

    # --------------------------
    # 2) A = beta * K * K^T (+ gate)
    # --------------------------
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,), padding_option="zero")
    b_beta = tl.where(m_t, b_beta, 0.0)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.where(m_t[:, None], b_k, 0.0)
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))

    b_A = b_A * exp(b_g_cumsum[:, None] - b_g_cumsum[None, :])

    # strict lower-triangular intra-chunk mask
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = tl.where(m_A, b_A, 0.0)

    # --------------------------
    # 3) solve tril: Ai = (I + A)^-1 (64x64 via 16x16 block merge)
    # --------------------------
    o_i = tl.arange(0, 16)
    m_A16 = o_i[:, None] > o_i[None, :]
    m_I16 = o_i[:, None] == o_i[None, :]
    t0 = i_t * BT + o_i
    t1 = i_t * BT + 16 + o_i
    t2 = i_t * BT + 32 + o_i
    t3 = i_t * BT + 48 + o_i
    m0 = t0 < T
    m1 = t1 < T
    m2 = t2 < T
    m3 = t3 < T

    b_A_11 = b_A[0:16, 0:16]
    b_A_22 = b_A[16:32, 16:32]
    b_A_33 = b_A[32:48, 32:48]
    b_A_44 = b_A[48:64, 48:64]

    b_Ai_11 = -tl.where(m_A16, b_A_11, 0.0)
    b_Ai_22 = -tl.where(m_A16, b_A_22, 0.0)
    b_Ai_33 = -tl.where(m_A16, b_A_33, 0.0)
    b_Ai_44 = -tl.where(m_A16, b_A_44, 0.0)

    for i in range(2, 16):
        b_a_11 = -b_A_11[i, :]
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where(((o_i == i) & (t0[i] < T))[:, None], b_a_11, b_Ai_11)
    for i in range(2, 16):
        b_a_22 = -b_A_22[i, :]
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where(((o_i == i) & (t1[i] < T))[:, None], b_a_22, b_Ai_22)
    for i in range(2, 16):
        b_a_33 = -b_A_33[i, :]
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where(((o_i == i) & (t2[i] < T))[:, None], b_a_33, b_Ai_33)
    for i in range(2, 16):
        b_a_44 = -b_A_44[i, :]
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where(((o_i == i) & (t3[i] < T))[:, None], b_a_44, b_Ai_44)

    b_Ai_11 += m_I16
    b_Ai_22 += m_I16
    b_Ai_33 += m_I16
    b_Ai_44 += m_I16

    b_A_21 = b_A[16:32, 0:16]
    b_A_31 = b_A[32:48, 0:16]
    b_A_32 = b_A[32:48, 16:32]
    b_A_41 = b_A[48:64, 0:16]
    b_A_42 = b_A[48:64, 16:32]
    b_A_43 = b_A[48:64, 32:48]

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21), b_Ai_11)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32), b_Ai_22)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43), b_Ai_33)
    b_Ai_31 = -tl.dot(b_Ai_33, tl.dot(b_A_31, b_Ai_11) + tl.dot(b_A_32, b_Ai_21))
    b_Ai_42 = -tl.dot(b_Ai_44, tl.dot(b_A_42, b_Ai_22) + tl.dot(b_A_43, b_Ai_32))
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11) + tl.dot(b_A_42, b_Ai_21) + tl.dot(b_A_43, b_Ai_31),
    )

    # Mask invalid rows/cols for short tail chunks.
    b_Ai_11 = tl.where(m0[:, None] & m0[None, :], b_Ai_11, 0.0)
    b_Ai_22 = tl.where(m1[:, None] & m1[None, :], b_Ai_22, 0.0)
    b_Ai_33 = tl.where(m2[:, None] & m2[None, :], b_Ai_33, 0.0)
    b_Ai_44 = tl.where(m3[:, None] & m3[None, :], b_Ai_44, 0.0)
    b_Ai_21 = tl.where(m1[:, None] & m0[None, :], b_Ai_21, 0.0)
    b_Ai_31 = tl.where(m2[:, None] & m0[None, :], b_Ai_31, 0.0)
    b_Ai_32 = tl.where(m2[:, None] & m1[None, :], b_Ai_32, 0.0)
    b_Ai_41 = tl.where(m3[:, None] & m0[None, :], b_Ai_41, 0.0)
    b_Ai_42 = tl.where(m3[:, None] & m1[None, :], b_Ai_42, 0.0)
    b_Ai_43 = tl.where(m3[:, None] & m2[None, :], b_Ai_43, 0.0)

    if STORE_A_INV:
        p_Ai_11 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 0, 0),
            (16, 16),
            (1, 0),
        )
        p_Ai_22 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 16, 16),
            (16, 16),
            (1, 0),
        )
        p_Ai_33 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 32, 32),
            (16, 16),
            (1, 0),
        )
        p_Ai_44 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 48, 48),
            (16, 16),
            (1, 0),
        )
        p_Ai_21 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 16, 0),
            (16, 16),
            (1, 0),
        )
        p_Ai_31 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 32, 0),
            (16, 16),
            (1, 0),
        )
        p_Ai_32 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 32, 16),
            (16, 16),
            (1, 0),
        )
        p_Ai_41 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 48, 0),
            (16, 16),
            (1, 0),
        )
        p_Ai_42 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 48, 16),
            (16, 16),
            (1, 0),
        )
        p_Ai_43 = tl.make_block_ptr(
            A_out + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + 48, 32),
            (16, 16),
            (1, 0),
        )
        tl.store(
            p_Ai_11,
            b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_22,
            b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_33,
            b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_44,
            b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_21,
            b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_31,
            b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_32,
            b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_41,
            b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_42,
            b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_43,
            b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )

    # --------------------------
    # 4) recompute u = Ai @ (beta * v)
    # --------------------------
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u_1 = tl.make_block_ptr(
            u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT + 0, i_v * BV), (16, BV), (1, 0)
        )
        p_u_2 = tl.make_block_ptr(
            u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT + 16, i_v * BV), (16, BV), (1, 0)
        )
        p_u_3 = tl.make_block_ptr(
            u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT + 32, i_v * BV), (16, BV), (1, 0)
        )
        p_u_4 = tl.make_block_ptr(
            u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT + 48, i_v * BV), (16, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.where(m_t[:, None], b_v, 0.0)
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_vb_1, b_vb_2 = b_vb[0:16, :], b_vb[16:32, :]
        b_vb_3, b_vb_4 = b_vb[32:48, :], b_vb[48:64, :]
        b_u_1 = tl.dot(b_Ai_11.to(b_v.dtype), b_vb_1, allow_tf32=False)
        b_u_2 = tl.dot(b_Ai_21.to(b_v.dtype), b_vb_1, allow_tf32=False) + tl.dot(
            b_Ai_22.to(b_v.dtype), b_vb_2, allow_tf32=False
        )
        b_u_3 = (
            tl.dot(b_Ai_31.to(b_v.dtype), b_vb_1, allow_tf32=False)
            + tl.dot(b_Ai_32.to(b_v.dtype), b_vb_2, allow_tf32=False)
            + tl.dot(b_Ai_33.to(b_v.dtype), b_vb_3, allow_tf32=False)
        )
        b_u_4 = (
            tl.dot(b_Ai_41.to(b_v.dtype), b_vb_1, allow_tf32=False)
            + tl.dot(b_Ai_42.to(b_v.dtype), b_vb_2, allow_tf32=False)
            + tl.dot(b_Ai_43.to(b_v.dtype), b_vb_3, allow_tf32=False)
            + tl.dot(b_Ai_44.to(b_v.dtype), b_vb_4, allow_tf32=False)
        )
        tl.store(p_u_1, b_u_1.to(p_u_1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u_2, b_u_2.to(p_u_2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u_3, b_u_3.to(p_u_3.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u_4, b_u_4.to(p_u_4.dtype.element_ty), boundary_check=(0, 1))

    # --------------------------
    # 5) recompute w = Ai @ (beta * exp(g_cumsum) * k)
    # --------------------------
    b_eg = exp(b_g_cumsum)
    b_scale = b_beta.to(tl.float32) * b_eg

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w_1 = tl.make_block_ptr(
            w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + 0, i_k * BK), (16, BK), (1, 0)
        )
        p_w_2 = tl.make_block_ptr(
            w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0)
        )
        p_w_3 = tl.make_block_ptr(
            w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0)
        )
        p_w_4 = tl.make_block_ptr(
            w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.where(m_t[:, None], b_k, 0.0)
        b_kb = (b_k * b_scale[:, None]).to(b_k.dtype)
        b_kb_1, b_kb_2 = b_kb[0:16, :], b_kb[16:32, :]
        b_kb_3, b_kb_4 = b_kb[32:48, :], b_kb[48:64, :]
        b_w_1 = tl.dot(b_Ai_11.to(b_k.dtype), b_kb_1)
        b_w_2 = tl.dot(b_Ai_21.to(b_k.dtype), b_kb_1) + tl.dot(
            b_Ai_22.to(b_k.dtype), b_kb_2
        )
        b_w_3 = (
            tl.dot(b_Ai_31.to(b_k.dtype), b_kb_1)
            + tl.dot(b_Ai_32.to(b_k.dtype), b_kb_2)
            + tl.dot(b_Ai_33.to(b_k.dtype), b_kb_3)
        )
        b_w_4 = (
            tl.dot(b_Ai_41.to(b_k.dtype), b_kb_1)
            + tl.dot(b_Ai_42.to(b_k.dtype), b_kb_2)
            + tl.dot(b_Ai_43.to(b_k.dtype), b_kb_3)
            + tl.dot(b_Ai_44.to(b_k.dtype), b_kb_4)
        )
        tl.store(p_w_1, b_w_1.to(p_w_1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w_2, b_w_2.to(p_w_2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w_3, b_w_3.to(p_w_3.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w_4, b_w_4.to(p_w_4.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def fused_intra_chunk_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_g_cumsum: bool = False,
    output_A_inv: bool = False,
    a_output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """
    Fused intra-chunk path:
      local_cumsum(g) -> A = beta * K * K^T (+ gate) -> solve_tril -> recompute(w, u).

    Returns:
      g_cumsum_or_none, A_inv_or_none, w, u
    """
    B, T, Hg, K = k.shape
    H, V = beta.shape[-1], v.shape[-1]
    assert chunk_size == 64, "Only chunk_size=64 is supported by fused_intra_chunk_fwd."
    assert K <= 256, "Current fused kernel supports K <= 256."
    if cu_seqlens is not None:
        assert k.shape[0] == 1, "Only batch size 1 is supported with cu_seqlens."

    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)
    g_cumsum = torch.empty_like(g, dtype=torch.float32) if (output_g_cumsum and g is not None) else None
    A_inv = (
        torch.empty(B, T, H, BT, device=k.device, dtype=a_output_dtype)
        if output_A_inv
        else None
    )

    fused_intra_chunk_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        w=w,
        u=u,
        A_out=A_inv,
        g_out=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )

    return g_cumsum, A_inv, w, u

