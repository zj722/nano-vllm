import triton 
import triton.language as tl
import torch
import torch.nn


@triton.jit
def fused_dynamic_quantize_kernel(
    x_ptr,          
    y_ptr,          
    scale_ptr,       
    M, K,           
    stride_xm, stride_xk,
    stride_ym, stride_yk,
    BLOCK_K: tl.constexpr 
):
    pid_m = tl.program_id(0)
    
    x_row_ptr = x_ptr + pid_m * stride_xm
    y_row_ptr = y_ptr + pid_m * stride_ym
    scale_row_ptr = scale_ptr + pid_m * 1
    
    offs_k = tl.arange(0, BLOCK_K)
    
    x = tl.load(x_row_ptr + offs_k * stride_xk, mask=offs_k < K, other=0.0)
 
    abs_x = tl.math.abs(x)
    max_val = tl.max(abs_x, axis=0)
    max_val = tl.maximum(max_val, 1e-12)
    scale = max_val / 448.0
    y = (x / scale).to(tl.float8e4nv) 

    tl.store(y_row_ptr + offs_k * stride_yk, y, mask=offs_k < K)
    tl.store(scale_row_ptr, scale)

def triton_dynamic_quantize(x: torch.Tensor):
    M, K = x.shape
    x_fp8 = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    
    BLOCK_K = triton.next_power_of_2(K)

    grid = (M, )
    fused_dynamic_quantize_kernel[grid](
        x, x_fp8, x_scale,
        M, K,
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        BLOCK_K=BLOCK_K
    )
    return x_fp8, x_scale



_PREFILL_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),  # 产生大量碎片，喂饱 36 SM
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
]

_DECODE_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=2, num_stages=3), # 注意这里用了少见的 2 Warps!
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
]


@triton.jit
def _compute_pid_mn(
    pid,
    num_pid_m,
    num_pid_n,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.autotune(
    configs=_PREFILL_CONFIGS,
    key=["N", "K"],
)
@triton.jit
def fp8_block_gemm_prefill_kernel(
    # pointers
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,

    # shapes
    M, N, K,

    # strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_b_scale_k, stride_b_scale_n,

    # compile-time constants
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUT_DTYPE_BF16: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _compute_pid_mn(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    tl.multiple_of(offs_k, 8)
    tl.max_contiguous(offs_m, BLOCK_SIZE_M)
    tl.max_contiguous(offs_n, BLOCK_SIZE_N)

    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    a_scale = a_scale[:, None]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # base pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    scale_n_idx = (pid_n * BLOCK_SIZE_N) // 128

    for k_tile in range(0, num_k_tiles):
        k_offsets = k_tile * BLOCK_SIZE_K + offs_k

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        scale_k_idx = (k_tile * BLOCK_SIZE_K) // 128
        b_scale = tl.load(
            b_scale_ptr + scale_k_idx * stride_b_scale_k + scale_n_idx * stride_b_scale_n
        )
        acc += tl.dot(a, b, out_dtype=tl.float32) * b_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    acc = acc * a_scale

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if OUT_DTYPE_BF16:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(c_ptrs, acc, mask=c_mask)


# ============================================================
# Decode kernel: SPLIT_K > 1
# atomic add into FP32 output buffer
# ============================================================

@triton.autotune(
    configs=_DECODE_CONFIGS,
    key=["N", "K", "SPLIT_K"],
)
@triton.jit
def fp8_block_gemm_splitk_kernel(
    # pointers
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,

    # shapes
    M, N, K,

    # strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_b_scale_k, stride_b_scale_n,

    # compile-time constants
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_split = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _compute_pid_mn(pid_mn, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    tl.multiple_of(offs_k, 8)
    tl.max_contiguous(offs_m, BLOCK_SIZE_M)
    tl.max_contiguous(offs_n, BLOCK_SIZE_N)

    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    a_scale = a_scale[:, None]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    total_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tiles_per_split = tl.cdiv(total_k_tiles, SPLIT_K)

    start_tile = pid_split * tiles_per_split
    end_tile = tl.minimum(start_tile + tiles_per_split, total_k_tiles)

    scale_n_idx = (pid_n * BLOCK_SIZE_N) // 128

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (start_tile * BLOCK_SIZE_K + offs_k)[None, :] * stride_ak
    b_ptrs = b_ptr + (start_tile * BLOCK_SIZE_K + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

    for k_tile in range(start_tile, end_tile):
        k_offsets = k_tile * BLOCK_SIZE_K + offs_k

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        scale_k_idx = (k_tile * BLOCK_SIZE_K) // 128
        b_scale = tl.load(
            b_scale_ptr + scale_k_idx * stride_b_scale_k + scale_n_idx * stride_b_scale_n
        )

        acc += tl.dot(a, b, out_dtype=tl.float32) * b_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    acc = acc * a_scale

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.atomic_add(c_ptrs, acc, mask=c_mask)


# ============================================================
# launcher
# ============================================================

def triton_fp8_block_gemm_optimized(
    x_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    out: torch.Tensor | None = None,
    block_size_k: int = 128,
    split_k: int | None = None,
    return_bf16: bool = True,
) -> torch.Tensor:

    assert x_fp8.is_cuda and weight_fp8.is_cuda
    assert x_scale.is_cuda and weight_scale_inv.is_cuda
    assert x_fp8.ndim == 2 and weight_fp8.ndim == 2
    assert x_scale.ndim == 1
    assert weight_scale_inv.ndim == 2

    M, K = x_fp8.shape
    K_w, N = weight_fp8.shape
    assert K == K_w, f"K mismatch: {K} vs {K_w}"
    assert block_size_k == 128, "Current scale indexing logic assumes 128-wide K blocks."

    # heuristic
    if split_k is None:
        if M <= 16:
            split_k = 4  
        elif M <= 64:
            split_k = 4 
        elif M <= 128:
            split_k = 2 
        else:
            split_k = 1 

    if split_k == 1:
        if out is None:
            out = torch.empty((M, N), device=x_fp8.device,
                              dtype=torch.bfloat16 if return_bf16 else torch.float32)

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        fp8_block_gemm_prefill_kernel[grid](
            x_fp8, weight_fp8, out, x_scale, weight_scale_inv,
            M, N, K,
            x_fp8.stride(0), x_fp8.stride(1),
            weight_fp8.stride(0), weight_fp8.stride(1),
            out.stride(0), out.stride(1),
            weight_scale_inv.stride(0), weight_scale_inv.stride(1),
            # BLOCK_SIZE_K=block_size_k,
            OUT_DTYPE_BF16=(out.dtype == torch.bfloat16),
        )
        return out

    if out is None:
        out_fp32 = torch.zeros((M, N), device=x_fp8.device, dtype=torch.float32)
    else:
        assert out.dtype in (torch.float32, torch.bfloat16)
        if out.dtype == torch.float32:
            out.zero_()
            out_fp32 = out
        else:
            out_fp32 = torch.zeros((M, N), device=x_fp8.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k,
    )

    fp8_block_gemm_splitk_kernel[grid](
        x_fp8, weight_fp8, out_fp32, x_scale, weight_scale_inv,
        M, N, K,
        x_fp8.stride(0), x_fp8.stride(1),
        weight_fp8.stride(0), weight_fp8.stride(1),
        out_fp32.stride(0), out_fp32.stride(1),
        weight_scale_inv.stride(0), weight_scale_inv.stride(1),
        #BLOCK_SIZE_K=block_size_k,
        SPLIT_K=split_k,
    )

    if out is not None and out.dtype == torch.float32:
        return out
    return out_fp32.to(torch.bfloat16) if return_bf16 else out_fp32
