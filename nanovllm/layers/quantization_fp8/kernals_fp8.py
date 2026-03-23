import triton 
import triton.language as tl
import torch
import torch.nn


@triton.jit
def fused_dynamic_quantize_kernel(
    x_ptr,           # 输入: 高精度的 X [M, K] (BF16)
    y_ptr,           # 输出: 量化后的 X [M, K] (FP8)
    scale_ptr,       # 输出: 算出来的 Scale [M, 1] (FP32)
    M, K,            # 维度
    stride_xm, stride_xk,
    stride_ym, stride_yk,
    BLOCK_K: tl.constexpr # 让一个 Block 一口气处理完一整行 K
):
    # 每一个 Program (包工头) 只负责处理一个 Token (也就是矩阵的一行)
    pid_m = tl.program_id(0)
    
    # 获取这一行的内存偏移量
    x_row_ptr = x_ptr + pid_m * stride_xm
    y_row_ptr = y_ptr + pid_m * stride_ym
    scale_row_ptr = scale_ptr + pid_m * 1
    
    # 生成这一行 K 个元素的偏移序列 (假设 K 足够小能被装进 SRAM，比如 4096)
    offs_k = tl.arange(0, BLOCK_K)
    
    # ----------------------------------------------------
    # 一次性读入一整行的高精度数据 (BF16)
    # ----------------------------------------------------
    x = tl.load(x_row_ptr + offs_k * stride_xk, mask=offs_k < K, other=0.0)
    
    # ----------------------------------------------------
    # 在 SRAM (寄存器) 中极速完成 4 步操作，绝不碰显存！
    # ----------------------------------------------------
    # 1. 取绝对值
    abs_x = tl.math.abs(x)
    # 2. 找全局最大值
    max_val = tl.max(abs_x, axis=0)
    # 防止除零
    max_val = tl.maximum(max_val, 1e-12)
    # 3. 算 Scale (FP8 e4m3 最大值是 448.0)
    scale = max_val / 448.0
    # 4. 执行量化并转换格式
    y = (x / scale).to(tl.float8e4nv) # tl.float8e4nv 是 Triton 里的 e4m3 格式
    
    # ----------------------------------------------------
    # 把最终结果写回显存
    # ----------------------------------------------------
    tl.store(y_row_ptr + offs_k * stride_yk, y, mask=offs_k < K)
    tl.store(scale_row_ptr, scale)

# Python 包装函数
def triton_dynamic_quantize(x: torch.Tensor):
    M, K = x.shape
    # 提前挖好输出坑位
    x_fp8 = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    
    # 寻找比 K 大的最小的 2 的幂次方 (Triton 要求 BLOCK_SIZE 必须是 2 的幂)
    BLOCK_K = triton.next_power_of_2(K)
    
    # Grid 极其简单：有几行 (M)，就派几个 Block 过去
    grid = (M, )
    
    fused_dynamic_quantize_kernel[grid](
        x, x_fp8, x_scale,
        M, K,
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        BLOCK_K=BLOCK_K
    )
    
    return x_fp8, x_scale





@triton.jit
def fp8_split_k_gemm_kernel(
    # --- 内存指针 ---
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
    M, N, K,
    
    # --- 内存步长 ---
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_b_scale_k, stride_b_scale_n,
    
    # --- 编译期常量 ---
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr  # 核心魔法：K 维度的切分份数
):
    # -----------------------------------------------------------
    # 1. 三维空间定位：行 (M), 列 (N), 以及 深度切片 (K)
    # -----------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2) # 获取当前负责的是第几个 K 切片

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # 2. 提前加载 Activation Scale
    # -----------------------------------------------------------
    a_scale_ptrs = a_scale_ptr + offs_m
    a_scale = tl.load(a_scale_ptrs, mask=offs_m < M, other=0.0) 
    a_scale = tl.expand_dims(a_scale, 1)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # 3. Split-K 核心计算逻辑：算出自己该负责哪几段 K
    # -----------------------------------------------------------
    total_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    blocks_per_split = tl.cdiv(total_k_blocks, SPLIT_K)
    
    # 计算当前 pid_k 的起止位置
    start_k = pid_k * blocks_per_split
    end_k = start_k + blocks_per_split
    if end_k > total_k_blocks:
        end_k = total_k_blocks

    # 只循环属于自己的那一段 K
    for k in range(start_k, end_k):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        scale_k_idx = (k * BLOCK_SIZE_K) // 128
        scale_n_idx = (pid_n * BLOCK_SIZE_N) // 128
        
        b_scale_ptrs = b_scale_ptr + (scale_k_idx * stride_b_scale_k + scale_n_idx * stride_b_scale_n)
        b_scale = tl.load(b_scale_ptrs)
        # # 加载 Block Scale (精妙之处：k 是绝对坐标，直接用依然完美映射！)
        # b_scale_ptrs = b_scale_ptr + (k * stride_b_scale_k + pid_n * stride_b_scale_n)
        # b_scale = tl.load(b_scale_ptrs)
        
        # 硬件 FP8 乘法
        local_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        local_acc += tl.dot(a, b,local_acc,  out_dtype=tl.float32)
        local_acc = local_acc * b_scale
        
        acc += local_acc

    acc = acc * a_scale
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)


    #tl.atomic_add(c_ptrs, acc, mask=mask)
    if SPLIT_K == 1:
        # 🚀 PREFILL 模式 (大矩阵)：
        # 没有多 Block 竞争，不用原子加法！
        # 在寄存器里直接转成 BF16，然后暴力覆盖写入目标显存
        c_bf16 = acc.to(tl.bfloat16)
        tl.store(c_ptrs, c_bf16, mask=mask)
    else:
        # 🏎️ DECODE 模式 (小矩阵)：
        # K 被切碎了，必须用高精度 FP32 做原子加法，防止精度爆炸
        tl.atomic_add(c_ptrs, acc, mask=mask)


# # ==========================================
# # 智能启动器 (The Smart Launcher)
# # ==========================================
def triton_fp8_block_gemm(
    x_fp8: torch.Tensor, 
    weight_fp8: torch.Tensor, 
    x_scale: torch.Tensor, 
    weight_scale_inv: torch.Tensor, 
    output_fp32: torch.Tensor,
    block_size_k: int = 128
) -> torch.Tensor:

    M, K = x_fp8.shape
    K_w, N = weight_fp8.shape
    assert K == K_w
    
    if M <= 32: 
        # Decode 阶段：矩阵极小，必须把 K 切碎来唤醒所有 GPU 核心！
        SPLIT_K = 2 # default 16
        BLOCK_SIZE_M = 16   # 缩小 M 块尺寸，减少无效线程
        BLOCK_SIZE_N = 128 # default 128
        num_stages = 4 
        num_warps = 8
    else:
        # Prefill 阶段：矩阵巨大，GPU 已经忙不过来了，禁止切分 K，减少原子加法冲突！
        SPLIT_K = 1
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        num_stages = 3
        num_warps = 8
    
    # # Decode 阶段：矩阵极小，必须把 K 切碎来唤醒所有 GPU 核心！
    # SPLIT_K = 16 # default 16
    # BLOCK_SIZE_M = 16   # 缩小 M 块尺寸，减少无效线程
    # BLOCK_SIZE_N = 128 # default 128
    # num_stages = 4 
    # num_warps = 4
    
    BLOCK_SIZE_K = block_size_k

    # -------------------------------------------------------------
    # 🚨 极其关键的安全设计：输出坑位必须是 ZERO 初始化的 FP32！
    # 因为底层有多个线程块用 Atomic Add 往里累加，如果不清零，结果会带上显存垃圾。
    # 用 FP32 累加能保证精度绝对不掉，算完再转 BF16。
    # -------------------------------------------------------------
   

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        SPLIT_K  # 第三维度的网格数量！
    )

    fp8_split_k_gemm_kernel[grid](
        x_fp8, weight_fp8, output_fp32, x_scale, weight_scale_inv,
        M, N, K,
        x_fp8.stride(0), x_fp8.stride(1),
        weight_fp8.stride(0), weight_fp8.stride(1),
        output_fp32.stride(0), output_fp32.stride(1),
        weight_scale_inv.stride(0), weight_scale_inv.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages = num_stages
    )

    # 累加完毕，完美降维回 BF16 送给下一层网络
    return output_fp32.to(torch.bfloat16)





@triton.jit
def _fused_dequantize_weight_kernel(
    weight_ptr, scale_ptr, output_ptr,
    K, N, block_size: tl.constexpr,
    stride_wk, stride_wn,
    stride_sk, stride_sn,
    stride_ok, stride_on,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 拿到当前线程块的坐标
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算当前块要处理的 k 和 n 的全局索引
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 边界保护掩码
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    # 1. 读取 FP8 权重
    w_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    w_fp8 = tl.load(w_ptrs, mask=mask, other=0.0)

    # 2. 极其聪明的寻址：通过整除 block_size，直接定位到正确的 Scale！
    # 这样就彻底消灭了 repeat_interleave！
    scale_k_idx = offs_k // block_size
    scale_n_idx = offs_n // block_size
    s_ptrs = scale_ptr + (scale_k_idx[:, None] * stride_sk + scale_n_idx[None, :] * stride_sn)
    scale = tl.load(s_ptrs, mask=mask, other=1.0)

    # 3. 在 SRAM 里瞬间完成数据类型转换和乘法
    w_f32 = w_fp8.to(tl.float32)
    w_bf16 = (w_f32 * scale).to(tl.bfloat16)

    # 4. 把最终纯净的 BF16 权重写回显存
    out_ptrs = output_ptr + (offs_k[:, None] * stride_ok + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, w_bf16, mask=mask)

def triton_dequantize_weight(weight_fp8: torch.Tensor, scale_inv: torch.Tensor, block_size: int) -> torch.Tensor:
    K, N = weight_fp8.shape
    # 只挖一个最终结果的坑位，绝不产生临时变量！
    output_bf16 = torch.empty((K, N), device=weight_fp8.device, dtype=torch.bfloat16)

    # Triton 调优块大小 (保持和你的 scale block_size 一致或者更小)
    BLOCK_K = 128
    BLOCK_N = 128

    grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N))

    _fused_dequantize_weight_kernel[grid](
        weight_fp8, scale_inv, output_bf16,
        K, N, block_size,
        weight_fp8.stride(0), weight_fp8.stride(1),
        scale_inv.stride(0), scale_inv.stride(1),
        output_bf16.stride(0), output_bf16.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=4
    )

    return output_bf16





# ============================================================
# autotune configs
# ============================================================

_PREFILL_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=4,
    ),
]

_DECODE_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=5,
    ),
]


# ============================================================
# helper: grouped pid mapping for better L2 locality
# Triton matmul tutorial style
# ============================================================

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


# ============================================================
# Prefill kernel: SPLIT_K == 1
# No atomic
# ============================================================

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

        # same math as your original kernel
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

    # initialize pointers at split-local start
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

    # split-k path must accumulate into FP32 output buffer
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
    """
    Semantics preserved w.r.t. the original kernel:
      acc += dot(a_block, b_block) * b_scale[k_block, n_block]
      acc *= a_scale[row]
    """

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
        # decode-like small-M: use split-k to improve occupancy
        if M <= 16:
            split_k = 4
        elif M <= 32:
            split_k = 2
        else:
            split_k = 1

    if split_k == 1:
        # direct output path
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

    # split-k path: accumulate in fp32 scratch, then cast if needed
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
