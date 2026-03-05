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
        local_acc += tl.dot(a, b, out_dtype=tl.float32)
        local_acc = local_acc * b_scale
        
        acc += local_acc

    # -----------------------------------------------------------
    # 4. 乘上 A_Scale，并使用 Atomic Add (原子加法) 写入显存
    # -----------------------------------------------------------
    acc = acc * a_scale
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # ⚠️ 极其关键：因为有多个 pid_k 都在算同一块 C 矩阵，必须用原子加法，否则数据会互相覆盖！
    tl.atomic_add(c_ptrs, acc, mask=mask)


# ==========================================
# 智能启动器 (The Smart Launcher)
# ==========================================
def triton_fp8_block_gemm(
    x_fp8: torch.Tensor, 
    weight_fp8: torch.Tensor, 
    x_scale: torch.Tensor, 
    weight_scale_inv: torch.Tensor, 
    block_size_k: int = 128
) -> torch.Tensor:
    
    M, K = x_fp8.shape
    K_w, N = weight_fp8.shape
    assert K == K_w
    
    # -------------------------------------------------------------
    # 🌟 智能换挡系统 (Heuristics) 🌟
    # 根据 M 的大小自动决定是否使用 Split-K
    # -------------------------------------------------------------
    if M <= 32: 
        # Decode 阶段：矩阵极小，必须把 K 切碎来唤醒所有 GPU 核心！
        SPLIT_K = 16 # default 16
        BLOCK_SIZE_M = 16   # 缩小 M 块尺寸，减少无效线程
        BLOCK_SIZE_N = 128 # default 128
        num_stages = 4 
        num_warps = 4
    else:
        # Prefill 阶段：矩阵巨大，GPU 已经忙不过来了，禁止切分 K，减少原子加法冲突！
        SPLIT_K = 1
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        num_stages = 3
        num_warps = 8
        
    BLOCK_SIZE_K = block_size_k

    # -------------------------------------------------------------
    # 🚨 极其关键的安全设计：输出坑位必须是 ZERO 初始化的 FP32！
    # 因为底层有多个线程块用 Atomic Add 往里累加，如果不清零，结果会带上显存垃圾。
    # 用 FP32 累加能保证精度绝对不掉，算完再转 BF16。
    # -------------------------------------------------------------
    output_fp32 = torch.zeros((M, N), device=x_fp8.device, dtype=torch.float32)

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
        num_stages=num_stages,
        num_warps=num_warps
    )

    # 累加完毕，完美降维回 BF16 送给下一层网络
    return output_fp32.to(torch.bfloat16)





# """
# m = seq_Len * batch
# k = hidden_size
# n = 
# """
# @triton.autotune(
#     configs=[
#         # 偏向 Prefill 的大 Block 配置 (需要更多 SRAM，需要更多 stages 来掩盖延迟)
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        
#         # 偏向 Decode 的小 Block 配置
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
#     ],
#     key=['M', 'N', 'K'], # 当这三个维度发生变化时，重新寻找最优解
# )
# @triton.jit
# def fp8_block_gemm_kernel(
#     # 1. 内存指针 (Pointers)
#     a_ptr,          # 激活值 X_fp8，形状 [M, K]
#     b_ptr,          # 权重 W_fp8，形状 [K, N] (假设已转置方便读取)
#     c_ptr,          # 输出 Y_bf16，形状 [M, N]
#     a_scale_ptr,    # 激活值 Scale，形状 [M, 1]
#     b_scale_ptr,    # 权重 Block Scale，形状 [K//128, N//128]

#     M, N, K,

#     # software defined task row size, may need multiple round for same block to execute.
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     stride_b_scale_k, stride_b_scale_n, # 权重 scale 的步长

#     BLOCK_SIZE_M: tl.constexpr, 
#     BLOCK_SIZE_N: tl.constexpr, 
#     BLOCK_SIZE_K: tl.constexpr, # set to 128, same to scaling scheme.
# ):
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
    
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
#     # -----------------------------------------------------------
#     # 阶段 2：提前加载属于这几行的 Activation Scale
#     # 因为 x_scale 是 per-token 的，它只跟 M (行号) 有关，整个 K 循环中都不变
#     # -----------------------------------------------------------
#     a_scale_ptrs = a_scale_ptr + offs_m
#     # a_scale is on sram
#     a_scale = tl.load(a_scale_ptrs, mask=offs_m < M, other=0.0) 
#     # 将形状从 [BLOCK_M] 变为 [BLOCK_M, 1]，方便后续做矩阵乘法时的广播
#     a_scale = tl.expand_dims(a_scale, 1)

#     # -----------------------------------------------------------
#     # 阶段 3：创建总累加器 (FP32 寄存器)
#     # -----------------------------------------------------------
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         # 计算当前这块 K 的偏移量
#         offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
#         # 1. 计算内存地址并加载 FP8 的 A 块和 B 块
#         """
#         broadcast add
#         A is a column vector, 
#         B is a row vector
#         A will expand itself to a square horizontally,
#         B will expand it self to a square vertically
#         finally do a elementwise add.
#         """
#         a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
#         # 从显存读入缓存，明确告诉 Triton 这是 FP8 格式
#         a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
#         b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
#         # 2. 加载当前这 128x128 块对应的 Weight Block Scale!
#         # 因为权重 Scale 也是按块存的，这里的索引计算非常精妙
#         b_scale_ptrs = b_scale_ptr + (k * stride_b_scale_k + pid_n * stride_b_scale_n)
#         # 读出一个单一的 float32 数值 (当前这 128x128 块的灵魂缩放因子)
#         b_scale = tl.load(b_scale_ptrs)
        
#         # 3. 硬件级魔法：FP8 的矩阵乘法！
#         # 结果 local_acc 是 [BLOCK_M, BLOCK_N] 大小的 FP32 寄存器矩阵
#         local_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#         local_acc += tl.dot(a, b, out_dtype=tl.float32)
        
#         # 4. 乘上局部的 Weight Scale (在高速缓存中进行，不碰显存)
#         local_acc = local_acc * b_scale
        
#         # 5. 累加到全局的累加器中
#         acc += local_acc

#     # -----------------------------------------------------------
#     # 阶段 5：Epilogue (收尾：乘上输入 Scale，转回 BF16 并写回显存)
#     # -----------------------------------------------------------
#     # 乘上阶段 2 提前准备好的 Activation Scale
#     acc = acc * a_scale
    
#     # 将最终结果强转回 BF16
#     c = acc.to(tl.bfloat16)
    
#     # 计算输出矩阵 C 的内存地址并写回
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))




# def triton_fp8_block_gemm(
#     x_fp8: torch.Tensor,               # 激活值 X: [M, K], float8_e4m3fn
#     weight_fp8: torch.Tensor,          # 权重 W: [K, N], float8_e4m3fn (注意：如果是 PyTorch Linear 的权重，请传 weight.t() 进来)
#     x_scale: torch.Tensor,             # 激活值 Scale: [M, 1], float32
#     weight_scale_inv: torch.Tensor,    # 权重 Block Scale: [K//128, N//128], float32
#     block_size_k: int = 128            # 你的分块大小
# ) -> torch.Tensor:
    
#     # -----------------------------------------------------------------
#     # 1. 获取物理维度并进行严格的防御性检查 (Sanity Checks)
#     # -----------------------------------------------------------------
#     M, K = x_fp8.shape
#     K_w, N = weight_fp8.shape
    
#     assert K == K_w, f"维度不匹配: X 的列数({K}) 必须等于 W 的行数({K_w})"
#     assert x_fp8.dtype == torch.float8_e4m3fn, "X 必须是 float8_e4m3fn"
#     assert weight_fp8.dtype == torch.float8_e4m3fn, "W 必须是 float8_e4m3fn"
#     assert x_scale.dtype == torch.float32, "x_scale 必须是 float32"
#     assert weight_scale_inv.dtype == torch.float32, "weight_scale_inv 必须是 float32"
    
#     # -----------------------------------------------------------------
#     # 2. 为 Kernel 准备输出的“空坑位”
#     # Triton 不会帮你 return 数据，你必须提前在显存里划好一块地盘
#     # -----------------------------------------------------------------
#     output_bf16 = torch.empty((M, N), device=x_fp8.device, dtype=torch.bfloat16)
    
#     # -----------------------------------------------------------------
#     # 3. 定义 Grid (计算网格：告诉 GPU 派多少个线程块去干活)
#     # -----------------------------------------------------------------
#     # triton.cdiv 是向上取整除法 (Ceil Divide)，比如 M=130, BLOCK_M=128，会分配 2 个 Block
#     def grid(META):
#         return (
#             triton.cdiv(M, META['BLOCK_SIZE_M']), 
#             triton.cdiv(N, META['BLOCK_SIZE_N']),
#         )
    
#     # -----------------------------------------------------------------
#     # 4. 召唤 Kernel，移交全部参数与指针！
#     # -----------------------------------------------------------------
#     fp8_block_gemm_kernel[grid](
#         # --- 传递数据指针 ---
#         a_ptr=x_fp8,
#         b_ptr=weight_fp8,
#         c_ptr=output_bf16,
#         a_scale_ptr=x_scale,
#         b_scale_ptr=weight_scale_inv,
        
#         # --- 传递形状维度 ---
#         M=M, N=N, K=K,
        
#         # --- 传递物理内存步长 (Strides) ---
#         # .stride(0) 告诉 Triton 跳到下一行要跨过多少元素
#         # .stride(1) 告诉 Triton 跳到下一列要跨过多少元素
#         stride_am=x_fp8.stride(0), stride_ak=x_fp8.stride(1),
#         stride_bk=weight_fp8.stride(0), stride_bn=weight_fp8.stride(1),
#         stride_cm=output_bf16.stride(0), stride_cn=output_bf16.stride(1),
#         stride_b_scale_k=weight_scale_inv.stride(0), 
#         stride_b_scale_n=weight_scale_inv.stride(1),
        
#         # # --- 传递编译期常量 (Meta-parameters) ---
#         # # 这些数字决定了 Triton 在底层申请多大的共享内存 (SRAM)
#         # # 对于 FP8 计算，128x128 是最能跑满 Tensor Core 带宽的黄金比例
#         # BLOCK_SIZE_M=128,
#         # BLOCK_SIZE_N=128,
#         # BLOCK_SIZE_K=block_size_k,
#     )
    
#     # 5. 满载着计算结果的坑位现在可以作为返回值交还给 PyTorch 了
#     return output_bf16






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