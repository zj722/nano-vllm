import torch
import triton
import triton.language as tl

# ========================================================
# 1. 搬运你之前写好的、带有 Auto-tune 的 Prefill GEMM Kernel
# ========================================================
# ========================================================
# 1. 搬运你之前写好的、带有 Auto-tune 的 Prefill GEMM Kernel
# ========================================================
@triton.autotune(
    configs=[
        # Config 1: 均衡型。SRAM 需求 = (16KB + 16KB) * 3 = 96KB (完美卡在 99KB 极限内！)
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        
        # Config 2: 大吞吐型。为了保持 N=256，把流水线降到 2 级。SRAM = (16KB + 32KB) * 2 = 96KB
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=2, num_warps=8),
        
        # Config 3: 高并发型。把 M 减小，换取更深的流水线来掩盖延迟。SRAM = (8KB + 16KB) * 4 = 96KB
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def dummy_fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        acc += tl.dot(a, b, out_dtype=tl.float32)
        
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_triton_fp8(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    dummy_fp8_gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c

# ========================================================
# 2. 性能大比拼 (Benchmark)
# ========================================================
def main():
    # 模拟大模型 Prefill 阶段的巨大矩阵
    M, N, K = 2048, 4096, 4096
    print(f"Testing Prefill Matrix Size: M={M}, N={N}, K={K}\n")

    # 准备纯 BF16 的数据 (给 PyTorch cuBLAS 用)
    a_bf16 = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    b_bf16 = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)

    # 准备纯 FP8 的数据 (给你的 Triton Kernel 用)
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.to(torch.float8_e4m3fn)

    # 测速 1: PyTorch 原生 BF16 Linear (底层的巅峰 cuBLAS)
    pytorch_ms = triton.testing.do_bench(lambda: torch.matmul(a_bf16, b_bf16))
    
    # 测速 2: 你的 Triton FP8 Kernel
    triton_ms = triton.testing.do_bench(lambda: run_triton_fp8(a_fp8, b_fp8))

    # 计算 TFLOPS (每秒万亿次浮点运算)
    # 乘法和加法算两次操作，所以是 2 * M * N * K
    flops = 2 * M * N * K
    pytorch_tflops = (flops / (pytorch_ms * 1e-3)) / 1e12
    triton_tflops = (flops / (triton_ms * 1e-3)) / 1e12

    print(f"🥊 Round 1: PyTorch BF16 (cuBLAS)")
    print(f"⏱️  Time: {pytorch_ms:.3f} ms")
    print(f"🚀 TFLOPS: {pytorch_tflops:.1f} TFLOPS\n")

    print(f"🥊 Round 2: Custom Triton FP8")
    print(f"⏱️  Time: {triton_ms:.3f} ms")
    print(f"🚀 TFLOPS: {triton_tflops:.1f} TFLOPS\n")
    
    speedup = pytorch_ms / triton_ms
    print(f"🏆 FP8 Speedup vs BF16: {speedup:.2f}x Faster!")

if __name__ == "__main__":
    main()