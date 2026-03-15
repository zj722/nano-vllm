import torch
import triton
import triton.language as tl
# 确保你的相对路径或绝对路径正确
from nanovllm.layers.quantization_fp8.kernals_fp8 import triton_fp8_block_gemm, fp8_split_k_gemm_kernel

# ========================================================
# 性能大比拼 (Benchmark)
# ========================================================
def main():
    # 锁定随机种子，保证每次测试数据一致
    torch.manual_seed(0)

    # 模拟大模型 Prefill 阶段的巨大矩阵 (完美符合 128 的倍数)
    M, N, K = 1024, 4096, 4096
    print(f"🔥 Testing Prefill Matrix Size: M={M}, N={N}, K={K}\n")

    # 1. 准备纯 BF16 的数据 (给 PyTorch cuBLAS 用)
    a_bf16 = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    b_bf16 = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)

    # 2. 准备纯 FP8 的数据 (给你的 Triton Kernel 用)
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.to(torch.float8_e4m3fn)

    # 🚨 3. 为你的 FP8 Kernel 准备必须的 Scale 和 Output 显存
    # x_scale: [M] 维度的 Per-token scale
    x_scale = torch.ones((M,), dtype=torch.float32, device='cuda')
    # weight_scale_inv: Block-wise 的 scale
    weight_scale_inv = torch.ones(((K + 127) // 128, (N + 127) // 128), dtype=torch.float32, device='cuda')
    # output_fp32: 用于 Split-K 原子累加的安全底板
    output_fp32 = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    # ========================================================
    # 验证正确性 (Sanity Check)
    # ========================================================
    print("🔍 正在验证计算正确性...")
    out_ref = torch.matmul(a_bf16, b_bf16)
    
    # 注意：你的 Kernel 返回的是降维后的 BF16 结果
    out_triton = triton_fp8_block_gemm(
        a_fp8, b_fp8, x_scale, weight_scale_inv, output_fp32
    )
    
    max_diff = torch.max(torch.abs(out_ref - out_triton)).item()
    print(f"✅ 最大绝对误差 (Max Diff): {max_diff:.4f} (由于 FP8 精度截断，此误差在 0.5 左右属正常现象)\n")
    print("-" * 50)

    # ========================================================
    # 开始测速
    # ========================================================
    # 测速 1: PyTorch 原生 BF16 matmul (底层的巅峰 cuBLASLt)
    pytorch_ms = triton.testing.do_bench(lambda: torch.matmul(a_bf16, b_bf16))
    
    # 测速 2: 你的 Triton FP8 Kernel (切记传入所有必要参数)
    # 注意：为了防止上一次 run 污染 output_fp32，在 lambda 里每次需要将其清零 (可选，仅影响严谨度不影响速度)
    triton_ms = triton.testing.do_bench(
        lambda: triton_fp8_block_gemm(a_fp8, b_fp8, x_scale, weight_scale_inv, output_fp32)
    )

    # 计算 TFLOPS (每秒万亿次浮点运算)
    flops = 2 * M * N * K
    pytorch_tflops = (flops / (pytorch_ms * 1e-3)) / 1e12
    triton_tflops = (flops / (triton_ms * 1e-3)) / 1e12

    # 输出结果
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