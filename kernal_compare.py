import torch
import os
import random 
from nanovllm.layers.quantization_fp8.kernals_fp8 import triton_dynamic_quantize, triton_fp8_block_gemm, triton_dequantize_weight
import triton
import torch.nn.functional as F
def run_benchmark():
    random.seed(0)
    torch.manual_seed(0)

    # 模拟你的输入场景
    num_seqs = 12

    max_input_len = 128
    
    prompt_lens = [max_input_len for _ in range(num_seqs)]
    M = sum(prompt_lens)
    K = 1536
    N = 1536

    print(f"🔥 模拟 Prefill 场景启动")
    print(f"Batch Size (Seq 数量): {num_seqs}")
    print(f"总 Token 数 (M): {M}")
    print(f"矩阵维度: M={M}, K={K}, N={N}")
    print("-" * 50)

    # ---------------------------------------------------------
    # 🚨 核心修改：使用 F.linear 标准的权重形状 [out_features, in_features] -> [N, K]
    # ---------------------------------------------------------
    x_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device='cuda')
    weight_bf16_linear = torch.randn((N, K), dtype=torch.bfloat16, device='cuda') 

    # 对于 FP8 Kernel，推理框架通常会在加载权重时，提前将其转置为 [K, N] 并变为连续内存
    x_fp8 = x_bf16.to(torch.float8_e4m3fn)
    weight_fp8 = weight_bf16_linear.t().contiguous().to(torch.float8_e4m3fn)
    
    # 初始化 Scale (Per-Token 和 Block-wise)
    x_scale = torch.ones((M,), dtype=torch.float32, device='cuda')
    weight_scale_inv = torch.ones(((K + 127) // 128, (N + 127) // 128), dtype=torch.float32, device='cuda')
    
    output_fp32 = torch.empty((M, N), dtype=torch.float32, device='cuda')

    # 1. 正确性验证
    print("🔍 正在验证计算正确性...")
    # 对比对象换成 torch.nn.functional.linear
    out_ref = F.linear(x_bf16, weight_bf16_linear)
    out_triton = triton_fp8_block_gemm(x_fp8, weight_fp8, x_scale, weight_scale_inv, output_fp32)
    
    max_diff = torch.max(torch.abs(out_ref - out_triton)).item()
    print(f"最大绝对误差 (Max Diff): {max_diff:.4f} (FP8 本身会掉精度，这是正常的)")
    print("-" * 50)

    # 2. 性能测试
    print("🚀 开始测速...")

    def benchmark_torch_linear():
        return F.linear(x_bf16, weight_bf16_linear)

    def benchmark_triton():
        return triton_fp8_block_gemm(x_fp8, weight_fp8, x_scale, weight_scale_inv, output_fp32)

    # 预热和测速
    ms_torch = triton.testing.do_bench(benchmark_torch_linear, warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(benchmark_triton, warmup=25, rep=100)

    # 计算 TFLOPS (2 * M * N * K 为乘加操作总数)
    tflops_torch = (2 * M * N * K) / (ms_torch * 1e-3) / 1e12
    tflops_triton = (2 * M * N * K) / (ms_triton * 1e-3) / 1e12

    # 输出结果
    print(f"{'Kernel':<25} | {'Latency (ms)':<15} | {'TFLOPS':<15}")
    print(f"{'-'*60}")
    print(f"{'torch.nn.functional.linear':<25} | {ms_torch:<15.3f} | {tflops_torch:<15.3f}")
    print(f"{'Triton FP8 Custom':<25} | {ms_triton:<15.3f} | {tflops_triton:<15.3f}")
    
    speedup = ms_torch / ms_triton
    print(f"{'-'*60}")
    if speedup > 1:
        print(f"🎉 结论: 你的 FP8 Kernel 比 PyTorch 原生 Linear 快了 {speedup:.2f} 倍！")
    else:
        print(f"⚠️ 结论: 你的 FP8 Kernel 慢了 {1/speedup:.2f} 倍。")

if __name__ == "__main__":
    run_benchmark()