import torch
import triton
import triton.language as tl

# 旧 kernel
from nanovllm.layers.quantization_fp8.kernals_fp8 import (
    triton_fp8_block_gemm,
)

# 新 kernel
from nanovllm.layers.quantization_fp8.kernals_fp8 import (
    triton_fp8_block_gemm_optimized,
)


def benchmark_op(fn, warmup=10, rep=100):
    """
    简单封装 benchmark，避免把首次 lazy compile 明显混进去
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"

    # ------------------------------------------------------------
    # 你可以改这三个维度测试不同场景
    # Prefill 大矩阵
    # ------------------------------------------------------------
    M, N, K = 128*1024, 1024, 4096
    print(f"🔥 Testing Matrix Size: M={M}, N={N}, K={K}\n")

    # ------------------------------------------------------------
    # 1. 准备 BF16 baseline 数据
    # ------------------------------------------------------------
    a_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)

    # ------------------------------------------------------------
    # 2. 准备 FP8 数据
    # ------------------------------------------------------------
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.to(torch.float8_e4m3fn)

    # row-wise activation scale: [M]
    x_scale = torch.ones((M,), dtype=torch.float32, device="cuda")

    # block-wise weight scale: [K_block, N_block]
    weight_scale_inv = torch.ones(
        ((K + 127) // 128, (N + 127) // 128),
        dtype=torch.float32,
        device="cuda"
    )

    # ------------------------------------------------------------
    # 3. 为旧 / 新 kernel 准备输出 buffer
    # ------------------------------------------------------------
    output_fp32_old = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    output_fp32_new = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    # ------------------------------------------------------------
    # 4. 正确性检查
    # ------------------------------------------------------------
    print("🔍 正在验证计算正确性...")

    out_ref = torch.matmul(a_bf16, b_bf16)

    # 旧 kernel
    output_fp32_old.zero_()
    out_old = triton_fp8_block_gemm(
        a_fp8,
        b_fp8,
        x_scale,
        weight_scale_inv,
        output_fp32_old
    )

    # 新 kernel
    output_fp32_new.zero_()
    out_new = triton_fp8_block_gemm_optimized(
        x_fp8=a_fp8,
        weight_fp8=b_fp8,
        x_scale=x_scale,
        weight_scale_inv=weight_scale_inv,
        out=output_fp32_new,
        block_size_k=128,
        split_k=None,
        return_bf16=True
    )

    max_diff_old = torch.max(torch.abs(out_ref - out_old)).item()
    max_diff_new = torch.max(torch.abs(out_ref - out_new)).item()

    print(f"✅ 原 Triton kernel 最大绝对误差: {max_diff_old:.4f}")
    print(f"✅ 新 Triton kernel 最大绝对误差: {max_diff_new:.4f}")
    print("（由于 FP8 截断，存在明显误差是正常的，重点看新旧 kernel 是否数值一致量级）\n")
    print("-" * 60)

    # ------------------------------------------------------------
    # 5. 预热（避免首次编译污染 steady-state benchmark）
    # ------------------------------------------------------------
    print("🔥 预热中...")

    # PyTorch
    _ = torch.matmul(a_bf16, b_bf16)

    # 旧 kernel
    output_fp32_old.zero_()
    _ = triton_fp8_block_gemm(
        a_fp8,
        b_fp8,
        x_scale,
        weight_scale_inv,
        output_fp32_old
    )

    # 新 kernel
    output_fp32_new.zero_()
    _ = triton_fp8_block_gemm_optimized(
        x_fp8=a_fp8,
        weight_fp8=b_fp8,
        x_scale=x_scale,
        weight_scale_inv=weight_scale_inv,
        out=output_fp32_new,
        block_size_k=128,
        split_k=None,
        return_bf16=True
    )

    torch.cuda.synchronize()
    print("✅ 预热完成\n")

    # ------------------------------------------------------------
    # 6. Benchmark
    # ------------------------------------------------------------
    pytorch_ms = benchmark_op(
        lambda: torch.matmul(a_bf16, b_bf16),
        warmup=10,
        rep=100
    )

    old_triton_ms = benchmark_op(
        lambda: triton_fp8_block_gemm(
            a_fp8,
            b_fp8,
            x_scale,
            weight_scale_inv,
            output_fp32_old.zero_()
        ),
        warmup=10,
        rep=100
    )

    new_triton_ms = benchmark_op(
        lambda: triton_fp8_block_gemm_optimized(
            x_fp8=a_fp8,
            weight_fp8=b_fp8,
            x_scale=x_scale,
            weight_scale_inv=weight_scale_inv,
            out=output_fp32_new.zero_(),
            block_size_k=128,
            split_k=None,
            return_bf16=True
        ),
        warmup=10,
        rep=100
    )

    # ------------------------------------------------------------
    # 7. 计算 TFLOPS
    # ------------------------------------------------------------
    flops = 2 * M * N * K

    pytorch_tflops = (flops / (pytorch_ms * 1e-3)) / 1e12
    old_triton_tflops = (flops / (old_triton_ms * 1e-3)) / 1e12
    new_triton_tflops = (flops / (new_triton_ms * 1e-3)) / 1e12

    # ------------------------------------------------------------
    # 8. 输出结果
    # ------------------------------------------------------------
    print("📊 Benchmark Results\n")

    print("🥊 PyTorch BF16 (cuBLAS/cuBLASLt)")
    print(f"⏱️  Time:   {pytorch_ms:.3f} ms")
    print(f"🚀 TFLOPS: {pytorch_tflops:.1f} TFLOPS\n")

    print("🥊 Original Triton FP8 Kernel")
    print(f"⏱️  Time:   {old_triton_ms:.3f} ms")
    print(f"🚀 TFLOPS: {old_triton_tflops:.1f} TFLOPS\n")

    print("🥊 New Triton FP8 Kernel")
    print(f"⏱️  Time:   {new_triton_ms:.3f} ms")
    print(f"🚀 TFLOPS: {new_triton_tflops:.1f} TFLOPS\n")

    print("📈 Relative Speed")
    print(f"Old Triton vs PyTorch: {pytorch_ms / old_triton_ms:.2f}x")
    print(f"New Triton vs PyTorch: {pytorch_ms / new_triton_ms:.2f}x")
    print(f"New Triton vs Old Triton: {old_triton_ms / new_triton_ms:.2f}x")


if __name__ == "__main__":
    main()