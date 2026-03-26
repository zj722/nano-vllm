import torch
import triton
import triton.language as tl


from nanovllm.layers.quantization_fp8.kernals_fp8 import (
    triton_fp8_block_gemm_optimized,
)


def benchmark_op(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"

    # ------------------------------------------------------------
    # modify this for different test scenarios
    # ------------------------------------------------------------
    M, N, K = 16 * 1, 1024, 4096
    print(f"🔥 Testing Matrix Size: M={M}, N={N}, K={K}\n")

    a_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)

    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.to(torch.float8_e4m3fn)

    x_scale = torch.ones((M,), dtype=torch.float32, device="cuda")

    weight_scale_inv = torch.ones(
        ((K + 127) // 128, (N + 127) // 128),
        dtype=torch.float32,
        device="cuda"
    )
    output_fp32_old = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    output_fp32_new = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    print("test compute accuracy")

    out_ref = torch.matmul(a_bf16, b_bf16)
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
    max_diff_new = torch.max(torch.abs(out_ref - out_new)).item()
    print(f"max error: {max_diff_new:.4f}")
    print("-" * 60)

    print("warmup")
    _ = torch.matmul(a_bf16, b_bf16)

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
    print("finish warmup\n")


    pytorch_ms = benchmark_op(
        lambda: torch.matmul(a_bf16, b_bf16),
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

    flops = 2 * M * N * K

    pytorch_tflops = (flops / (pytorch_ms * 1e-3)) / 1e12
    new_triton_tflops = (flops / (new_triton_ms * 1e-3)) / 1e12

    print("Benchmark Results\n")

    print("PyTorch BF16 (cuBLAS/cuBLASLt)")
    print(f"Time:   {pytorch_ms:.3f} ms")
    print(f"TFLOPS: {pytorch_tflops:.1f} TFLOPS\n")


    print("New Triton FP8 Kernel")
    print(f"Time:   {new_triton_ms:.3f} ms")
    print(f"TFLOPS: {new_triton_tflops:.1f} TFLOPS\n")

    print("Relative Speed")
    print(f"New Triton vs PyTorch: {pytorch_ms / new_triton_ms:.2f}x")


if __name__ == "__main__":
    main()