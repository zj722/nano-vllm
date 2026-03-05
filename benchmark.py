import os
import time
import torch
from nanovllm import LLM, SamplingParams

def run_advanced_benchmark(llm, batch_size, input_len, output_len):
    # 造固定长度的假数据
    prompt_token_ids = [[100 for _ in range(input_len)] for _ in range(batch_size)]
    
    # ---------------------------------------------------------
    # 步骤 1: 测量 TTFT (只生成 1 个 Token，纯 Prefill)
    # ---------------------------------------------------------
    sp_prefill = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)
    
    # 预热 (必须预热一次，防止首次显存分配干扰)
    llm.generate(prompt_token_ids[:1], sp_prefill, use_tqdm=False)
    torch.cuda.synchronize()
    
    # 开始测 TTFT
    t0 = time.perf_counter()
    llm.generate(prompt_token_ids, sp_prefill, use_tqdm=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    ttft_time = t1 - t0  # 这就是 Prefill 的绝对耗时！
    
    # ---------------------------------------------------------
    # 步骤 2: 测量总耗时 (生成 output_len 个 Token)
    # ---------------------------------------------------------
    sp_full = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
    
    t2 = time.perf_counter()
    llm.generate(prompt_token_ids, sp_full, use_tqdm=False)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    
    total_time = t3 - t2
    
    # ---------------------------------------------------------
    # 步骤 3: 科学算账 (分离 Prefill 和 Decode)
    # ---------------------------------------------------------
    # Decode 阶段纯耗时 = 总时间 - Prefill 时间
    decode_time = total_time - ttft_time
    
    # 我们在这个阶段真正生成的 Decode Token 数量是 (output_len - 1)
    decode_steps = output_len - 1
    
    # TPOT (Time Per Output Token) -> 衡量延迟 (Latency), 单位: 毫秒/步
    tpot_ms = (decode_time / decode_steps) * 1000 
    
    # 纯 Decode 吞吐量 (Throughput) -> 单位: tok/s
    decode_throughput = (batch_size * decode_steps) / decode_time
    
    return ttft_time, tpot_ms, decode_throughput

def main():
    print("🚀 Initializing Nano-vLLM Engine...")
    path = os.path.expanduser("/mnt/d/huggingface/Qwen3-0.6B-FP8/")
    
    # 🚨 极度关键：enforce_eager=False，强行开启 CUDA Graph！
    llm = LLM(path, enforce_eager=False, max_model_len=4096)
    
    INPUT_LEN = 512
    OUTPUT_LEN = 128
    BATCH_SIZES = [1, 4, 16, 32, 64] 
    
    print("\n" + "="*70)
    print(f"{'Batch':<6} | {'TTFT (Prefill)':<18} | {'TPOT (Latency)':<18} | {'Decode Throughput':<18}")
    print("="*70)
    
    for bs in BATCH_SIZES:
        try:
            ttft, tpot, throughput = run_advanced_benchmark(llm, bs, INPUT_LEN, OUTPUT_LEN)
            print(f"{bs:<6} | {ttft:>6.3f} s            | {tpot:>6.2f} ms/step       | {throughput:>8.1f} tok/s")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{bs:<6} | ❌ OOM (Out of Memory)")
                break
            else:
                raise e

if __name__ == "__main__":
    main()