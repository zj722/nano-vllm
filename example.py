import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

torch.compiler.allow_in_graph(flash_attn_varlen_func)
torch.compiler.allow_in_graph(flash_attn_with_kvcache)

def main():
    
    path = os.path.expanduser("/mnt/d/huggingface/Qwen3-0.6B-FP8/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)    

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = ["write a short story around 500 words"] * 2
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
