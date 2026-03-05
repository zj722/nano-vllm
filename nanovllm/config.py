import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    quantization: str | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # following will get the config.json from huggingface downloaded model root folder.
        # also convert json to a AutoConfig object for later use.
        self.hf_config = AutoConfig.from_pretrained(self.model)

        if self.quantization is None:
            quant_cfg = getattr(self.hf_config, "quantization_config", None)
            if quant_cfg:
                method = quant_cfg.get("quant_method")
                if method:
                    self.quantization = method.lower()
        self.hf_config.quantization = self.quantization
        
        print(f"\033[92m[DEBUG] Config detected quantization: {self.quantization}\033[0m")

        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
