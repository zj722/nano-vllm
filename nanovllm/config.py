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
    quantization: Optional[str] = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # following will get the config.json from huggingface downloaded model root folder.
        # also convert json to a AutoConfig object for later use.
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

        if self.quantization is None:
            # 逻辑 A: 检查路径名是否包含 fp8 
            if "fp8" in self.model.lower():
                self.quantization = "fp8"
            # 逻辑 B: 检查 config.json 里的量化配置 (兼容 HF 官方格式)
            elif hasattr(self.hf_config, "quantization_config"):
                quant_method = self.hf_config.quantization_config.get("quant_method")
                if quant_method:
                    self.quantization = quant_method.lower()
