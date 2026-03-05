from nanovllm.layers.quantization.quant_configBase import QuantizeMethodBase
from nanovllm.layers.quantization.quant_configBase import QuantizationConfig

from ..linear import LinearBase, ReplicatedLinear, ColumnParallelLinear

import torch
from torch import nn
import torch.distributed as dist




class AWQConfig(QuantizationConfig):
    def__init__(
        self,
        weight_bit: int,
        group_size: int,
        zero_point: bool,
        module_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight_bit = weight_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.module_to_not_convert = module_to_not_convert or []
        if self.weight_bit != 4:
            raise ValueError("Only 4-bit weight quantization is supported for AWQ.")
        # 代表一个int32 类型里包含了8个4-bit的weight.
        self.pack_factor = 32 // self.weight_bit


    # 声明硬件门槛,限定GPU型号
    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75
    

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)


    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        return None



    """
    Question: is it necessary for huggingface to Nano-vllm name mapper.
    """


class AWQQKVParallelLinear(QKVParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        head_num: int,
        head_num_kv: int | None = None,
        bias: bool = False,
        group_size: int = 128,
        weight_bits: int = 4,
    ):
        # 1. 调用父类初始化，算好单卡 local 的头数和 output_size
        super().__init__(hidden_size, head_size, head_num, head_num_kv, bias)
        
        # 2. ⚠️ 极其关键：删掉基类申请的 FP16 空白权重，释放显存！
        del self.weight
        if hasattr(self, "bias") and self.bias is not None:
            # 如果你有自己的量化 bias 处理，也可以在这里调整
            pass

        self.pack_factor = 32 // weight_bits  # 8
        self.group_size = group_size
        
        # AWQ 算子通常要求权重矩阵是转置的格式 [in_features, out_features]
        # 注意：这里的 self.local_output_size 是在父类计算并除以 tp_size 后的总 QKV 宽度
        local_out_features = (self.num_head + self.num_head_kv * 2) * self.head_size
        num_groups = hidden_size // self.group_size
        
        # 3. 申请量化专属的三大件 (注意 dtype 和缩小的维度)
        self.qweight = nn.Parameter(
            torch.empty(hidden_size, local_out_features // self.pack_factor, dtype=torch.int32),
            requires_grad=False
        )
        self.qzeros = nn.Parameter(
            torch.empty(num_groups, local_out_features // self.pack_factor, dtype=torch.int32),
            requires_grad=False
        )
        self.scales = nn.Parameter(
            torch.empty(num_groups, local_out_features, dtype=torch.float16),
            requires_grad=False
        )

        # 挂载专属的 weight_loader
        self.qweight.weight_loader = self.weight_loader
        self.qzeros.weight_loader = self.weight_loader
        self.scales.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, load_shard_id: str):
        """
        :param param: 当前正在加载的目标参数 (self.qweight, self.qzeros, 或 self.scales)
        :param loaded_weight: 引擎从磁盘读出的全局权重张量
        :param load_shard_id: "q", "k", 或 "v"
        """
        param_data = param.data
        assert load_shard_id in ["q", "k", "v"]

        # 1. 判定当前参数是否被压缩 (Packed)
        # qweight 和 qzeros 的列数被压缩了 8 倍，而 scales 是正常的 FP16
        is_packed = (param is self.qweight) or (param is self.qzeros)
        pack_factor = self.pack_factor if is_packed else 1

        # 2. 计算【逻辑维度】的 offset 和 size (也就是原始未量化时的 FP16 维度)
        if load_shard_id == "q":
            logical_shard_size = self.num_head * self.head_size
            logical_shard_offset = 0
        elif load_shard_id == "k":
            logical_shard_size = self.num_head_kv * self.head_size
            logical_shard_offset = self.num_head * self.head_size
        else: # "v"
            logical_shard_size = self.num_head_kv * self.head_size
            logical_shard_offset = (self.num_head + self.num_head_kv) * self.head_size

        # 3. 转化为【物理维度】的 offset 和 size
        physical_shard_size = logical_shard_size // pack_factor
        physical_shard_offset = logical_shard_offset // pack_factor

        # 4. 执行量化版的三步走：圈地、切片、填空
        
        # AWQ 的列并行切分维度通常是 dim=1 (因为它的形状是 [in, out])
        awq_tp_dim = 1 
        
        # 第一步：在目标大显存上圈出对应的 Q/K/V 物理边界
        param_data = param_data.narrow(awq_tp_dim, physical_shard_offset, physical_shard_size)
        
        # 第二步：将全局磁盘张量均分为 tp_size 份，取属于本卡的切片
        loaded_weight_shard = loaded_weight.chunk(self.tp_size, dim=awq_tp_dim)[self.tp_rank]
        
        # ⚠️ 严谨性校验：确保切出来的物理尺寸完全匹配
        assert param_data.shape == loaded_weight_shard.shape, \
            f"Shape mismatch: {param_data.shape} vs {loaded_weight_shard.shape}"
            
        # 第三步：底层内存覆写
        param_data.copy_(loaded_weight_shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear 已经没用了，必须调用底层 Triton/CUDA Kernel
        # import triton_kernels
        # return triton_kernels.awq_gemm(x, self.qweight, self.scales, self.qzeros, self.pack_factor)
        raise NotImplementedError("Need AWQ INT4 Kernel here.")