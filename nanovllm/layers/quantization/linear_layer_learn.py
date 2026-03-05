import torch 
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

# must be multiple 
def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator



class LinearBase (nn.module):
    def __init__ (
        self,
        # following two are one GPU local parameters
        input_size : int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim # 0 or 1, parallel in row / column
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # nn.parameter is a subclass of nn.tensor, so weight is a nn.prameter object.
        # weight_loader is a new attribute add to weight object.
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # self.weightloader is not defined here, but will be defined later in subclass.
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # register_paramter is a method of nn.module.
            self.register_parameter("bias", None)

        def forward (self, x:torch.Tensor) -> torch.Tensor:
            raise NotImplementedError



# no parallel, all gpus have the same weight and bias
class ReplicatedLinear(LinearBase):
    def __init__ (
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# parallel in column, each gpu has a shard of weight and bias, output is global output_size
class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):  
        # tp.size is used in super.ini__. and it need to be calculated beforehand.
        tp.size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)
    
    def weight_loader(self, param:nn.Parameter, loaded_weight: toch.tensor):
        param_data = param.data
        shard_size = param_data.size(self.to_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)



# Declare the partition loigic of qkv on one gpu, high level partition among gpus
# are declared in cloumnParallelLinear
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int, # head dim
        head_num: int, # q head number
        head_num_kv: int | None = None, # kv head number two seperate variable for GQA purpose
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads # if kv head num is none, means is not GQA form, kv head = q head number
        self.head_size = head_size
        self.num_head = divide(head_num, tp_size) # partition q heads
        self.num_head_kv = divide(head_num_kv, tp_size)
        output_size = (head_num + head_num_kv * 2) * self.head_size # total output dimension of stacked Wq, Wk, and Wv
        super().__init__ (hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, load_share_id: str):
        param_data = param.data
        assert loaded_shared_id in["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_head * self.head_size # q part
            shard_offset = 0
        elif loaded_shared_id == "k"
            shard_size = self.num_head_kv * self. head_size
            shard_offset = self.num_heads * self.head_size # start after q
        else:
            shard_size = self.num_head_kv * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shared_offset, shared_size) # 在大的param 中圈出一个写入的目标区域,这个区域大小等于当前要写入的内容wq/Wk/Wv)
        loaded_weight = loaded_weight.chunk(self.tp_size, selff.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)



"""
if 适配quantization
改变的（适配）：物理显存的申请尺寸（torch.empty 变小了）、参数的数量（新增了 scales 等）、
底层前向计算的 Kernel（F.linear 换成自定义 CUDA Kernel）。

算子直接使用triton定义一个,然后把F.linear 替换成triton的启动函数就行.
"""