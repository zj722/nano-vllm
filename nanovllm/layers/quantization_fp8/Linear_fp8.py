
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from .kernals_fp8 import triton_dynamic_quantize, triton_fp8_block_gemm, triton_dequantize_weight



def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# ==========================================
# 1. 核心基类 (Base Class)
# ==========================================
class LinearBase_fp8(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        block_size: int = 128,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.block_size = block_size

        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False
        )
        self.weight.weight_loader = self.weight_loader

        scale_output_size = divide(output_size, block_size)
        scale_input_size = divide(input_size, block_size)
        
        # scale should have name of weight_scale_inv, align with keys in .safetensors files
        self.weight_scale_inv = nn.Parameter(
            torch.empty(scale_output_size, scale_input_size, dtype=torch.float32),
            requires_grad=False
        )
        self.weight_scale_inv.scale_loader = self.scale_loader 

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter("bias", None)

    # expand scale to full weight shape and dequantized by doing element-wise multiplication
    def _dequantize_weight(self) -> torch.Tensor:
        # 一行代码，调用我们写好的光速算子
        return triton_dequantize_weight(self.weight, self.weight_scale_inv, self.block_size)

    def _dynamic_quantize_activation_per_token(self, x: torch.Tensor):
        """
        将高精度激活值动态量化为 FP8 e4m3 格式 (Token-wise)
        
        参数:
            x: 形状为 [Tokens, hidden_size] 的激活值张量 (通常是 bfloat16 或 float16)
            Token = seq_len * batch_size
        返回:
            x_fp8: 形状为 [Tokens, hidden_size] 的 FP8 激活值
            x_scale: 形状为 [Tokens, 1] 的缩放因子 (float32)
        """
        # 物理常识：FP8 e4m3 格式的最大可表示数值是 448.0
        FP8_MAX = 448.0
        
        # 阶段 1：找每一行的绝对最大值 (Amax)
        # dim=-1 代表沿着 input_size 这个维度找最大值
        # keepdim=True 让输出形状保持 [Tokens, 1]，方便后面做除法广播
        amax = x.abs().amax(dim=-1, keepdim=True) 
        
        # 防止出现全 0 的行导致除以 0 的惨剧，设一个极小的底线
        amax = torch.clamp(amax, min=1e-12)
        x_scale = (amax / FP8_MAX).to(torch.float32)
        x_fp8 = (x / x_scale).to(torch.float8_e4m3fn)
        
        return x_fp8, x_scale



# ==========================================
# 2. basic parallelism (Column / Row)
# ==========================================
class ColumnParallelLinear_fp8(LinearBase_fp8):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)
        # ==========================================
        # 新增：为 CUDA Graph 准备的“录像机”和“静态内存坑位”
        # ==========================================
        self.is_graph_captured = False
        self.g = None
        self.static_x = None
        self.static_y = None
    """
    param: baseclas self.weight or self.weight_scale_inv
    loaded_weight: 从 .safetensors 文件加载的完整权重张量
    """

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        """
        self.tp_dim (也就是 dimension)：你要在哪个维度动刀？这里是 0（按行切）。
        start_idx (也就是 start)：这一刀从第几个索引开始下刀?
        shard_size (也就是 length)：切下来多厚（多少行）的一块肉？
        """
        param.data.copy_(loaded_weight.narrow(self.tp_dim, start_idx, shard_size))

    def scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor):
        # Scale 的 loader 逻辑对于纯 Column 切分是一模一样的！
        # 因为 param.data 已经是被除以过 block_size 的浓缩版形状了
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        param.data.copy_(loaded_scale.narrow(self.tp_dim, start_idx, shard_size))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape 
        # 压扁成 2D: [Batch * Seq, Hidden]
        x_2d = x.view(-1, original_shape[-1])
        M = x_2d.shape[0]
        if M > 1:
            x_fp8, x_scale = self._dynamic_quantize_activation_per_token(x_2d)
            # 注意：这里的 triton_fp8_block_gemm 是上一轮写好带有 Split-K 的智能启动器
            y_2d = triton_fp8_block_gemm(x_fp8, self.weight.t(), x_scale, self.weight_scale_inv.t())
            if self.bias is not None:
                y_2d = y_2d + self.bias
            return y_2d.view(*original_shape[:-1], -1)
        else:
            # Decode: 极致务实，Dequant Weight + 官方 BF16 Linear
            weight_bf16 = self._dequantize_weight()
            return F.linear(x, weight_bf16)


        # x_fp8, x_scale = triton_dynamic_quantize(x_2d)
        # return triton_fp8_block_gemm(x_fp8, self.weight.t(), x_scale, self.weight_scale_inv.t(), )
        # # return F.linear(x, self.weight, self.bias).to(toch.bfloat16)

class RowParallelLinear_fp8(LinearBase_fp8):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(self.tp_dim, start_idx, shard_size))

    def scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        param.data.copy_(loaded_scale.narrow(self.tp_dim, start_idx, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dequant_w = self._dequantize_weight()
        # y = F.linear(x, dequant_w, self.bias if self.tp_rank == 0 else None)
        original_shape = x.shape 
        x_2d = x.view(-1, original_shape[-1])
        x_fp8, x_scale = triton_dynamic_quantize(x_2d)
        y_2d = triton_fp8_block_gemm(x_fp8, self.weight.t(), x_scale, self.weight_scale_inv.t(), )
        if self.tp_size > 1:
            dist.all_reduce(y_2d)

        y = y_2d.view(*original_shape[:-1], -1)
        return y

        original_shape = x.shape 
        # 压扁成 2D: [Batch * Seq, Hidden]
        x_2d = x.view(-1, original_shape[-1])
        M = x_2d.shape[0]
        if M > 1:
            x_fp8, x_scale = self._dynamic_quantize_activation_per_token(x_2d)
            # 注意：这里的 triton_fp8_block_gemm 是上一轮写好带有 Split-K 的智能启动器
            y_2d = triton_fp8_block_gemm(x_fp8, self.weight.t(), x_scale, self.weight_scale_inv.t())
            if self.bias is not None:
                y_2d = y_2d + self.bias
            if self.tp_size > 1:
                dist.all_reduce(y_2d)

            y = y_2d.view(*original_shape[:-1], -1)
            return y
        else:
            # Decode: 极致务实，Dequant Weight + 官方 BF16 Linear
            weight_bf16 = self._dequantize_weight()
            y = F.linear(x, weight_bf16)
            if self.bias is not None:
                y = y + self.bias
            if self.tp_size > 1:
                dist.all_reduce(y)
            return y
           


# ==========================================
# 3. Fused Kernels: Merged & QKV
# ==========================================
class MergedColumnParallelLinear_fp8(ColumnParallelLinear_fp8):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor, loaded_shard_id: int):
        # 核心修改：浓缩矩阵的 offset 和 size，必须除以 block_size
        base_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        base_size = self.output_sizes[loaded_shard_id] // self.tp_size
        
        shard_offset = divide(base_offset, self.block_size)
        shard_size = divide(base_size, self.block_size)
        
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_scale = loaded_scale.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_scale)



class QKVParallelLinear_fp8(ColumnParallelLinear_fp8):
    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, total_num_kv_heads: int | None = None, bias: bool = False):
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def _get_qkv_bounds(self, loaded_shard_id: str):
        """提取计算 offset 和 size 的通用逻辑"""
        if loaded_shard_id == "q":
            return 0, self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            return self.num_heads * self.head_size, self.num_kv_heads * self.head_size
        else: # "v"
            return (self.num_heads + self.num_kv_heads) * self.head_size, self.num_kv_heads * self.head_size

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        shard_offset, shard_size = self._get_qkv_bounds(loaded_shard_id)
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor, loaded_shard_id: str):
        base_offset, base_size = self._get_qkv_bounds(loaded_shard_id)
        
        # 核心修改：浓缩矩阵处理
        shard_offset = divide(base_offset, self.block_size)
        shard_size = divide(base_size, self.block_size)
        
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_scale = loaded_scale.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_scale)