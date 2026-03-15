
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from .kernals_fp8 import triton_dynamic_quantize, triton_fp8_block_gemm_optimized, triton_dequantize_weight



def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# ==========================================
# 1. Base Class
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
        self.output_size = output_size

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

        self.register_buffer(
            "weight_t_packed",
            torch.empty(input_size, output_size, dtype=torch.float8_e4m3fn),
            persistent=False
        )
        self.register_buffer(
            "weight_scale_t_packed",
            torch.empty(scale_input_size, scale_output_size, dtype=torch.float32),
            persistent=False
        )

        # 提前分配内存
        self.max_decode_batch = 256
        # 使用 register_buffer 注册，不参与梯度更新，但会随模型自动分发到对应 GPU
        self.register_buffer(
            "output_fp32_workspace",
            torch.zeros((self.max_decode_batch, self.output_size), dtype=torch.float32),
            persistent=False # persistent=False 表示它不需要被保存到 weights checkpoint 里
        )



    @torch.no_grad()
    def refresh_packed_weight_views(self):
        self.weight_t_packed.copy_(self.weight.data.t().contiguous())
        self.weight_scale_t_packed.copy_(self.weight_scale_inv.data.t().contiguous())



    # expand scale to full weight shape and dequantized by doing element-wise multiplication
    def _dequantize_weight(self) -> torch.Tensor:
        # call kernal.
        return triton_dequantize_weight(self.weight, self.weight_scale_inv, self.block_size)

    # pytorch quantization methods
    def _dynamic_quantize_activation_per_token(self, x: torch.Tensor):
        """
        quant activation to FP8 e4m3, element-wise (row-wise)
        param:
            x has shape of [length, hidden_size] bf16.
            length = batch_size * seq_len
        return:
            x_fp8 same shape as input
            scale in fp32 [length, 1]
        """
        FP8_MAX = 448.0
        amax = x.abs().amax(dim=-1, keepdim=True) 
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

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        """
        self.tp_dim (也就是 dimension)：你要在哪个维度动刀？这里是 0（按行切）。
        start_idx (也就是 start)：这一刀从第几个索引开始下刀?
        shard_size (也就是 length)：切下来多厚（多少行）的一块肉？
        """
        #param.data.copy_(loaded_weight.narrow(self.tp_dim, start_idx, shard_size))
        local_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(local_weight)
        self.weight_t_packed.copy_(local_weight.t().contiguous())
    def scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        #param.data.copy_(loaded_scale.narrow(self.tp_dim, start_idx, shard_size))
        local_scale = loaded_scale.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(local_scale)
        self.weight_scale_t_packed.copy_(local_scale.t().contiguous())
    # use only fp8 gemm kernal
    def forward(self, x_fp8: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:

        original_shape = x_fp8.shape 
        
        x_fp8_2d = x_fp8.view(-1, original_shape[-1])
        x_scale_1d = x_scale.view(-1)

        M = x_fp8_2d.shape[0]
        if M <= self.max_decode_batch:
            out_fp32 = self.output_fp32_workspace[:M, :]
            out_fp32.zero_()
        else:
            out_fp32 = torch.empty(
                (M, self.output_size),
                device=x_fp8.device,
                dtype=torch.float32
            )



        y_2d_fp32 = triton_fp8_block_gemm_optimized(
            x_fp8=x_fp8_2d,
            weight_fp8=self.weight_t_packed,
            x_scale=x_scale_1d,
            weight_scale_inv=self.weight_scale_t_packed,
            out=out_fp32,
            block_size_k=128,
            split_k=None,
            return_bf16=False
        )

        if self.bias is not None:
            y_2d_fp32 = y_2d_fp32 + self.bias

        y_2d_bf16 = y_2d_fp32.to(torch.bfloat16)
        return y_2d_bf16.view(*original_shape[:-1], -1)

class RowParallelLinear_fp8(LinearBase_fp8):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        #param.data.copy_(loaded_weight.narrow(self.tp_dim, start_idx, shard_size))
        local_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(local_weight)
        self.weight_t_packed.copy_(local_weight.t().contiguous())

    def scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        #param.data.copy_(loaded_scale.narrow(self.tp_dim, start_idx, shard_size))
        local_scale = loaded_scale.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(local_scale)
        self.weight_scale_t_packed.copy_(local_scale.t().contiguous())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape 
        x_2d = x.view(-1, original_shape[-1])
        x_fp8, x_scale = triton_dynamic_quantize(x_2d)
        x_fp8_2d = x_fp8.view(-1, original_shape[-1])

        x_scale_1d = x_scale.view(-1)

        M = x_fp8_2d.shape[0]
        if M <= self.max_decode_batch:
            out_fp32 = self.output_fp32_workspace[:M, :]
            out_fp32.zero_()
        else:
            out_fp32 = torch.empty(
                (M, self.output_size),
                device=x_fp8.device,
                dtype=torch.float32
            )
        y_2d_fp32 = triton_fp8_block_gemm_optimized(
            x_fp8=x_fp8_2d,
            weight_fp8=self.weight_t_packed,
            x_scale=x_scale_1d,
            weight_scale_inv=self.weight_scale_t_packed,
            out=out_fp32,
            block_size_k=128,
            split_k=None,
            return_bf16=False
        )

        if self.bias is not None:
            y_2d_fp32 = y_2d_fp32 + self.bias

        y_2d_bf16 = y_2d_fp32.to(torch.bfloat16)
        if self.tp_size > 1:
            dist.all_reduce(y_2d_bf16)
        return y_2d_bf16.view(*original_shape[:-1], -1)


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
        self.refresh_packed_weight_views()
    def scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor, loaded_shard_id: int):
        base_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        base_size = self.output_sizes[loaded_shard_id] // self.tp_size
        
        shard_offset = divide(base_offset, self.block_size)
        shard_size = divide(base_size, self.block_size)
        
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_scale = loaded_scale.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_scale)
        self.refresh_packed_weight_views()


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
        self.refresh_packed_weight_views()
    def scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor, loaded_shard_id: str):
        base_offset, base_size = self._get_qkv_bounds(loaded_shard_id)
    
        shard_offset = divide(base_offset, self.block_size)
        shard_size = divide(base_size, self.block_size)
        
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_scale = loaded_scale.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_scale)      
        self.refresh_packed_weight_views() # ⬅️ 补上这行