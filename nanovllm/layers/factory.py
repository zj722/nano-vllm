# nanovllm/layers/factory.py
from nanovllm.layers import linear as fp16_linear
from nanovllm.layers import layernorm as fp16_norm
from nanovllm.layers.quantization_fp8 import Linear_fp8 as fp8_linear
from nanovllm.layers.quantization_fp8.layernorm_fp8 import RMSNorm_FP8

# --- 兼容性包装器：让 BF16 线性层接受 scale 但忽略它 ---
def wrap_linear_compat(BaseClass):
    class CompatibleLinear(BaseClass):
        def forward(self, x, scale=None): 
            # 无论传不传 scale，BF16 版都只传 x 给基类
            return super().forward(x)
    return CompatibleLinear

# --- 兼容性包装器：让 BF16 Norm 统一返回格式 ---
def wrap_norm_compat(BaseClass):
    class CompatibleNorm(BaseClass):
        def forward(self, x, residual=None):
            out = super().forward(x, residual)
            # 统一输出：(hidden_states, scale, residual)
            if residual is None:
                # 原版返回 x，包装后返回 (x, None)
                return out, None
            else:
                # 原版返回 (x, res)，包装后返回 (x, None, res)
                return out[0], None, out[1]
    return CompatibleNorm

def get_linear_layer(layer_type: str, config):
    quant = getattr(config, "quantization", None)
    if quant == "fp8":
        mapping = {
            "QKV": fp8_linear.QKVParallelLinear_fp8,
            "MergedColumn": fp8_linear.MergedColumnParallelLinear_fp8,
            "Row": fp8_linear.RowParallelLinear_fp8,
            "Column": fp8_linear.ColumnParallelLinear_fp8,
        }
    else:
        # 包装原有的 BF16 类，不改 linear.py 源码
        mapping = {
            "QKV": wrap_linear_compat(fp16_linear.QKVParallelLinear),
            "MergedColumn": wrap_linear_compat(fp16_linear.MergedColumnParallelLinear),
            "Row": wrap_linear_compat(fp16_linear.RowParallelLinear),
            "Column": wrap_linear_compat(fp16_linear.ColumnParallelLinear),
        }
    return mapping[layer_type]

def get_norm_layer(config):
    quant = getattr(config, "quantization", None)
    if quant == "fp8":
        return RMSNorm_FP8
    else:
        return wrap_norm_compat(fp16_norm.RMSNorm)