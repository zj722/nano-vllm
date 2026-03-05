# nanovllm/layers/factory.py
from nanovllm.layers import linear as fp16_linear
from nanovllm.layers.quantization_fp8 import Linear_fp8 as fp8_linear

def get_linear_layer(layer_type: str, config):
    """
    根据配置动态返回对应的线性层类。
    
    Args:
        layer_type: "QKV", "MergedColumn", "Row", "Column"
        config: 包含 quantization 属性的 Config 对象
    """
    quant = getattr(config, "quantization", None)
    
    # 建立映射表
    # 键是通用的层名称，值是具体的类（包括来自不同文件的类）
    if quant == "fp8":
        mapping = {
            "QKV": fp8_linear.QKVParallelLinear_fp8,
            "MergedColumn": fp8_linear.MergedColumnParallelLinear_fp8,
            "Row": fp8_linear.RowParallelLinear_fp8,
            "Column": fp8_linear.ColumnParallelLinear_fp8,
        }
    else:
        # 默认 FP16 分支：映射到你 linear.py 中不带后缀的原生类
        mapping = {
            "QKV": fp16_linear.QKVParallelLinear,
            "MergedColumn": fp16_linear.MergedColumnParallelLinear,
            "Row": fp16_linear.RowParallelLinear,
            "Column": fp16_linear.ColumnParallelLinear,
        }
    
    if layer_type not in mapping:
        raise ValueError(f"Unknown layer type: {layer_type}. Should be one of {list(mapping.keys())} ")
        
    return mapping[layer_type]