import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for tensor_name in f.keys():
                
                # 1. 精准解析后缀和基础名字
                if tensor_name.endswith(".weight_scale_inv"):
                    is_scale = True
                    attr_suffix = ".weight_scale_inv"
                    # base key is something like model.layers.0.self_attn.q_proj
                    base_key = tensor_name[:-17]  # 砍掉 ".weight_scale_inv"
                elif tensor_name.endswith(".weight"):
                    is_scale = False
                    attr_suffix = ".weight"
                    base_key = tensor_name[:-7]   # 砍掉 ".weight"
                else:
                    # 比如 bias, layer_norm 等其他参数，不走 scale_loader 逻辑
                    is_scale = False
                    attr_suffix = ""
                    base_key = tensor_name
                
                # 2. 查表路由逻辑 (处理 QKV 等融合算子)
                mapped = False
                for k_prefix, (v_target, shard_id) in packed_modules_mapping.items():
                    if k_prefix in base_key:
                        # 完美拼装：基础名字替换 + 正确后缀
                        param_name = base_key.replace(k_prefix, v_target) + attr_suffix
                        
                        param = model.get_parameter(param_name)
                        
                        loader_func_name = "scale_loader" if is_scale else "weight_loader"
                        loader = getattr(param, loader_func_name)
                        
                        loader(param, f.get_tensor(tensor_name), shard_id)
                        mapped = True
                        break
                
                # 3. 处理不需要融合的普通 Layer (如 RowParallelLinear)
                if not mapped:
                    # 获取参数 (此时因为我们把属性名改成了 weight_scale_inv，这里绝对能找到了)
                    param = model.get_parameter(tensor_name)
                    
                    loader_func_name = "scale_loader" if is_scale else "weight_loader"
                    
                    if hasattr(param, loader_func_name):
                        loader = getattr(param, loader_func_name)
                        loader(param, f.get_tensor(tensor_name))
                    else:
                        param.data.copy_(f.get_tensor(tensor_name))