# nanovllm/utils/loader.py
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# =========================================================
# 通用配置区：以后增加新量化方法（如 INT4/AWQ），只需在这里添加后缀映射
# =========================================================
SUFFIX_MAP = {
    ".weight_scale_inv": "scale_loader",  # FP8 缩放因子
    ".scales": "scale_loader",           # AWQ/GPTQ 缩放因子 (预留)
    ".qzeros": "zero_loader",            # GPTQ 零点 (预留)
    ".weight": "weight_loader",          # 标准权重
    ".bias": "weight_loader",            # 标准偏置
}

def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for tensor_name in f.keys():
                loaded_weight = f.get_tensor(tensor_name)
                
                # 1. [通用识别逻辑] 
                # 根据 SUFFIX_MAP 自动识别后缀类型
                loader_func_name = "weight_loader" # 默认加载器
                attr_suffix = ""
                base_key = tensor_name
                
                for suffix, func_name in SUFFIX_MAP.items():
                    if tensor_name.endswith(suffix):
                        loader_func_name = func_name
                        attr_suffix = suffix
                        base_key = tensor_name.rsplit(suffix, 1)[0]
                        break

                # 2. [路由逻辑] 处理 QKV/GateUp 映射
                mapped = False
                for k_prefix, (v_target, shard_id) in packed_modules_mapping.items():
                    if k_prefix in base_key:
                        param_name = base_key.replace(k_prefix, v_target) + attr_suffix
                        param = model.get_parameter(param_name)
                        
                        # 核心：调用 Parameter 身上绑定的 Loader
                        loader = getattr(param, loader_func_name)
                        loader(param, loaded_weight, shard_id)
                        mapped = True
                        break
                
                # 3. [普通层加载]
                if not mapped:
                    try:
                        param = model.get_parameter(tensor_name)
                        if hasattr(param, loader_func_name):
                            loader = getattr(param, loader_func_name)
                            loader(param, loaded_weight)
                        else:
                            # 兜底逻辑：对于没有特殊 loader 的参数（如 RMSNorm 的 weight）
                            param.data.copy_(loaded_weight)
                    except Exception as e:
                        print(f"Warning: Skip {tensor_name} or Error: {e}")