# test.py
from nanovllm.layers.factory import get_linear_layer

# 定义一个简单的 Mock 类，避开 Config 的复杂初始化
class MockConfig:
    def __init__(self, quant):
        self.quantization = quant

# 1. 测试 FP8 分发
config_fp8 = MockConfig("fp8")
cls_fp8 = get_linear_layer("QKV", config_fp8)
print(f"FP8 Mode -> Class: {cls_fp8.__name__}") 

# 2. 测试 FP16 分发
config_fp16 = MockConfig(None)
cls_fp16 = get_linear_layer("QKV", config_fp16)
print(f"FP16 Mode -> Class: {cls_fp16.__name__}")

if cls_fp8.__name__ == "QKVParallelLinear_fp8" and cls_fp16.__name__ == "QKVParallelLinear":
    print("\033[92m[SUCCESS] Factory logic is 100% correct!\033[0m")