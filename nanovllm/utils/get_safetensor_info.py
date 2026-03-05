import torch
capability = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name()
print(f"显卡名称: {name}")
print(f"计算能力 (Compute Capability): {capability[0]}.{capability[1]}")