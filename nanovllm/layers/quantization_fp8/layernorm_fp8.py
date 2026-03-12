import torch
import torch.nn as nn

class RMSNorm_FP8(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        
        # === 新增：直接在 Norm 内部完成 FP8 动态量化 ===
        # 针对每个 Token (dim=-1) 计算最大值并生成 Scale
        abs_x = torch.abs(x)
        max_val = torch.max(abs_x, dim=-1, keepdim=True).values
        max_val = torch.clamp(max_val, min=1e-12) # 防止除以 0
        scale = max_val / 448.0
        
        # 转换为 FP8 格式
        x_fp8 = (x / scale).to(torch.float8_e4m3fn)
        return x_fp8, scale

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        
        # === 新增：直接在 Norm 内部完成 FP8 动态量化 ===
        abs_x = torch.abs(x)
        max_val = torch.max(abs_x, dim=-1, keepdim=True).values
        max_val = torch.clamp(max_val, min=1e-12)
        scale = max_val / 448.0
        
        x_fp8 = (x / scale).to(torch.float8_e4m3fn)
        return x_fp8, scale, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)