import torch
from torch import nn

# 1. 把它变成一个模块级别的纯函数，不带 self
@torch.compile
def _rms_forward_functional(x: torch.Tensor, weight: torch.Tensor, eps: float):
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x

@torch.compile
def _add_rms_forward_functional(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x, residual

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # 2. 类里面的方法去掉 @torch.compile，去调用上面的纯函数
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_forward_functional(x, self.weight, self.eps)

    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _add_rms_forward_functional(x, residual, self.weight, self.eps)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
