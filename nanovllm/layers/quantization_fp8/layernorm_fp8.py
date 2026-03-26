import torch
import torch.nn as nn


@torch.compile
def _rms_norm_fp8_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float):
    orig_dtype = x.dtype
    x_float = x.float()
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(var + eps)
    x_out = x_normed.to(orig_dtype) * weight
    abs_max = torch.max(torch.abs(x_out), dim=-1, keepdim=True).values
    scale = torch.clamp(abs_max, min=1e-12) / 448.0
    x_fp8 = (x_out / scale).to(torch.float8_e4m3fn)
    
    return x_fp8, scale

@torch.compile
def _add_rms_norm_fp8_kernel(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    orig_dtype = x.dtype
    x_float = x.float() + residual.float()
    new_residual = x_float.to(orig_dtype)
    
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(var + eps)
    x_out = x_normed.to(orig_dtype) * weight
    
    abs_max = torch.max(torch.abs(x_out), dim=-1, keepdim=True).values
    scale = torch.clamp(abs_max, min=1e-12) / 448.0
    x_fp8 = (x_out / scale).to(torch.float8_e4m3fn)
    
    return x_fp8, scale, new_residual


class RMSNorm_FP8(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return _rms_norm_fp8_kernel(x, self.weight, self.eps)
        else:
            return _add_rms_norm_fp8_kernel(x, residual, self.weight, self.eps)