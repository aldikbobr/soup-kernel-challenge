"""Qwen2 RMSNorm forward and backward in one Triton kernel each.

After `prepare_model_for_kbit_training` the norms and the residual stream are fp32, and
`Qwen2RMSNorm.forward` runs as seven PyTorch kernels forward (cast, pow, mean, add, rsqrt, mul,
mul) and about as many backward, each a full pass over a (tokens, 1536) fp32 tensor. Here it is
one pass each way, with the same arithmetic in fp32:

    y = w * cast_to_input_dtype(x * rsqrt(mean(x^2) + eps))

Backward (w frozen, as under LoRA): with xhat = x * rstd and g = dy * w,
    dx = rstd * (g - xhat * mean(g * xhat)).
Only x and one fp32 rstd per row are kept for backward, the same as the base keeps.
"""

import torch
import triton
import triton.language as tl

from ._device import triton_ok


@triton.jit
def soup_rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, stride_x, stride_y, n_cols, eps,
                            BLOCK: tl.constexpr, ROUND_TO_INPUT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=valid, other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    xhat = x * rstd
    if ROUND_TO_INPUT:  # the base casts x * rstd back to the input dtype before scaling by w
        xhat = xhat.to(x_ptr.dtype.element_ty).to(tl.float32)
    tl.store(y_ptr + row * stride_y + cols, (w * xhat).to(y_ptr.dtype.element_ty), mask=valid)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def soup_rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, stride_dy, stride_x,
                            stride_dx, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    dy = tl.load(dy_ptr + row * stride_dy + cols, mask=valid, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x + cols, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=valid, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xhat = x * rstd
    g = dy * w
    dot = tl.sum(g * xhat, axis=0) / n_cols
    dx = rstd * (g - xhat * dot)
    tl.store(dx_ptr + row * stride_dx + cols, dx.to(dx_ptr.dtype.element_ty), mask=valid)


def _warps(block):
    return 4 if block <= 1024 else 8 if block <= 4096 else 16


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        rows, cols = x2.shape
        out_dtype = torch.promote_types(weight.dtype, x.dtype)
        y = torch.empty(rows, cols, device=x.device, dtype=out_dtype)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(cols)
        soup_rmsnorm_fwd_kernel[(rows,)](x2, weight, y, rstd, x2.stride(0), y.stride(0), cols,
                                         float(eps), BLOCK=block,
                                         ROUND_TO_INPUT=x.dtype != torch.float32,
                                         num_warps=_warps(block))
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return y.view(*shape[:-1], cols)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        rows, cols = x2.shape
        dy2 = dy.reshape(-1, cols)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty(rows, cols, device=x2.device, dtype=x2.dtype)
        block = triton.next_power_of_2(cols)
        soup_rmsnorm_bwd_kernel[(rows,)](dy2, x2, weight, rstd, dx, dy2.stride(0), x2.stride(0),
                                         dx.stride(0), cols, BLOCK=block, num_warps=_warps(block))
        return dx.view(ctx.shape), None, None


def rms_norm(x, weight, eps):
    return RMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Qwen2RMSNorm.forward through our kernels while its weight is frozen (it is, under LoRA);
    a trainable weight, a too-wide row or a tensor Triton cannot read takes the original path."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        if self.weight.requires_grad or hidden_states.shape[-1] > 16384 \
                or not triton_ok(hidden_states, self.weight) \
                or not hidden_states.dtype.is_floating_point:
            return original(self, hidden_states)
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)

    forward._soup_patched = True
    cls.forward = forward
