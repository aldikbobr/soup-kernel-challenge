"""NF4 weight dequantization in one Triton kernel, for the 4-bit linears of training.

For a batch of 2048 tokens bitsandbytes 0.50 on a T4 runs every 4-bit linear as "dequantize the
weight to fp16, then cuBLAS": in the forward it first dequantizes the 8-bit absmax of the nested
(double) quantization with one kernel, adds the offset with another, then dequantizes the NF4
weight with a third; the backward does the same again for grad_input = grad @ W.

`soup_nf4_dequant_kernel` does all of it in one pass: it reads the packed 4-bit codes, the 8-bit
absmax and the second-level absmax, and writes the fp16 weight. The arithmetic follows
bitsandbytes step by step (fp32 `code2[q] * absmax2`, then `+ offset`, then `nf4[code] * scale`,
then one rounding to fp16) and fp32 contraction (FMA) is off for this kernel, so the weight is the
same, bit for bit, as bitsandbytes' own.

`NF4Linear` is the autograd function around it (forward: dequantize + F.linear; backward:
dequantize + matmul for the input gradient, nothing is saved but the packed weight), and
`patch_bnb_matmul_4bit()` routes `bitsandbytes.matmul_4bit` to it for every NF4 weight on a GPU.
"""

import torch
import triton
import triton.language as tl

from ._device import triton_ok

BYTES_PER_PROGRAM = 1024  # 2048 weights per program


@triton.jit
def soup_nf4_dequant_kernel(packed_ptr, absmax_ptr, absmax2_ptr, code2_ptr, offset_ptr,
                            nf4_ptr, out_ptr, n_bytes, BLOCK: tl.constexpr,
                            HALF_QBLOCK: tl.constexpr, NESTED_BLOCK: tl.constexpr,
                            NESTED: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=valid, other=0).to(tl.int32)
    hi = tl.load(nf4_ptr + (byte >> 4))      # bitsandbytes: high nibble is the even element
    lo = tl.load(nf4_ptr + (byte & 15))
    block = offs // HALF_QBLOCK              # one absmax per QBLOCK weights = QBLOCK / 2 bytes
    if NESTED:
        q = tl.load(absmax_ptr + block, mask=valid, other=0).to(tl.int32)
        scale = tl.load(code2_ptr + q) * tl.load(absmax2_ptr + block // NESTED_BLOCK,
                                                 mask=valid, other=0.0)
        scale = scale + tl.load(offset_ptr)
    else:
        scale = tl.load(absmax_ptr + block, mask=valid, other=0.0)
    out_hi = (hi * scale).to(out_ptr.dtype.element_ty)
    out_lo = (lo * scale).to(out_ptr.dtype.element_ty)
    out = tl.interleave(out_hi, out_lo)
    out_offs = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + out_offs, out, mask=out_offs < 2 * n_bytes)


def supported(quant_state):
    qs = quant_state
    if qs is None or getattr(qs, "quant_type", None) != "nf4" or len(qs.shape) != 2:
        return False
    if qs.blocksize % 2 or qs.code is None or qs.code.numel() != 16:
        return False
    if getattr(qs, "nested", False):
        s2 = qs.state2
        return s2 is not None and s2.blocksize == 256 and s2.code is not None \
            and qs.offset is not None
    return qs.absmax is not None and qs.absmax.dtype == torch.float32


def dequantize_nf4(packed, quant_state, dtype=torch.float16):
    """fp16 (or `dtype`) weight of shape quant_state.shape from the packed NF4 codes."""
    qs = quant_state
    n_out, n_in = qs.shape
    n_bytes = (n_out * n_in) // 2
    out = torch.empty(n_out, n_in, device=packed.device, dtype=dtype)
    flat = packed.reshape(-1)
    code = qs.code.to(device=packed.device, dtype=torch.float32)
    if getattr(qs, "nested", False):
        s2 = qs.state2
        offset = qs.offset
        if not torch.is_tensor(offset):
            offset = torch.tensor(float(offset), device=packed.device)
        offset = offset.to(device=packed.device, dtype=torch.float32).reshape(1)
        args = (qs.absmax, s2.absmax, s2.code, offset)
        nested = True
    else:
        args = (qs.absmax, qs.absmax, code, code)  # the nested slots are not read
        nested = False
    grid = (triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)
    soup_nf4_dequant_kernel[grid](flat, *args, code, out, n_bytes, BLOCK=BYTES_PER_PROGRAM,
                                  HALF_QBLOCK=qs.blocksize // 2, NESTED_BLOCK=256, NESTED=nested,
                                  enable_fp_fusion=False)
    return out


class NF4Linear(torch.autograd.Function):
    """y = x @ W.T + b for an NF4 weight, W dequantized on the fly in forward and backward."""

    @staticmethod
    def forward(ctx, x, packed, bias, quant_state):
        weight = dequantize_nf4(packed, quant_state, x.dtype)
        out = torch.nn.functional.linear(x, weight, bias)
        ctx.quant_state = quant_state
        ctx.save_for_backward(packed)
        ctx.bias_dtype = None if bias is None else bias.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (packed,) = ctx.saved_tensors
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            weight = dequantize_nf4(packed, ctx.quant_state, grad_out.dtype)
            grad_x = torch.matmul(grad_out, weight)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out.reshape(-1, grad_out.shape[-1]).sum(0, dtype=ctx.bias_dtype)
        return grad_x, None, grad_bias, None


_ORIGINAL = {}


def patch_bnb_matmul_4bit():
    """Route bitsandbytes.matmul_4bit to NF4Linear for NF4 weights that Triton can read.

    Linear4bit.forward looks up `bnb.matmul_4bit` on every call, so patching the module attribute
    is enough; anything we do not handle (CPU packing, another quant type, an `out=` buffer)
    takes the original path.
    """
    import bitsandbytes
    import bitsandbytes.autograd._functions as functions

    if "matmul_4bit" in _ORIGINAL:
        return
    original = bitsandbytes.matmul_4bit
    _ORIGINAL["matmul_4bit"] = original

    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        if out is None and supported(quant_state) and triton_ok(A, B) \
                and A.shape[-1] == quant_state.shape[1] and A.numel() > 0 \
                and not getattr(quant_state, "packing_format_for_cpu", False):
            return NF4Linear.apply(A, B, bias, quant_state)
        return original(A, B, quant_state, out=out, bias=bias)

    bitsandbytes.matmul_4bit = matmul_4bit
    functions.matmul_4bit = matmul_4bit
