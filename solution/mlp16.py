"""Keep the MLP's gate and up activations in fp16, as the 4-bit GEMM produced them.

In Soup's QLoRA run the MLP input is the fp32 output of the post-attention RMSNorm, and
bitsandbytes' Linear4bit casts its input to fp16, runs the fp16 GEMM and casts the output back to
the input's dtype. So `gate` and `up`, (tokens, 8960) each, come out as fp32 copies of fp16
numbers, and the SwiGLU keeps both for backward: 72 KB per token per layer, the largest
activation of the model.

Here the MLP casts its input to fp16 first (the cast Linear4bit would make anyway), so gate and up
stay fp16: the same values, half the bytes. SwiGLU computes in fp32 from them and rounds once to
fp16, the cast down_proj would make of the fp32 product, and the MLP output is fp16 (the same
values the base upcasts to fp32 before adding them to the fp32 residual). It runs the template's
SwiGLU example kernel, which does not count as ours.
"""

import torch

from ._device import triton_ok
from .swiglu import swiglu


def _fp16_4bit(linear):
    return getattr(linear, "compute_dtype", None) == torch.float16 \
        and type(linear).__name__ == "Linear4bit" and type(linear).__module__.startswith(
            "bitsandbytes")


def patch_qwen2_mlp_fp16():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_soup_fp16", False):
        return
    previous = cls.forward     # the example's patch (or the original)

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or x.dtype != torch.float32 or not triton_ok(x) \
                or not all(_fp16_4bit(m) for m in (self.gate_proj, self.up_proj, self.down_proj)):
            return previous(self, x)
        x16 = x.to(torch.float16)
        return self.down_proj(swiglu(self.gate_proj(x16), self.up_proj(x16)))

    forward._soup_fp16 = True
    cls.forward = forward
