"""Our solution: Triton kernels that make Soup's QLoRA run on a T4 lighter and faster.

`apply()` is called once by the harness, before Soup builds the model and the trainer. Every
patch is narrow: it only takes over when its inputs are what it was written for (tensors a Triton
kernel can read, a frozen weight, an NF4 weight, a training step with labels) and otherwise
calls the original code.

Our kernels (all in this folder):
- fused_loss.py  `soup_ce_rows_kernel`: LM head loss, entropy, accuracy and rewind row losses
                 chunk by chunk, without the (B, T, 151936) logits;
- nf4.py         `soup_nf4_dequant_kernel`: one-pass NF4 + double-quant dequantization for the
                 4-bit linears, forward and backward;
- rmsnorm.py     `soup_rmsnorm_fwd_kernel`, `soup_rmsnorm_bwd_kernel`: fp32 RMSNorm in one pass;
                 `soup_add_rmsnorm_fwd_kernel`, `soup_add_rmsnorm_bwd_kernel`: the residual add
                 after attention fused into the post-attention RMSNorm.
Two changes without a kernel: unpad.py runs the batch without its pad tokens, and embed16.py
keeps the frozen tied embedding in fp16 (lossless: it was loaded in fp16).
The template's SwiGLU example stays on as well (swiglu.py); it does not count as ours.
"""

from .embed16 import patch_soup_setup
from .head_patch import patch_qwen2_causal_lm, patch_sft_trainer
from .nf4 import patch_bnb_matmul_4bit
from .rmsnorm import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm
from .swiglu import patch_qwen2_mlp
from .unpad import patch_unpad
from .warmup import start as start_warmup


def apply():
    patch_qwen2_mlp()          # the template's example (kept: it saves activation memory)
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    patch_bnb_matmul_4bit()
    patch_qwen2_causal_lm()
    patch_sft_trainer()
    patch_unpad()              # pad tokens out of every token-wise op (no kernel of its own)
    patch_soup_setup()         # frozen tied embedding back to its lossless fp16 (no kernel)
    start_warmup()             # compile the kernels while Soup loads the model
