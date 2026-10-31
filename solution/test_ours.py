"""Our kernels against plain PyTorch / bitsandbytes, forward and backward.

On a GPU: `python -m pytest solution -q`. Without one, Triton's interpreter runs the same kernels
on the CPU: `TRITON_INTERPRET=1 python -m pytest solution -q`.
"""

import os

import pytest
import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    DEVICE = None

pytestmark = pytest.mark.skipif(DEVICE is None, reason="needs a CUDA GPU or TRITON_INTERPRET=1")


# ---------------------------------------------------------------- RMSNorm

def reference_rmsnorm(x, weight, eps):
    input_dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(input_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(2, 37, 1536), (5, 300)])
def test_rmsnorm_matches_qwen2(dtype, shape):
    from solution.rmsnorm import rms_norm

    torch.manual_seed(0)
    x = (torch.randn(shape, device=DEVICE) * 3).to(dtype).requires_grad_(True)
    weight = torch.rand(shape[-1], device=DEVICE) + 0.5         # fp32, frozen
    grad = torch.randn(shape, device=DEVICE)

    ref = reference_rmsnorm(x, weight, 1e-6)
    ref.backward(grad.to(ref.dtype))
    ref_dx, x.grad = x.grad.clone(), None

    out = rms_norm(x, weight, 1e-6)
    out.backward(grad.to(out.dtype))
    assert out.dtype == ref.dtype and out.shape == ref.shape
    tol = 1e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx.float(), rtol=tol * 10, atol=tol * 10)


# ---------------------------------------------------------------- NF4

def quantize(n_out, n_in, nested):
    import bitsandbytes.functional as F

    torch.manual_seed(1)
    w = torch.randn(n_out, n_in, device=DEVICE, dtype=torch.float16) * 0.02
    packed, state = F.quantize_4bit(w, blocksize=64, quant_type="nf4",
                                    compress_statistics=nested)
    return packed, state


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(256, 1536), (96, 320)])
def test_nf4_dequant_is_bitsandbytes_bit_for_bit(nested, shape):
    import bitsandbytes.functional as F

    from solution.nf4 import dequantize_nf4

    packed, state = quantize(*shape, nested)
    ref = F.dequantize_4bit(packed, state)
    out = dequantize_nf4(packed, state, torch.float16)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.equal(out, ref)


def test_nf4_linear_forward_backward():
    import bitsandbytes.functional as F

    from solution.nf4 import NF4Linear

    packed, state = quantize(192, 256, True)
    weight = F.dequantize_4bit(packed, state)                    # fp16, as in training
    x = torch.randn(3, 7, 256, device=DEVICE, dtype=torch.float16).requires_grad_(True)
    bias = torch.randn(192, device=DEVICE, dtype=torch.float16)
    grad = torch.randn(3, 7, 192, device=DEVICE, dtype=torch.float16)

    ref = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    ref.backward(grad.float())
    ref_dx, x.grad = x.grad.clone(), None

    out = NF4Linear.apply(x, packed, bias, state)
    out.backward(grad)
    assert out.dtype == torch.float16 and x.grad.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad.float(), ref_dx.float(), rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------- fused LM head loss

def reference_head(hidden, weight, labels, attention_mask, num_items):
    """What base Soup computes: fp16 logits (autocast), fp32 loss, TRL entropy, argmax, rewind."""
    logits = (hidden.half() @ weight.half().t()).float()
    shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           shift.reshape(-1), ignore_index=-100,
                                           reduction="none").view(shift.shape)
    loss = ce.sum() / num_items if num_items is not None else ce.sum() / (shift != -100).sum()
    logp = torch.log_softmax(logits, -1)
    entropy = -(logp.exp() * logp).sum(-1) * attention_mask
    return loss, entropy, logits.argmax(-1), ce


@pytest.mark.parametrize("num_items", [None, 17])
def test_fused_head_matches_reference(num_items):
    from solution import fused_loss

    torch.manual_seed(2)
    B, T, H, V = 3, 11, 64, 1000
    weight = torch.randn(V, H, device=DEVICE) * 0.2
    labels = torch.randint(0, V, (B, T), device=DEVICE)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=DEVICE)
    attention_mask[0, 8:] = 0          # right padding
    attention_mask[2, 5:] = 0
    labels[attention_mask == 0] = -100
    labels[:, :3] = -100               # a prompt without loss
    hidden = torch.randn(B, T, H, device=DEVICE, requires_grad=True)
    n = None if num_items is None else torch.tensor(num_items, device=DEVICE)

    ref_loss, ref_ent, ref_pred, ref_ce = reference_head(hidden, weight, labels, attention_mask, n)
    ref_loss.backward()
    ref_grad, hidden.grad = hidden.grad.clone(), None

    old = fused_loss.ROWS_PER_CHUNK
    fused_loss.ROWS_PER_CHUNK = 7      # several chunks
    try:
        loss, ent, pred, ce = fused_loss.fused_head_loss(hidden, weight, labels, attention_mask, n)
    finally:
        fused_loss.ROWS_PER_CHUNK = old
    (loss * 3.0).backward()

    shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    lab = shift != -100
    torch.testing.assert_close(loss, ref_loss, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(ent * attention_mask, ref_ent, rtol=2e-3, atol=2e-3)
    assert torch.equal(pred[lab], ref_pred[lab])
    torch.testing.assert_close(ce, ref_ce, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(hidden.grad, 3.0 * ref_grad, rtol=3e-3, atol=3e-4)


# ---------------------------------------------------------------- pad removal

def test_unpadded_step_matches_padded():
    """A tiny Qwen2 with LoRA dropout: same loss and LoRA gradients with and without unpadding
    (the dropout masks are the same, see unpad.patch_lora_dropout)."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution import unpad

    unpad.patch_unpad()
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa")
    model = Qwen2ForCausalLM(cfg).to(DEVICE)
    try:  # Kaggle's preinstalled torchao is too old for peft's plain-Linear dispatch (not used by the 4-bit run)
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.3,
                                                 target_modules=["q_proj", "v_proj"]))
    except ImportError as exc:
        pytest.skip(f"peft cannot wrap a plain Linear here: {exc}")
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.1)
    model.train()
    ids = torch.randint(0, 97, (3, 12), device=DEVICE)
    mask = torch.ones(3, 12, dtype=torch.long, device=DEVICE)
    mask[0, 7:] = 0
    mask[2, 10:] = 0
    labels = ids.masked_fill(mask == 0, -100)

    def step():
        model.zero_grad()
        torch.manual_seed(5)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        return out.loss.detach(), out.logits.detach(), grads

    loss, logits, grads = step()
    patched = {cls: cls.forward for cls in unpad.ORIGINAL}
    try:
        for cls, fn in unpad.ORIGINAL.items():
            cls.forward = fn
        ref_loss, ref_logits, ref_grads = step()
    finally:
        for cls, fn in patched.items():
            cls.forward = fn
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-5)
    real = mask.bool()
    torch.testing.assert_close(logits[real], ref_logits[real], rtol=1e-4, atol=1e-4)
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-4, atol=1e-5)


def test_fp16_embedding_is_lossless():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.embed16 import shrink_tied_embedding

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg).half().float()      # fp16 checkpoint upcast to fp32, as in Soup
    model.requires_grad_(False)
    ids = torch.randint(0, 97, (2, 9))
    ref = model(input_ids=ids).logits
    assert shrink_tied_embedding(model, devices=("cpu",))
    assert model.get_input_embeddings().weight.dtype == torch.float16
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    out = model(input_ids=ids).logits
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    fresh = Qwen2ForCausalLM(cfg).float()             # not fp16-exact: left alone
    fresh.requires_grad_(False)
    assert not shrink_tied_embedding(fresh, devices=("cpu",))


@pytest.mark.parametrize("delta_dtype", [torch.float16, torch.float32])
def test_add_rmsnorm_matches_add_then_norm(delta_dtype):
    from solution.rmsnorm import add_rms_norm

    torch.manual_seed(3)
    r = (torch.randn(2, 33, 1536, device=DEVICE) * 2).requires_grad_(True)
    a = torch.randn(2, 33, 1536, device=DEVICE, dtype=delta_dtype).requires_grad_(True)
    w = torch.rand(1536, device=DEVICE) + 0.5
    ds = torch.randn(2, 33, 1536, device=DEVICE)
    dy = torch.randn(2, 33, 1536, device=DEVICE)

    s_ref = r + a
    y_ref = reference_rmsnorm(s_ref, w, 1e-6)
    torch.autograd.backward([s_ref, y_ref], [ds, dy])
    dr_ref, da_ref = r.grad.clone(), a.grad.clone()
    r.grad = a.grad = None

    s, y = add_rms_norm(r, a, w, 1e-6)
    torch.autograd.backward([s, y], [ds, dy])
    assert s.dtype == torch.float32 and y.dtype == torch.float32 and a.grad.dtype == delta_dtype
    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(r.grad, dr_ref, rtol=1e-4, atol=1e-4)
    tol = 1e-4 if delta_dtype == torch.float32 else 2e-3
    torch.testing.assert_close(a.grad.float(), da_ref.float(), rtol=tol, atol=tol)


def _linear4bit(n_in, n_out):
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                              quant_type="nf4", compress_statistics=True)
    layer.weight = bnb.nn.Params4bit(torch.randn(n_out, n_in) * 0.05, requires_grad=False,
                                     quant_type="nf4", compress_statistics=True)
    return layer.to(DEVICE)


def test_mlp_fp16_activations_match():
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    from solution.mlp16 import patch_qwen2_mlp_fp16

    torch.manual_seed(4)
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128)
    mlp = modeling_qwen2.Qwen2MLP(cfg)
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = _linear4bit(64, 128), _linear4bit(64, 128), \
        _linear4bit(128, 64)
    x = torch.randn(2, 9, 64, device=DEVICE).requires_grad_(True)
    grad = torch.randn(2, 9, 64, device=DEVICE)

    original = modeling_qwen2.Qwen2MLP.forward
    ref = original(mlp, x)                            # whatever forward is installed now
    ref.backward(grad)
    ref_dx, x.grad = x.grad.clone(), None
    patch_qwen2_mlp_fp16()
    try:
        out = modeling_qwen2.Qwen2MLP.forward(mlp, x)
        out.backward(grad)
    finally:
        modeling_qwen2.Qwen2MLP.forward = original
    assert out.dtype == torch.float16 and ref.dtype == torch.float32
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, ref_dx, rtol=2e-3, atol=2e-3)
