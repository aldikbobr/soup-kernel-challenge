"""Keep the frozen tied embedding / LM head in fp16, where it came from.

Soup loads Qwen2.5 in fp16 on a T4, then `prepare_model_for_kbit_training` upcasts every
non-quantized parameter to fp32, the tied (151936, 1536) embedding included: 933 MB instead of
467 MB. The upcast values are exactly the fp16 ones, so storing them back in fp16 loses nothing
(we check it, element by element, before doing it). What reads the weight sees the same numbers:

* the embedding lookup returns fp32 as before (a forward hook upcasts the looked-up rows), so the
  fp32 residual stream is unchanged;
* the LM head runs in fp16 under autocast in the base anyway (and our fused head reads fp16), and
  outside autocast it upcasts the weight on the fly, exactly as the fp32 weight would compute.

This is no kernel of ours, only 467 MB of memory and one fp32->fp16 cast of the weight per step.
"""

import torch


def _upcast_output(module, args, output):
    return output.float() if output.dtype != torch.float32 else output


def _linear_any_dtype(self, x):
    weight = self.weight
    if not torch.is_autocast_enabled(x.device.type) and x.dtype != weight.dtype:
        weight = weight.to(x.dtype)
    return torch.nn.functional.linear(x, weight, self.bias)


def shrink_tied_embedding(model, devices=("cuda",)):
    """Store the frozen fp32 input embedding (and the LM head tied to it) as fp16 when that is
    lossless. Returns True when it did."""
    if not isinstance(model, torch.nn.Module):
        return False
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        return False
    try:
        embed, head = get_input(), get_output()
    except Exception:  # noqa: BLE001 - a model without them is not ours to touch
        return False
    if not isinstance(embed, torch.nn.Embedding) or not isinstance(head, torch.nn.Linear):
        return False
    weight = embed.weight
    if head.weight is not weight or weight.requires_grad or weight.dtype != torch.float32 \
            or head.bias is not None or weight.device.type not in devices:
        return False
    with torch.no_grad():
        half = weight.to(torch.float16)
        if not torch.equal(half.to(torch.float32), weight):
            return False
        weight.data = half
    embed.register_forward_hook(_upcast_output)
    head.forward = _linear_any_dtype.__get__(head)
    return True


def patch_soup_setup():
    """Shrink the embedding right after Soup has built the final PEFT model."""
    try:
        from soup_cli.trainer.sft import SFTTrainerWrapper
    except Exception:  # noqa: BLE001
        return
    original = SFTTrainerWrapper._setup_transformers
    if getattr(original, "_soup_patched", False):
        return

    def _setup_transformers(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            shrink_tied_embedding(model)
        return result

    _setup_transformers._soup_patched = True
    SFTTrainerWrapper._setup_transformers = _setup_transformers
