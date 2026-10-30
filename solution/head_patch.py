"""Wire the fused LM head loss into Qwen2 and TRL without changing what Soup logs.

* `Qwen2ForCausalLM.forward`: when our trainer asks for it (a training step with labels), the
  model runs the decoder, then `fused_head_loss` instead of `lm_head` + `loss_function`. It
  returns the loss with `logits=None`, and leaves the per-token entropy, argmax and
  cross-entropy for the trainer.
* `trl.SFTTrainer.compute_loss`: for that step it logs `entropy`, `num_tokens` and
  `mean_token_accuracy` with TRL's own formulas from those per-token values, and leaves the
  per-token cross-entropy where Soup's rewind recorder looks for a step without logits
  (`trainer._soup_chunked_step.token_loss`, the hook Soup itself uses for its chunked loss), so
  `rewind.jsonl` gets the same row losses.

Anything else (evaluation, a custom loss function, label smoothing, Liger, prompt tuning,
context-parallel `shift_labels`, a model that is not Qwen2) takes TRL's original path untouched.
"""

import threading
from types import SimpleNamespace

import torch

from ._device import triton_ok
from .fused_loss import fused_head_loss

STEP_ATTR = "_soup_chunked_step"   # soup_cli.utils.stream_loss_chunk.STEP_ATTR, read by rewind
_STATE = threading.local()


def _wanted():
    return getattr(_STATE, "want", False)


def patch_qwen2_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        weight = self.lm_head.weight
        probe = input_ids if input_ids is not None else inputs_embeds
        if not (_wanted() and labels is not None and self.training and torch.is_grad_enabled()
                and isinstance(logits_to_keep, int) and logits_to_keep == 0
                and "shift_labels" not in kwargs and not weight.requires_grad
                and getattr(self.lm_head, "bias", None) is None and probe is not None
                and triton_ok(probe, weight, labels)):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        loss, entropy, pred, token_ce = fused_head_loss(hidden, weight, labels, attention_mask,
                                                        num_items_in_batch)
        _STATE.step = SimpleNamespace(token_entropy=entropy, pred=pred, token_loss=token_ce)
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    forward._soup_patched = True
    cls.forward = forward


def _log_metrics(trainer, mode, inputs, labels, entropy_tokens, pred):
    """TRL 0.29.1 SFTTrainer.compute_loss's metrics, from per-token values instead of logits."""
    gather = trainer.accelerator.gather_for_metrics
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        entropy = torch.sum(entropy_tokens * attention_mask) / attention_mask.sum()
        entropy = gather(entropy).mean().item()
    trainer._metrics[mode]["entropy"].append(entropy)
    if mode == "train":
        trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    with torch.no_grad():
        shift_labels = labels[..., 1:]
        predictions = pred[..., :-1]
        mask = shift_labels != -100
        correct_tokens = gather(((predictions == shift_labels) & mask).sum())
        total_tokens = gather(mask.sum())
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def _logits_metrics(trainer, mode, inputs, labels, logits):
    """The same metrics from logits, for a model that did not take the fused path."""
    from trl.trainer.utils import entropy_from_logits

    with torch.no_grad():
        entropy_tokens = entropy_from_logits(logits)
        pred = logits.argmax(dim=-1)
    _log_metrics(trainer, mode, inputs, labels, entropy_tokens, pred)


def patch_sft_trainer():
    from transformers import Trainer
    from trl import SFTTrainer

    if getattr(SFTTrainer.compute_loss, "_soup_patched", False):
        return
    original = SFTTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        fused = (self.model.training and "labels" in inputs and "shift_labels" not in inputs
                 and "attention_mask" in inputs
                 and not getattr(self.args, "use_liger_kernel", False)
                 and getattr(self, "compute_loss_func", None) is None
                 and getattr(self, "label_smoother", None) is None
                 and getattr(self, "num_virtual_tokens", 0) == 0
                 and not getattr(self, "aux_loss_enabled", False)
                 and hasattr(self, "_metrics") and hasattr(self, "_total_train_tokens"))
        if not fused:
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train"
        inputs.pop("_prediction_loss_only", None)
        labels = inputs["labels"]
        inputs["use_cache"] = False
        _STATE.want, _STATE.step = True, None
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            step, _STATE.want, _STATE.step = _STATE.step, False, None
        if step is None:
            _logits_metrics(self, mode, inputs, labels, outputs.logits)
        else:
            _log_metrics(self, mode, inputs, labels, step.token_entropy, step.pred)
        setattr(self, STEP_ATTR, step)
        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_patched = True
    SFTTrainer.compute_loss = compute_loss
