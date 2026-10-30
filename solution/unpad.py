"""Remove the pad tokens inside a batch (the rules allow it; rows are never mixed across batches).

TRL pads every batch of 4 rows on the right to its longest row. The rows average about 220
tokens with a long tail, so a good part of every (4, T) batch is padding, and every linear,
norm, MLP, LoRA and attention of the 28 layers spends time and activation memory on it.

In a training step with a right-padded attention mask, `Qwen2Model.forward` here gathers the
real tokens into one packed (1, N, H) sequence (row after row, in order) and runs every
token-wise op on N rows instead of 4 * T. Attention runs once per row on that row's own slice of
q, k and v, causal, so a token attends to exactly the tokens it attends to in the base (with
right padding and a causal mask a real token never sees a pad). Positions, RoPE and every op are
those of the base; the last hidden state goes back to the padded (4, T, H) layout for the head.

LoRA dropout draws its random mask for the padded (4, T, H) shape and keeps the rows of the real
tokens, so the RNG stream and every dropout mask element are exactly the base's; backward keeps
the mask as bool (1 byte per element), as PyTorch's own dropout does.
"""

import threading
from types import SimpleNamespace

import torch

_PACK = threading.local()
ORIGINAL = {}   # (class, name) -> original method, for the tests


def _current():
    return getattr(_PACK, "info", None)


def _right_padded(mask):
    lengths = mask.sum(dim=1, keepdim=True)
    steps = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    return bool(torch.equal(mask, steps < lengths))


class _PackedDropout(torch.autograd.Function):
    """x * mask / (1 - p) with the given bool mask; saves only the mask."""

    @staticmethod
    def forward(ctx, x, mask, scale):
        ctx.save_for_backward(mask)
        ctx.scale = scale
        return x * (mask.to(x.dtype) * scale)

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        return grad * (mask.to(grad.dtype) * ctx.scale), None, None


def patch_lora_dropout():
    """nn.Dropout on a packed (1, N, H) tensor: same mask elements as on the padded tensor."""
    cls = torch.nn.Dropout
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward
    ORIGINAL[cls] = original

    def forward(self, x):
        info = _current()
        if info is None or not self.training or self.p == 0 or self.inplace or x.dim() != 3 \
                or x.shape[0] != 1 or x.shape[1] != info.n:
            return original(self, x)
        ones = torch.ones(info.batch, info.seq, x.shape[-1], device=x.device, dtype=x.dtype)
        keep = torch.nn.functional.dropout(ones, self.p, True)          # 0 or 1 / (1 - p)
        mask = keep.reshape(-1, x.shape[-1]).index_select(0, info.index) != 0
        return _PackedDropout.apply(x, mask.view(x.shape), 1.0 / (1.0 - self.p))

    forward._soup_patched = True
    cls.forward = forward


def patch_qwen2_attention():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2Attention
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward
    ORIGINAL[cls] = original

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        info = _current()
        if info is None or past_key_values is not None or self.sliding_window is not None \
                or hidden_states.shape[0] != 1 or hidden_states.shape[1] != info.n:
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        hidden_shape = (1, info.n, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = modeling_qwen2.apply_rotary_pos_emb(query, key, cos, sin)
        dropout = self.attention_dropout if self.training else 0.0
        outs = []
        for start, end in info.segments:                                   # one row at a time
            k = modeling_qwen2.repeat_kv(key[:, :, start:end], self.num_key_value_groups)
            v = modeling_qwen2.repeat_kv(value[:, :, start:end], self.num_key_value_groups)
            outs.append(torch.nn.functional.scaled_dot_product_attention(
                query[:, :, start:end], k, v, dropout_p=dropout, is_causal=True,
                scale=self.scaling))
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)        # (1, heads, N, D)
        out = out.transpose(1, 2).reshape(1, info.n, -1).contiguous()
        return self.o_proj(out), None

    forward._soup_patched = True
    cls.forward = forward


def patch_qwen2_model():
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2Model
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward
    ORIGINAL[cls] = original

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        if not (self.training and torch.is_grad_enabled() and input_ids is not None
                and inputs_embeds is None and past_key_values is None and not use_cache
                and position_ids is None and torch.is_tensor(attention_mask)
                and attention_mask.dim() == 2 and attention_mask.shape == input_ids.shape
                and not getattr(self, "has_sliding_layers", False)
                and getattr(self.config, "_attn_implementation", None) == "sdpa"
                and not any(getattr(layer, "gradient_checkpointing", False)
                            for layer in self.layers)    # recomputation would not see the pack
                and _current() is None):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        mask = attention_mask.to(torch.bool)
        if not _right_padded(mask):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        batch, seq = input_ids.shape
        index = mask.reshape(-1).nonzero().squeeze(1)
        n = index.numel()
        lengths = mask.sum(dim=1).tolist()
        segments, start = [], 0
        for length in lengths:
            if length:
                segments.append((start, start + length))
            start += length
        ids = input_ids.reshape(-1).index_select(0, index).unsqueeze(0)            # (1, N)
        positions = torch.arange(seq, device=input_ids.device).expand(batch, seq)
        positions = positions.reshape(-1).index_select(0, index).unsqueeze(0)      # (1, N)
        hidden = self.embed_tokens(ids)
        position_embeddings = self.rotary_emb(hidden, positions)
        _PACK.info = SimpleNamespace(index=index, batch=batch, seq=seq, n=n, segments=segments)
        try:
            for layer in self.layers[: self.config.num_hidden_layers]:
                hidden = layer(hidden, attention_mask=None, position_embeddings=position_embeddings,
                               position_ids=positions, past_key_values=None, use_cache=False,
                               **kwargs)
        finally:
            _PACK.info = None
        hidden = self.norm(hidden)
        padded = hidden.new_zeros(batch * seq, hidden.shape[-1]).index_copy(0, index, hidden[0])
        return BaseModelOutputWithPast(last_hidden_state=padded.view(batch, seq, -1),
                                       past_key_values=None)

    forward._soup_patched = True
    cls.forward = forward


def patch_unpad():
    patch_lora_dropout()
    patch_qwen2_attention()
    patch_qwen2_model()
