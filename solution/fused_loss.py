"""Fused LM head loss: loss, entropy, token accuracy and per-row losses without the full logits.

Base Soup computes `lm_head(hidden)` for every position of the padded batch, a (B, T, 151936)
tensor, then upcasts it to fp32 for the loss, keeps the log-softmax for backward, and reads it
again for TRL's entropy, TRL's token accuracy (after a `.contiguous()` copy) and Soup's rewind
row losses. That is several gigabytes alive at once.

Here the logits only ever exist for a chunk of rows at a time, and only for the rows that need
them (real tokens; pad positions contribute nothing to any output):

1. `chunk = hidden[rows] @ W.T` in fp16 on tensor cores (the same fp16 GEMM autocast runs in the
   base), a (C, V) buffer;
2. our Triton kernel `soup_ce_rows_kernel` makes one online pass over each row: max, log-sum-exp,
   entropy, argmax and the target's logit; for rows with a label it makes a second pass that
   overwrites the logits in place with `softmax - onehot` (fp16);
3. `d hidden[rows] = (softmax - onehot) @ W` right away (fp16 GEMM), so backward only scales it.

Outputs are the same quantities the base computes: the mean (or sum / num_items_in_batch) token
cross-entropy, per-token entropy for TRL, the argmax for TRL's accuracy, and per-token
cross-entropy for Soup's rewind log. Idea credit: the chunked "compute the gradient in the
forward" trick of fused linear cross-entropy (Liger-Kernel's FusedLinearCrossEntropy, Wijmans et
al. "Cut Your Losses in Large-Vocabulary Language Models"); the code here is our own.
"""

import torch
import triton
import triton.language as tl

ROWS_PER_CHUNK = 1024   # (1024, 151936) fp16 logits buffer = 311 MB
BLOCK_V = 4096


@triton.jit
def soup_ce_rows_kernel(logits_ptr, stride_row, target_ptr, entropy_ptr, ce_ptr, pred_ptr,
                        V, BLOCK_V: tl.constexpr, LABELLED: tl.constexpr):
    """One program per row of a (C, V) fp16 logits buffer.

    Pass 1 (online softmax): running max m, s = sum exp(x - m), t = sum exp(x - m) * x and the
    argmax. Then lse = m + log s, entropy = lse - t / s, cross-entropy = lse - x[target].
    Pass 2 (labelled rows only): overwrite the row with softmax - onehot(target), in place.
    """
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    cols0 = tl.arange(0, BLOCK_V)

    m = tl.full((), float("-inf"), tl.float32)
    s = tl.zeros((), tl.float32)
    t = tl.zeros((), tl.float32)
    best = tl.full((), float("-inf"), tl.float32)
    best_idx = tl.zeros((), tl.int32)
    for start in range(0, V, BLOCK_V):
        cols = start + cols0
        valid = cols < V
        x = tl.load(base + cols, mask=valid, other=float("-inf")).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        tile_arg = tl.argmax(x, axis=0)
        better = tile_max > best
        best_idx = tl.where(better, start + tile_arg, best_idx)
        best = tl.where(better, tile_max, best)
        m_new = tl.maximum(m, tile_max)
        alpha = tl.exp(m - m_new)
        e = tl.where(valid, tl.exp(x - m_new), 0.0)
        s = s * alpha + tl.sum(e, axis=0)
        t = t * alpha + tl.sum(tl.where(valid, e * x, 0.0), axis=0)
        m = m_new
    lse = m + tl.log(s)
    tl.store(entropy_ptr + row, lse - t / s)

    if LABELLED:
        target = tl.load(target_ptr + row).to(tl.int32)
        x_target = tl.load(base + target).to(tl.float32)
        tl.store(ce_ptr + row, lse - x_target)
        tl.store(pred_ptr + row, best_idx.to(tl.int64))
        for start in range(0, V, BLOCK_V):
            cols = start + cols0
            valid = cols < V
            x = tl.load(base + cols, mask=valid, other=0.0).to(tl.float32)
            g = tl.exp(x - lse) - tl.where(cols == target, 1.0, 0.0)
            tl.store(base + cols, g.to(base.dtype.element_ty), mask=valid)


def _rows_pass(logits, targets, entropy, ce, pred, labelled):
    rows, vocab = logits.shape
    soup_ce_rows_kernel[(rows,)](logits, logits.stride(0), targets, entropy, ce, pred, vocab,
                                 BLOCK_V=BLOCK_V, LABELLED=labelled, num_warps=8)


def head_terms(hidden, weight, shift_labels, real, chunk_rows=None):
    """Everything the base reads from the logits, for the flattened rows of one batch.

    hidden: (N, H) any float dtype; weight: (V, H); shift_labels: (N,) int, -100 where a row has
    no target; real: (N,) bool, rows whose entropy is wanted (attention_mask == 1).
    Returns (ce (N,) fp32, entropy (N,) fp32, pred (N,) int64 (-1 if unlabelled), grad_rows
    (n_labelled, H) fp32 = d(sum of ce) / d hidden[labelled rows], labelled row indices).
    """
    chunk_rows = chunk_rows or ROWS_PER_CHUNK
    n_rows = hidden.shape[0]
    device = hidden.device
    labelled = shift_labels != -100
    lab_idx = labelled.nonzero().squeeze(1)
    unl_idx = (real & ~labelled).nonzero().squeeze(1)
    w16 = weight if weight.dtype == torch.float16 else weight.to(torch.float16)
    entropy = torch.zeros(n_rows, device=device, dtype=torch.float32)
    ce = torch.zeros(n_rows, device=device, dtype=torch.float32)
    pred = torch.full((n_rows,), -1, device=device, dtype=torch.int64)
    grad_rows = torch.empty(lab_idx.numel(), hidden.shape[1], device=device, dtype=torch.float32)

    done = 0
    for idx in lab_idx.split(chunk_rows):
        h = hidden.index_select(0, idx).to(torch.float16)
        logits = h @ w16.t()                                    # (C, V) fp16, tensor cores
        targets = shift_labels.index_select(0, idx).contiguous()
        ent_c = torch.empty(idx.numel(), device=device, dtype=torch.float32)
        ce_c = torch.empty_like(ent_c)
        pred_c = torch.empty(idx.numel(), device=device, dtype=torch.int64)
        _rows_pass(logits, targets, ent_c, ce_c, pred_c, True)  # logits := softmax - onehot
        grad_rows[done:done + idx.numel()] = logits @ w16       # (C, H)
        done += idx.numel()
        entropy.index_copy_(0, idx, ent_c)
        ce.index_copy_(0, idx, ce_c)
        pred.index_copy_(0, idx, pred_c)
        del logits
    for idx in unl_idx.split(chunk_rows):
        h = hidden.index_select(0, idx).to(torch.float16)
        logits = h @ w16.t()
        ent_c = torch.empty(idx.numel(), device=device, dtype=torch.float32)
        _rows_pass(logits, ent_c, ent_c, ent_c, ent_c, False)   # entropy only
        entropy.index_copy_(0, idx, ent_c)
        del logits
    return ce, entropy, pred, grad_rows, lab_idx


class FusedHeadLoss(torch.autograd.Function):
    """loss = sum(ce over labelled rows) / denominator, d loss / d hidden from the forward."""

    @staticmethod
    def forward(ctx, hidden, weight, shift_labels, real, denominator):
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        with torch.no_grad():
            ce, entropy, pred, grad_rows, lab_idx = head_terms(
                flat, weight, shift_labels.reshape(-1), real.reshape(-1))
            if denominator is None:
                denom = torch.tensor(float(lab_idx.numel()), device=hidden.device)
            elif torch.is_tensor(denominator):
                denom = denominator.to(device=hidden.device, dtype=torch.float32)
            else:
                denom = torch.tensor(float(denominator), device=hidden.device)
            loss = ce.sum() / denom   # 0 / 0 = nan, as torch's mean over no targets
        ctx.save_for_backward(grad_rows, lab_idx, denom)
        ctx.shape, ctx.dtype = shape, hidden.dtype
        ctx.mark_non_differentiable(entropy, pred, ce)
        lead = shape[:-1]
        return loss, entropy.view(lead), pred.view(lead), ce.view(lead)

    @staticmethod
    def backward(ctx, grad_loss, *_unused):
        grad_rows, lab_idx, denom = ctx.saved_tensors
        grad = torch.zeros(ctx.shape[:-1].numel(), ctx.shape[-1], device=grad_rows.device,
                           dtype=torch.float32)
        grad.index_copy_(0, lab_idx, grad_rows * (grad_loss.float() / denom))
        return grad.view(ctx.shape).to(ctx.dtype), None, None, None, None


def fused_head_loss(hidden, weight, labels, attention_mask, num_items_in_batch=None):
    """(loss, per-token entropy, argmax, per-token cross-entropy), all (B, T) but the loss.

    Position t is scored against labels[:, t + 1], as transformers' ForCausalLMLoss does.
    """
    shift_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    if attention_mask is None:
        real = torch.ones_like(labels, dtype=torch.bool)
    else:
        real = attention_mask.to(torch.bool)
    return FusedHeadLoss.apply(hidden, weight, shift_labels, real, num_items_in_batch)
