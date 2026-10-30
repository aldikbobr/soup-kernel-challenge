# Soup Kernel Challenge — our solution

Decentrathon 6.0, track 1 "AI Engineering by Soup". Own Triton kernels and patches that make the
fixed `soup train` QLoRA run (Qwen2.5-1.5B, NF4, LoRA r=16 on q/v, 64 steps on a Kaggle T4)
**use less GPU memory and run faster, with the same training and the same Soup outputs**.

- Public Kaggle notebook (saved version with results): **<KAGGLE_LINK>**
- Final commit: **<COMMIT_LINK>**

> Кратко по-русски: мы убрали из обучения гигантскую матрицу логитов (152k словарь) за счёт
> слитого Triton-кернела loss/энтропии/точности, распаковываем 4-битные веса своим кернелом в
> один проход, считаем RMSNorm (и сложение с residual) одним кернелом, выбрасываем pad-токены
> из батча и храним то, что и так было fp16, в fp16. Результат — таблица ниже.

## Results (Kaggle T4, `harness/run_task.py`, public rows)

| | Base Soup 0.77.0 | Ours | Ratio |
|---|---|---|---|
| Whole `soup train`, s | <BASE_S> | <OURS_S> | speedup **<S>x** |
| Peak GPU memory (nvidia-smi), MiB | <BASE_MIB> | <OURS_MIB> | base/ours **<M>x** |
| Eval loss on rows 40000–40255 | <BASE_LOSS> | <OURS_LOSS> | <DELTA> % |
| Our kernels' share of training time (check 4) | — | <SHARE> % | ≥ 5 % required |

All 10 checks of `harness/check_rules.py`: <CHECKS>. Full output: `runs/result.json`,
`runs/checks.json` in the notebook above.

## What we did and why

Where base Soup spends memory and time on this task (README of the kit + our reading of the
code): the logits over the 151,936-token vocabulary (several GB alive at once, ~7 % of GPU time
plus many fp32 passes), element-wise ops and dtype casts (~44 %), NF4 dequantization (~5 %), and
every token-wise op also runs on the padding of each batch.

### Our Triton kernels (`solution/`)

| Kernel | File | Replaces |
|---|---|---|
| `soup_ce_rows_kernel` | `fused_loss.py` | `lm_head` → fp32 upcast → cross-entropy → TRL entropy → `.contiguous()` + argmax for accuracy → Soup's rewind row losses |
| `soup_nf4_dequant_kernel` | `nf4.py` | bitsandbytes' 3-kernel nested-NF4 dequantization (absmax dequant, + offset, NF4 dequant), forward and backward of every 4-bit linear |
| `soup_rmsnorm_fwd_kernel`, `soup_rmsnorm_bwd_kernel` | `rmsnorm.py` | Qwen2RMSNorm's ~7 fp32 PyTorch kernels each way |
| `soup_add_rmsnorm_fwd_kernel`, `soup_add_rmsnorm_bwd_kernel` | `rmsnorm.py` | `residual + attn_out` followed by the post-attention RMSNorm |

**1. Fused LM-head loss** (`fused_loss.py`, `head_patch.py`). The logits exist only for a chunk of
rows at a time, and only for real (non-pad) tokens. For each chunk: `logits = h @ Wᵀ` (fp16,
tensor cores — the same GEMM autocast runs in the base); our kernel makes one online pass per row
(running max, Σexp, Σexp·x, argmax, target logit → log-sum-exp, cross-entropy, entropy) and, for
rows with a label, a second pass that overwrites the logits **in place** with `softmax − onehot`;
then `dL/dh = (softmax − onehot) @ W` right away, so backward only rescales it. TRL's `entropy`,
`mean_token_accuracy`, `num_tokens` are logged with TRL's own formulas from these per-token
values; Soup's rewind recorder gets the per-token losses through the hook Soup itself provides for
steps without logits. Removes several GB at the peak.
Idea credits: the "compute the gradient in the forward, chunk the vocabulary GEMM" approach of
fused linear cross-entropy (Liger-Kernel's `FusedLinearCrossEntropy`; Wijmans et al., *Cut Your
Losses in Large-Vocabulary Language Models*, 2024). The code is ours.

**2. NF4 dequantization in one pass** (`nf4.py`). For a 2048-token batch bitsandbytes 0.50 runs
each 4-bit linear as dequantize + cuBLAS. Our kernel reads the packed 4-bit codes, the 8-bit
absmax and the second-level absmax and writes the fp16 weight in one pass; the arithmetic
follows bitsandbytes step by step with FMA contraction off, so **the weight is bit-for-bit
bitsandbytes'** (tested). Nothing but the packed weight is kept for backward.

**3. RMSNorm and residual-add + RMSNorm** (`rmsnorm.py`). fp32 math as in `Qwen2RMSNorm`, one
kernel each way; the add after attention is fused into the post-attention norm (writes the new
fp32 residual and the normed output in one pass; backward adds the residual gradient in the same
kernel).

### Changes without a kernel of ours

- **Pad tokens out of the batch** (`unpad.py`, allowed by the rules). Real tokens of the 4 rows are
  packed into one sequence for every token-wise op; attention runs per row, causal, on that row's
  slice (exactly the tokens a real token attends to in the base). LoRA dropout draws its random
  mask on the padded shape and keeps the real rows, so **every dropout mask element and the RNG
  stream are the base's**.
- **fp16 where the values are fp16 anyway** (`embed16.py`, `mlp16.py`): the frozen tied embedding
  was loaded in fp16 and only upcast by `prepare_model_for_kbit_training` (we check every element
  is exact before storing it back as fp16, −467 MB); the MLP's `gate`/`up` are outputs of an fp16
  GEMM that bitsandbytes upcasts to fp32 — we keep them fp16 (half of the largest activation).
  The SwiGLU itself is the template's example kernel (kept on; it does not count as ours).
- **Compile while loading** (`warmup.py`): Triton compiles our kernels on a worker thread of the
  same process while Soup loads the model, instead of stalling step 1 (the time still counts).

## How to run

On Kaggle (what the jury runs): import `templates/kaggle_notebook.ipynb`, accelerator **GPU T4
x2**, internet on, put this repository in `REPO_URL`, **Run All** (~15 min). Or open our public
notebook above and press Run All.

Locally on Linux with a CUDA GPU:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q                    # our kernels vs PyTorch / bitsandbytes
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q # Soup's 79 regression tests with our patches
python harness/run_task.py && python harness/check_rules.py && python harness/score.py runs/result.json
```

Without a GPU the kernel tests run in Triton's CPU interpreter:
`TRITON_INTERPRET=1 python -m pytest solution -q` (needs `numpy<2.3` for the interpreter).

## Correctness

- `solution/test_ours.py`: every kernel forward and backward against PyTorch; NF4 bit-exact
  against bitsandbytes; the fused head against fp16 logits + fp32 cross-entropy; the unpadded
  model step against the padded one (same loss, same LoRA gradients, with dropout on); the fp16
  embedding and fp16 MLP paths against the originals.
- Soup's 79 regression tests pass with `apply()`.
- End-to-end on a tiny Qwen2 through Soup's `RewindSFTTrainer`: identical rewind rows and token
  counts, `num_tokens` identical, loss / entropy within 0.05 %.

## Limits and what did not work

<LIMITS>

- The kernels are written for this run's shapes; anything else (another model, a trainable norm or
  embedding, left padding, gradient checkpointing, Liger, packing) falls back to the original code
  path, so it is correct but not faster.
- The MLP and attention matmuls (the largest share of GPU time) still run in cuBLAS; a fused NF4
  GEMM in Triton is the next step.

## Files

| Path | What |
|---|---|
| `solution/__init__.py` | `apply()`: installs every patch |
| `solution/fused_loss.py`, `solution/head_patch.py` | fused LM-head loss and its wiring into Qwen2 / TRL / rewind |
| `solution/nf4.py` | NF4 dequant kernel and the 4-bit linear autograd function |
| `solution/rmsnorm.py` | RMSNorm and add+RMSNorm kernels |
| `solution/unpad.py` | pad-token removal |
| `solution/embed16.py`, `solution/mlp16.py` | lossless fp16 storage |
| `solution/warmup.py` | kernel compilation during model loading |
| `solution/test_ours.py` | tests |
| `solution/swiglu.py` | the template's example (unchanged) |

The task's own instructions are in the starter kit:
https://github.com/MakazhanAlpamys/soup-kernel-challenge

## License

Apache-2.0, see `LICENSE` and `NOTICE` (starter kit and Soup's tests, Apache-2.0).
