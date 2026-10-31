# Submission: soup-kernel-challenge (aldikbobr)

## Links
- Repository, final commit: https://github.com/aldikbobr/soup-kernel-challenge/commits/main (final commit = `COMMIT` in the saved Kaggle version)
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in the README):
  https://www.kaggle.com/code/aldikdeveloper/notebook9442385db9
- Demo video (up to 3 minutes): <VIDEO_LINK>
- Presentation: https://github.com/aldikbobr/soup-kernel-challenge/blob/main/PRESENTATION.md

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| <NAME> | captain | aldikbobr | aldikdeveloper | <CONTACT> |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `soup_ce_rows_kernel` | `solution/fused_loss.py` | LM head logits → fp32 cross-entropy, TRL entropy, argmax accuracy, rewind row losses | part of the total |
| `soup_nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes' 3-kernel nested-NF4 dequantization | part of the total |
| `soup_rmsnorm_fwd_kernel`, `soup_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2RMSNorm (~7 PyTorch kernels each way) | part of the total |
| `soup_add_rmsnorm_fwd_kernel`, `soup_add_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | residual add + post-attention RMSNorm | part of the total |

Total share of our kernels in the training time (check 4): <SHARE> %.

Ideas taken from other kernels, with links (we wrote the code ourselves): fused linear
cross-entropy — Liger-Kernel `FusedLinearCrossEntropy` (https://github.com/linkedin/Liger-Kernel),
Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*, 2024
(https://arxiv.org/abs/2411.09009); NF4 format — bitsandbytes (https://github.com/bitsandbytes-foundation/bitsandbytes).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 86.38 | 59.73 | speedup 1.45x |
| Peak GPU memory, MiB | 14 345 | 4 485 | base/ours 3.20x |
| Eval loss (public rows) | <BASE_LOSS> | <OURS_LOSS> | <DELTA> % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (logits over the 152k vocabulary, NF4 dequantization,
RMSNorm, padding), how the kernels work, results against the base (speed, memory, quality),
limits and next steps.

## Limitations
- Kernels are written for this run's shapes; other configurations fall back to the original code
  (correct, not faster).
- The MLP and attention matmuls still run in cuBLAS; a fused NF4 GEMM in Triton is the next step.
