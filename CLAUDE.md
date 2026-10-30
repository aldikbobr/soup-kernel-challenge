# Handoff: Soup Kernel Challenge (Decentrathon 6.0, track 1 "AI Engineering by Soup")

Talk to the user in Russian. The user is not a GPU programmer; they handle accounts, Kaggle and
submission, you do the engineering. Delete this file and ЧТО_СДЕЛАТЬ.md before the final submission.

## Task (full spec: the starter kit README in git history, commit c7c2fcd, and the organizers' doc)
Make the fixed `soup train` QLoRA run (Qwen2.5-1.5B-Instruct NF4, LoRA r=16 alpha=32 dropout 0.05
on q/v, batch 4, max 512 tokens, 64 steps, fp16 on a Kaggle T4) faster and lighter with our own
`@triton.jit` kernels in `solution/`, called via `solution.apply()`. Score: 20 (all 10 checks of
`harness/check_rules.py` pass) + 35 speed (`ln s / ln s_best`) + 30 memory (`ln m / ln m_best`) +
15 README. Our kernels must take >= 5% of training time (CUDA events, steps 3+). Do not edit
`harness/`, `tests/`, `task/`. No unsloth/liger/xformers/flash-attn/torchao etc.

Deadline: **31 Oct 2026, 23:59 Astana time (UTC+5)**. Submit on the platform
(edu.astanahub.com, hackathon page → "Сдать решение"): team GitHub repo (made by the platform),
public Kaggle notebook (saved version, COMMIT = final commit), demo video <= 3 min, presentation.
Fill `templates/SUBMISSION.md` → copy to repo root as `SUBMISSION.md`. Team captain must be 18+.

## State (all committed on `main`; remote `upstream` = organizers' template, never push there)
Done and tested on CPU only (no GPU in the original sandbox):
- `solution/fused_loss.py` + `head_patch.py`: `soup_ce_rows_kernel` — chunked LM head loss,
  entropy, argmax, per-token CE without full logits; gradient computed in forward. TRL metrics
  (entropy, mean_token_accuracy, num_tokens) and Soup rewind (`_soup_chunked_step.token_loss`) kept.
- `solution/nf4.py`: `soup_nf4_dequant_kernel`, bit-exact vs bitsandbytes; `NF4Linear` replaces
  `bitsandbytes.matmul_4bit` for NF4 weights.
- `solution/rmsnorm.py`: RMSNorm fwd/bwd kernels + fused residual-add+RMSNorm (decoder patch).
- `solution/unpad.py`: pad-token removal (packed batch, per-row SDPA, dropout masks identical to base).
- `solution/embed16.py`: tied embedding stored as fp16 (lossless check). `mlp16.py`: gate/up fp16.
- `solution/warmup.py`: compiles kernels on a thread during model load.
- Tests: `TRITON_INTERPRET=1 python -m pytest solution -q` (16 pass on CPU; needs numpy<2.3),
  `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q` (79 pass). On GPU: `python -m pytest solution -q`.
- `README.md`: draft, placeholders `<...>` to fill with Kaggle numbers.

## Next steps
1. Code lives at https://github.com/aldikbobr/soup-kernel-challenge (origin). If the platform gives a team repo, push there too. User-facing steps: ЧТО_СДЕЛАТЬ.md.
2. Kaggle: import `templates/kaggle_notebook.ipynb`, GPU T4 x2, internet on, REPO_URL = team repo,
   Run All (~15 min). Read `runs/result.json`, `runs/checks.json`. Never yet run on a real T4:
   expect first-run fixes. Use `python harness/run_task.py --only solution --out runs_try --profile`
   for quick iterations.
3. Watch: check 4 (kernel share >= 5%), check 5/6 (metrics within 3%), check 9 (nvidia-smi peak >=
   97% of torch reserved), Triton on sm75 (tl.interleave, tl.argmax), compile time.
4. Fill README numbers, SUBMISSION.md, presentation (what/why/how/results/limits), video script.
Ideas left: fused NF4 GEMM in Triton, fp8 MLP activation storage (risky for quality).
