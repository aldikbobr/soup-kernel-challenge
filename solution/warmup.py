"""Compile our Triton kernels while Soup loads the model, instead of in the first training step.

Every measured run starts with an empty Triton cache, so each kernel variant is compiled (a
second or so of CPU time each) the first time it runs. `start()` compiles them on a worker thread
of this same process right when `apply()` runs, so the compilation overlaps Soup's start-up
(imports, tokenizer, dataset, loading and quantizing the model) instead of stalling step 1. It
uses Triton's own `warmup` (compile only, nothing is launched and nothing is computed) with the
same argument types and specializations the training step passes, so step 1 finds the compiled
kernels in Triton's cache. The time still counts: it is part of the measured run either way.
"""

import threading

import torch

_STARTED = []


def _compile_all():
    from .fused_loss import BLOCK_V, soup_ce_rows_kernel
    from .nf4 import BYTES_PER_PROGRAM, soup_nf4_dequant_kernel
    from .rmsnorm import (_warps, soup_add_rmsnorm_bwd_kernel, soup_add_rmsnorm_fwd_kernel,
                          soup_rmsnorm_bwd_kernel, soup_rmsnorm_fwd_kernel)
    from .swiglu import BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel

    dev = "cuda"
    f16 = torch.empty(64, device=dev, dtype=torch.float16)
    f32 = torch.empty(64, device=dev, dtype=torch.float32)
    i64 = torch.empty(64, device=dev, dtype=torch.int64)
    u8 = torch.empty(64, device=dev, dtype=torch.uint8)
    grid = (1,)
    vocab, hidden = 151936, 1536    # Qwen2.5-1.5B; other sizes simply compile on first use

    soup_ce_rows_kernel.warmup(f16, vocab, i64, f32, f32, i64, vocab, grid=grid,
                               BLOCK_V=BLOCK_V, LABELLED=True, num_warps=8)
    soup_ce_rows_kernel.warmup(f16, vocab, f32, f32, f32, f32, vocab, grid=grid,
                               BLOCK_V=BLOCK_V, LABELLED=False, num_warps=8)
    soup_nf4_dequant_kernel.warmup(u8, u8, f32, f32, f32, f32, f16, 1179648, grid=grid,
                                   BLOCK=BYTES_PER_PROGRAM, HALF_QBLOCK=32, NESTED_BLOCK=256,
                                   NESTED=True, enable_fp_fusion=False)
    block = 2048
    soup_rmsnorm_fwd_kernel.warmup(f32, f32, f32, f32, hidden, hidden, hidden, 1e-6, grid=grid,
                                   BLOCK=block, ROUND_TO_INPUT=False, num_warps=_warps(block))
    soup_rmsnorm_bwd_kernel.warmup(f32, f32, f32, f32, f32, hidden, hidden, hidden, hidden,
                                   grid=grid, BLOCK=block, num_warps=_warps(block))
    soup_add_rmsnorm_fwd_kernel.warmup(f32, f16, f32, f32, f32, f32, hidden, 1e-6, grid=grid,
                                       BLOCK=block, num_warps=_warps(block))
    soup_add_rmsnorm_bwd_kernel.warmup(f32, f32, f32, f32, f32, f32, f16, hidden, grid=grid,
                                       BLOCK=block, num_warps=_warps(block))
    example_swiglu_fwd_kernel.warmup(f16, f16, f16, 8960 * 16, grid=grid, BLOCK=BLOCK)
    example_swiglu_bwd_kernel.warmup(f16, f16, f16, f16, f16, 8960 * 16, grid=grid, BLOCK=BLOCK)
    torch.cuda.synchronize()


def _run():
    try:
        _compile_all()
    except Exception:  # noqa: BLE001 - a failed warm-up only means compiling on first use
        pass


def start():
    """Start compiling on a worker thread (once, and only with a GPU)."""
    if _STARTED or not torch.cuda.is_available():
        return
    thread = threading.Thread(target=_run, name="soup-triton-warmup", daemon=True)
    _STARTED.append(thread)
    thread.start()
