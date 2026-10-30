"""Where our Triton kernels may run.

On a GPU they run on CUDA tensors. With TRITON_INTERPRET=1 (Triton's own CPU interpreter, used by
our CPU tests) they also run on CPU tensors. Nothing else changes between the two.
"""

import os

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def triton_ok(*tensors):
    """True when every tensor lives where a Triton kernel can read it."""
    for t in tensors:
        if t is None:
            continue
        if not (t.is_cuda or (INTERPRET and t.device.type == "cpu")):
            return False
    return True
