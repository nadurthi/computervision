"""Latency and GPU-memory measurement shared by all runners.

    with Timed(device) as t:
        pred = method(...)
    t.seconds, t.peak_mb      # wall time with CUDA synchronisation on both ends; peak torch allocation (None on CPU)

Peak memory is what PyTorch allocated during the call (torch.cuda.max_memory_allocated after a reset). Methods that
run an external CUDA binary (ACMMP, COLMAP) do not allocate through PyTorch, so their peak is reported from
nvidia-smi's used memory sampled right after the call instead, which includes the CUDA context and is an upper bound.
"""
import subprocess
import time


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def _smi_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.split()[0])
    except Exception:
        return None


class Timed:
    def __init__(self, device="cuda", external=False):
        self.cuda = str(device).startswith("cuda")
        self.external = external
        self.seconds = None
        self.peak_mb = None

    def __enter__(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.cuda:
            import torch
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0
        if self.cuda:
            import torch
            self.peak_mb = _smi_used_mb() if self.external else torch.cuda.max_memory_allocated() / 2 ** 20
        return False
