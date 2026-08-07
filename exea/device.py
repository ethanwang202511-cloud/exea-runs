"""ROCm-safe device selection and precision helpers.

On AMD MI300X the ROCm build of PyTorch still exposes the accelerator through
the *cuda* API (``torch.cuda.is_available()`` is True, tensors go to
``"cuda"``). So the correct device string on both NVIDIA and AMD is ``"cuda"``;
we never hardcode a vendor. Everything degrades to CPU when no accelerator is
present (e.g. the local smoke test on a Mac).

Rules enforced here:
  * Pick device once, at runtime, honouring the EXEA_DEVICE override.
  * Prefer bf16 autocast on the accelerator (MI300X has excellent bf16); never
    silently use fp16 which can overflow. Autocast is OFF on CPU.
  * Expose ``is_rocm()`` for logging/diagnostics only — code paths must not
    branch on vendor; they branch on *capability*.
"""
from __future__ import annotations

import os
import contextlib

import torch


def _detect_device() -> str:
    override = os.environ.get("EXEA_DEVICE", "").strip().lower()
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    # Fall back to CPU. Apple MPS is intentionally NOT auto-selected (it lacks
    # some ops and the server has CUDA/ROCm anyway); request it explicitly with
    # EXEA_DEVICE=mps for a local sanity check if ever needed.
    return "cpu"


_DEVICE: str | None = None


def get_device() -> torch.device:
    """Return the process-wide torch.device, detected once."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = _detect_device()
    return torch.device(_DEVICE)


def is_accelerator() -> bool:
    return get_device().type == "cuda"


def is_rocm() -> bool:
    """True if the CUDA runtime is actually AMD ROCm/HIP. Diagnostics only."""
    if not torch.cuda.is_available():
        return False
    ver = getattr(torch, "version", None)
    return bool(getattr(ver, "hip", None))


def amp_dtype() -> torch.dtype:
    """Preferred autocast dtype on the accelerator.

    bf16 on any accelerator that supports it (MI300X does), else fp16, else
    fp32. bf16 is chosen over fp16 to avoid overflow in unattended training.
    """
    if not is_accelerator():
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


@contextlib.contextmanager
def autocast():
    """Autocast context that is a no-op on CPU and bf16/fp16 on accelerator."""
    if is_accelerator():
        with torch.autocast(device_type="cuda", dtype=amp_dtype()):
            yield
    else:
        yield


def device_report() -> dict:
    """Human-readable device facts for the run log / manifest."""
    dev = get_device()
    rep = {
        "device": str(dev),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "is_rocm": is_rocm(),
        "amp_dtype": str(amp_dtype()),
    }
    if dev.type == "cuda":
        try:
            rep["gpu_name"] = torch.cuda.get_device_name(0)
            rep["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            rep["gpu_mem_gb"] = round(props.total_memory / 1e9, 1)
        except Exception as e:  # never let diagnostics crash the run
            rep["gpu_query_error"] = repr(e)
    return rep
