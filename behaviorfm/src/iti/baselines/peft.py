"""Lightweight PEFT adapter classes for the ITI decoder PEFT baseline panel.

Three adapters:
  - LoRALinear: drop-in wrapper around any nn.Linear (including
    NonDynamicallyQuantizableLinear). rank-r LoRA with alpha scaling.
    Base weight is frozen; only A and B are trainable.

No VPT: prepending prompt tokens to the PoseDecoder cross-attention
query sequence would require patching the PoseDecoder forward method,
which risks breaking the apples-to-apples comparison.  VPT is
therefore skipped (see run_peft_baselines.py for the note).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank perturbation.

    forward(x) = base(x) + (B @ A @ x.T).T * (alpha / r)

    The base weight tensor is kept as a plain (non-parameter) buffer so
    that it participates in state_dict serialization but is excluded from
    optimizer updates.  A is initialised from N(0, 1/sqrt(r)); B is
    initialised to zero so the initial output is identical to the frozen base.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        out_features, in_features = base.weight.shape
        self.in_features = in_features
        self.out_features = out_features

        # Store frozen base weight and bias as buffers (not parameters).
        self.register_buffer("weight", base.weight.detach().clone())
        if base.bias is not None:
            self.register_buffer("bias", base.bias.detach().clone())
        else:
            self.bias = None

        # Trainable LoRA matrices.
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)  # same as Linear default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}"
        )


def wrap_decoder_with_lora(decoder: nn.Module, r: int = 4, alpha: float = 8.0) -> int:
    """Replace every nn.Linear in *decoder* with a LoRALinear wrapper in-place.

    Returns the total number of trainable LoRA parameters added.
    Operates on named children recursively, replacing modules by direct
    attribute assignment on their parent — the standard PyTorch pattern for
    in-place module surgery.

    Only `nn.Linear` (and its subclasses such as
    NonDynamicallyQuantizableLinear) are wrapped; other module types are
    left untouched.
    """
    n_lora_params = 0

    def _replace(parent: nn.Module, prefix: str = "") -> None:
        nonlocal n_lora_params
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                lora_mod = LoRALinear(child, r=r, alpha=alpha)
                setattr(parent, child_name, lora_mod)
                n_lora_params += lora_mod.lora_A.numel() + lora_mod.lora_B.numel()
            else:
                _replace(child, full_name)

    _replace(decoder)
    return n_lora_params
