import torch

from .base import BaseMetric, is_decoder_block
from .registry import register_metric
from ..tools import logger


def _parse_layer_idx(layer_name):
    parts = layer_name.split(".")
    if len(parts) < 3 or not parts[2].isdigit():
        return None
    return int(parts[2])


@register_metric()
class ActivationEnergyRetention(BaseMetric):
    """
    ActivationEnergyRetention
    --------------------------
    Ratio of L2 energy between adjacent transformer blocks:

        AER(l) = ‖hₗ‖₂² / ‖hₗ₋₁‖₂²

    Indicates how much representational energy survives each layer transition.
    Values close to 1 mean energy is preserved; values >> 1 indicate amplification;
    values << 1 indicate suppression.

    Usage:
        dc = DeepCT(model, metrics=["activation_energy_retention"])
    """

    name = "activation_energy_retention"
    target_layers = is_decoder_block

    def __init__(self):
        super().__init__()
        self.energies = {}

    @torch.inference_mode()
    def update(self, layer_name, hidden_states, **kwargs):
        idx = _parse_layer_idx(layer_name)
        if idx is None:
            logger.warning(
                f"[ActivationEnergyRetention] Cannot parse layer index from "
                f"'{layer_name}', skip."
            )
            return

        h = hidden_states.detach()
        if h.dtype in (torch.bfloat16, torch.float16):
            h = h.to(torch.float32)

        energy = (h ** 2).sum().item()
        self.energies[idx] = (layer_name, energy)

    def compute(self):
        results = {}
        sorted_idx = sorted(self.energies.keys())
        for i, idx in enumerate(sorted_idx):
            name, energy = self.energies[idx]
            if i == 0:
                results[name] = float("nan")
                continue
            prev_idx = sorted_idx[i - 1]
            _, prev_energy = self.energies[prev_idx]
            if prev_energy <= 0:
                results[name] = float("nan")
            else:
                results[name] = energy / prev_energy
        self.values = results
        return self.values
