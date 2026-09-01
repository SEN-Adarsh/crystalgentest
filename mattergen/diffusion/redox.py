# Physics-informed guidance: differentiable bond-valence / redox loss.
"""Differentiable bond-valence-sum (BVS) loss.

For each redox-active transition metal we compute the Brown-Altermatt bond
valence sum over its first-shell anion bonds and penalise values outside the
oxidation-state window that metal can actually reach in a cathode operating
window. A second term pushes each structure towards overall charge neutrality.

Both terms are differentiable in the bond lengths, so they shape the denoiser's
predicted geometry rather than merely being reported.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter

from mattergen.diffusion.physics import PhysicsGraph

# Brown-Altermatt bond valence parameters r0, in Angstrom, keyed by
# (cation atomic number, anion atomic number). b is taken as 0.37 A throughout.
# ponytail: literature values, good to ~0.02 A. Recalibrate against your own
# reference set if the redox term saturates.
BV_R0: Dict[Tuple[int, int], float] = {
    # -- oxides --
    (22, 8): 1.815,  # Ti-O
    (23, 8): 1.803,  # V-O
    (24, 8): 1.724,  # Cr-O
    (25, 8): 1.790,  # Mn-O
    (26, 8): 1.759,  # Fe-O
    (27, 8): 1.692,  # Co-O
    (28, 8): 1.654,  # Ni-O
    (29, 8): 1.679,  # Cu-O
    (41, 8): 1.911,  # Nb-O
    (42, 8): 1.907,  # Mo-O
    (74, 8): 1.917,  # W-O
    # -- fluorides (oxyfluoride cathodes) --
    (22, 9): 1.723,
    (23, 9): 1.710,
    (24, 9): 1.635,
    (25, 9): 1.698,
    (26, 9): 1.670,
    (27, 9): 1.640,
    (28, 9): 1.599,
}

# Oxidation-state window each redox-active metal can occupy in a cathode.
REDOX_WINDOW: Dict[int, Tuple[float, float]] = {
    22: (3.0, 4.0),  # Ti
    23: (3.0, 5.0),  # V
    24: (3.0, 4.0),  # Cr
    25: (3.0, 4.0),  # Mn
    26: (2.0, 4.0),  # Fe
    27: (3.0, 4.0),  # Co
    28: (2.0, 4.0),  # Ni
    29: (2.0, 3.0),  # Cu
    41: (4.0, 5.0),  # Nb
    42: (4.0, 6.0),  # Mo
    74: (4.0, 6.0),  # W
}

# Nominal charges for ions that do not take part in redox.
FIXED_OXIDATION: Dict[int, float] = {
    3: 1.0,  # Li
    11: 1.0,  # Na
    19: 1.0,  # K
    4: 2.0,  # Be
    12: 2.0,  # Mg
    20: 2.0,  # Ca
    30: 2.0,  # Zn
    38: 2.0,  # Sr
    5: 3.0,  # B
    13: 3.0,  # Al
    57: 3.0,  # La
    6: 4.0,  # C
    14: 4.0,  # Si
    50: 4.0,  # Sn
    15: 5.0,  # P
    52: 4.0,  # Te
    8: -2.0,  # O
    9: -1.0,  # F
    16: -2.0,  # S
    17: -1.0,  # Cl
}

ANION_NUMBERS = (8, 9, 16, 17)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mean of `values` weighted by `weights`, safe when the weights sum to zero."""
    total = weights.sum()
    if total <= 0.0:
        return (values * weights).sum()
    return (values * weights).sum() / total


class DifferentiableRedoxLoss(nn.Module):
    """Bond-valence redox-window penalty plus a charge-neutrality penalty."""

    def __init__(self, b: float = 0.37, neutrality_weight: float = 0.5):
        super().__init__()
        self.b = b
        self.neutrality_weight = neutrality_weight

    def bond_valence_sums(self, graph: PhysicsGraph) -> torch.Tensor:
        """Bond valence sum per node, summed over first-shell cation-anion bonds."""
        z = graph.atomic_numbers
        valences = torch.zeros(z.size(0), device=graph.dist.device, dtype=graph.dist.dtype)

        for (z_cation, z_anion), r0 in BV_R0.items():
            mask = (z[graph.center] == z_cation) & (z[graph.neighbor] == z_anion)
            if not mask.any():
                continue
            s_ij = torch.exp((r0 - graph.dist[mask]) / self.b)
            valences = valences.index_add(0, graph.center[mask], s_ij)

        return valences

    def forward(self, graph: PhysicsGraph) -> Optional[torch.Tensor]:
        """Returns the scalar redox loss, or None if the batch has no redox-active metal."""
        z = graph.atomic_numbers
        device, dtype = graph.dist.device, graph.dist.dtype

        redox_mask = torch.zeros_like(z, dtype=torch.bool)
        lower = torch.zeros(z.size(0), device=device, dtype=dtype)
        upper = torch.zeros(z.size(0), device=device, dtype=dtype)
        for z_metal, (lo, hi) in REDOX_WINDOW.items():
            is_metal = z == z_metal
            if not is_metal.any():
                continue
            redox_mask |= is_metal
            lower = torch.where(is_metal, torch.full_like(lower, lo), lower)
            upper = torch.where(is_metal, torch.full_like(upper, hi), upper)

        if not redox_mask.any():
            return None

        bvs = self.bond_valence_sums(graph)

        # 1. Keep every redox-active metal inside its accessible window.
        metal_bvs = bvs[redox_mask]
        metal_weight = graph.node_weight[redox_mask]
        below = torch.relu(lower[redox_mask] - metal_bvs)
        above = torch.relu(metal_bvs - upper[redox_mask])
        loss_window = _weighted_mean(below.square() + above.square(), metal_weight)

        # 2. Charge neutrality per structure: BVS-derived charge on the redox
        #    metals plus nominal charges elsewhere should cancel.
        charge = torch.where(redox_mask, bvs, torch.zeros_like(bvs))
        for z_ion, ox in FIXED_OXIDATION.items():
            is_ion = (z == z_ion) & ~redox_mask
            if is_ion.any():
                charge = torch.where(is_ion, torch.full_like(charge, ox), charge)

        charge_per_graph = scatter(
            charge, graph.batch_idx, dim=0, dim_size=graph.num_graphs, reduce="sum"
        )
        atoms_per_graph = scatter(
            torch.ones_like(charge), graph.batch_idx, dim=0, dim_size=graph.num_graphs, reduce="sum"
        )
        weight_per_graph = scatter(
            graph.node_weight, graph.batch_idx, dim=0, dim_size=graph.num_graphs, reduce="mean"
        )
        # Normalise per atom so the penalty does not scale with structure size.
        neutrality = (charge_per_graph / atoms_per_graph.clamp(min=1.0)).square()
        loss_neutrality = _weighted_mean(neutrality, weight_per_graph)

        return loss_window + self.neutrality_weight * loss_neutrality
