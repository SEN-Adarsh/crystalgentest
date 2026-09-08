# Physics-informed guidance: motif statistics for generated crystals.
"""Corner/edge/face-sharing motif statistics for cation polyhedra.

Discrete counterpart of the connectivity penalty in `polyhedra.py`: two
cations sharing one, two, or three anions form a corner-, edge-, or
face-sharing pair of coordination polyhedra. These counts are the Run-0
metric for the polyhedral-guidance paper (motif distribution before/after
steering), so the bond criteria deliberately match the loss: 3.2 A cutoff
(`physics.DEFAULT_CTOFF`), anions from `redox.ANION_NUMBERS`, cations from
the redox-active transition metals (`redox.REDOX_WINDOW`).

Counting is per cation atom (textbook convention): a cation's neighbours
are the cation images that share at least one anion with it, each
classified by how many anions they share. Enumerated directly in cartesian
space via pymatgen's periodic neighbour lists; no differentiability needed
here, that is the loss's job.

ponytail: unlike `build_physics_graph`, this has no 20-neighbour cap, so it
also serves as ground truth for bond counts near the cap.
"""

from collections import defaultdict
from itertools import combinations
from math import acos, degrees
from typing import Dict, List, Optional

import numpy as np
from pymatgen.core import Structure

from crystalgen.diffusion.physics import DEFAULT_CUTOFF
from crystalgen.diffusion.redox import ANION_NUMBERS, REDOX_WINDOW

MOTIF_NAMES = {1: "corner", 2: "edge", 3: "face"}


def count_sharing_motifs(
    structure: Structure,
    cation_atomic_numbers: Optional[List[int]] = None,
    cutoff: float = DEFAULT_CUTOFF,
) -> Dict[str, float]:
    """Classify cation-cation contacts in one structure by sharing mode.

    Args:
        structure: pymatgen Structure (any cell; periodic images handled).
        cation_atomic_numbers: cations to include. Defaults to the redox-active
            transition metals of `redox.REDOX_WINDOW`.
        cutoff: cation-anion bond cutoff in Angstrom (same value as the loss).

    Returns:
        Dict with per-neighbour counts and fractions ("corner"/"edge"/"face"),
        mean bridge angle per motif in degrees, and mean cation-anion
        coordination number.
    """
    if cation_atomic_numbers is None:
        cation_atomic_numbers = sorted(REDOX_WINDOW.keys())
    cation_set = set(cation_atomic_numbers)
    anion_set = set(ANION_NUMBERS)
    z = np.asarray(structure.atomic_numbers)

    result: Dict[str, float] = {f"{m}_pairs": 0 for m in MOTIF_NAMES.values()}
    result.update(
        {
            "corner_frac": 0.0,
            "edge_frac": 0.0,
            "face_frac": 0.0,
            "total_neighbours": 0,
            "mean_cation_cn": 0.0,
            "num_cation_atoms": int(sum(1 for zz in z if zz in cation_set)),
            "bridge_angle_deg": {},
        }
    )

    anion_sites = [i for i in range(len(structure)) if z[i] in anion_set]
    if not anion_sites:
        return result

    all_nbrs = structure.get_all_neighbors(cutoff, include_index=True)
    cart = structure.cart_coords

    # For every anion, the cation images bonded to it. An event is one anion
    # and two bonded cation images; it contributes a neighbour to each
    # cation's central image, translated so that cation sits at home.
    shared: Dict[tuple, set] = defaultdict(set)
    angles: Dict[tuple, List[float]] = defaultdict(list)

    for a in anion_sites:
        bonded = [
            (nbr.index, tuple(int(round(x)) for x in nbr.image))
            for nbr in all_nbrs[a]
            if z[nbr.index] in cation_set
        ]
        if len(bonded) < 2:
            continue
        for (i, j1), (k, j2) in combinations(bonded, 2):
            vec_i = cart[i] + np.asarray(j1) @ structure.lattice.matrix - cart[a]
            vec_k = cart[k] + np.asarray(j2) @ structure.lattice.matrix - cart[a]
            norm_i = np.linalg.norm(vec_i)
            norm_k = np.linalg.norm(vec_k)
            if norm_i < 1e-6 or norm_k < 1e-6:
                continue
            cos = float(np.dot(vec_i, vec_k) / (norm_i * norm_k))
            angle = degrees(acos(max(-1.0, min(1.0, cos))))

            dk = tuple(x - y for x, y in zip(j2, j1))
            shared[(i, k, dk)].add((a, j1))
            angles[(i, k, dk)].append(angle)
            shared[(k, i, tuple(-x for x in dk))].add((a, j2))
            angles[(k, i, tuple(-x for x in dk))].append(angle)

    cation_nodes = [i for i in range(len(structure)) if z[i] in cation_set]
    instances = {1: 0, 2: 0, 3: 0}
    angle_sum = {1: [], 2: [], 3: []}
    for key, anions in shared.items():
        mode = min(len(anions), 3)
        instances[mode] += 1
        angle_sum[mode].extend(angles[key])

    total = sum(instances.values())
    for mode, name in MOTIF_NAMES.items():
        result[f"{name}_pairs"] = instances[mode]
        result[f"{name}_frac"] = instances[mode] / total if total else 0.0
        result["bridge_angle_deg"][name] = (
            sum(angle_sum[mode]) / len(angle_sum[mode]) if angle_sum[mode] else None
        )
    result["total_neighbours"] = total
    if cation_nodes:
        result["mean_cation_cn"] = sum(
            sum(1 for nbr in all_nbrs[i] if z[nbr.index] in anion_set) for i in cation_nodes
        ) / len(cation_nodes)
    return result


def motif_histogram(structures: List[Structure], **kwargs) -> Dict[str, float]:
    """Aggregate `count_sharing_motifs` over many structures (Run-0 metric)."""
    totals = {"corner": 0, "edge": 0, "face": 0}
    angles = {"corner": [], "edge": [], "face": []}
    n_scored = 0
    for s in structures:
        stats = count_sharing_motifs(s, **kwargs)
        if stats["total_neighbours"] == 0:
            continue
        n_scored += 1
        for name in totals:
            totals[name] += stats[f"{name}_pairs"]
            angle = stats["bridge_angle_deg"][name]
            if angle is not None:
                angles[name].append(angle)
    total = sum(totals.values())
    out = {f"{name}_frac": (totals[name] / total if total else 0.0) for name in totals}
    out["total_neighbours"] = total
    out["structures_scored"] = n_scored
    for name in totals:
        out[f"{name}_bridge_angle_mean"] = (
            sum(angles[name]) / len(angles[name]) if angles[name] else None
        )
    return out
