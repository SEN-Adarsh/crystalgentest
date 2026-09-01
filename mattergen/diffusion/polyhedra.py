# Physics-informed guidance: polyhedral geometry and connectivity.
"""Polyhedral awareness for cathode generation.

Two complementary terms, both evaluated on the shared short-range PBC graph
built by `mattergen.diffusion.physics.build_physics_graph`:

`PolyhedralGeometryLoss`
    Regularises the intra-polyhedral anion-cation-anion angles, driving each
    coordination polyhedron towards a regular shape. Octahedral centres are
    pushed towards 90 / 180 degrees via f(cos) = cos^2 * (cos + 1)^2, which
    vanishes at cos = 0 and cos = -1; four-coordinate centres are pushed towards
    the tetrahedral 109.47 degrees via (cos + 1/3)^2. This is a cheap
    continuous-shape-measure proxy: it is zero exactly at the ideal polyhedron
    and grows smoothly with distortion.

`PolyhedralConnectivityLoss`
    Drives the *network* of polyhedra towards open, corner-sharing motifs and
    away from edge- and face-sharing ones, which collapse Li diffusion channels.
    Both terms are read off the bridging anion: for two cations M1, M2 bonded to
    a common anion O, the M-M separation and the M-O-M angle follow from the two
    bond vectors, so no second graph and no discrete connectivity label are
    needed (see the class docstring).

Angles and distances are measured across periodic boundaries because the bond
vectors come from the shared graph, so a polyhedron completed by periodic images
is scored correctly.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from mattergen.diffusion.physics import PhysicsGraph, build_physics_graph
from mattergen.diffusion.redox import ANION_NUMBERS, _weighted_mean

MIN_NEIGHBORS_FOR_ANGLE = 2
TETRAHEDRAL_COORDINATION = 4
# cos(109.47 deg) = -1/3
TETRAHEDRAL_COS = -1.0 / 3.0


def _cation_anion_bonds(graph: PhysicsGraph):
    """The subset of bonds that point from a cation to a coordinating anion.

    Returns (centers, neighbors, vecs) or None if the structure has no such bond.
    `vecs` points from the cation towards the anion.
    """
    z = graph.atomic_numbers
    anion_z = torch.tensor(ANION_NUMBERS, device=z.device)

    center_is_cation = ~torch.isin(z[graph.center], anion_z)
    neighbor_is_anion = torch.isin(z[graph.neighbor], anion_z)
    bond_mask = center_is_cation & neighbor_is_anion

    if not bond_mask.any():
        return None
    return graph.center[bond_mask], graph.neighbor[bond_mask], graph.vec[bond_mask]


class PolyhedralGeometryLoss(nn.Module):
    """Penalises coordination polyhedra that deviate from a regular shape."""

    def forward(self, graph: PhysicsGraph) -> Optional[torch.Tensor]:
        """Returns the scalar geometry loss, or None if no cation has two anion neighbours."""
        bonds = _cation_anion_bonds(graph)
        if bonds is None:
            return None
        centers, _, vecs = bonds

        unit_vecs = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)

        # ponytail: python loop over cations, not over the batch. At a 3.2 A
        # cutoff this is tens of iterations per step; vectorise only if profiling
        # says it matters.
        per_center_losses = []
        per_center_weights = []
        for cation in torch.unique(centers):
            cation_vecs = unit_vecs[centers == cation]
            num_bonds = cation_vecs.size(0)
            if num_bonds < MIN_NEIGHBORS_FOR_ANGLE:
                continue

            cosines = cation_vecs @ cation_vecs.T
            # Exclude the diagonal: a bond with itself has cos = 1, which would
            # otherwise add a constant penalty per bond.
            off_diagonal = ~torch.eye(num_bonds, dtype=torch.bool, device=cosines.device)
            cosines = cosines[off_diagonal]

            if num_bonds == TETRAHEDRAL_COORDINATION:
                # ponytail: coordination number is read off the current, possibly
                # half-denoised geometry, so a 4-coordinate centre is treated as
                # tetrahedral and everything else as octahedral. Square-planar and
                # 5-coordinate centres therefore get the octahedral target, which
                # is the right limit for the cathode motifs we care about.
                penalty = (cosines - TETRAHEDRAL_COS).square()
            else:
                penalty = cosines.square() * (cosines + 1.0).square()

            per_center_losses.append(penalty.mean())
            per_center_weights.append(graph.node_weight[cation])

        if not per_center_losses:
            return None

        return _weighted_mean(torch.stack(per_center_losses), torch.stack(per_center_weights))


# Kept so existing configs and imports referring to the octahedral-only name
# continue to work.
OctahedralGeometryLoss = PolyhedralGeometryLoss


class PolyhedralConnectivityLoss(nn.Module):
    """Favours corner-sharing polyhedra over edge- and face-sharing ones.

    For every pair of cations M1, M2 bonded to a common anion O, both quantities
    of interest follow from the two bond vectors v1 = O - M1 and v2 = O - M2:

        M2 - M1 = v1 - v2                 (cation separation, PBC-correct)
        cos(M1-O-M2) = v1_hat . v2_hat    (bridging angle at the anion)

    Sharing mode does not need to be classified explicitly, because it is already
    encoded in the separation: corner-sharing octahedra sit at d_MM ~ 3.8-4.2 A,
    edge-sharing at ~2.8-3.1 A, face-sharing closer still. A hinge penalty below
    `min_cation_dist` therefore fires on edge- and face-sharing pairs and is
    silent on corner-sharing ones. This is also the only form that is
    differentiable: a discrete shared-vertex count has zero gradient everywhere.

    The second term keeps the surviving corner bridges wide (M-O-M at or above
    `min_bridge_angle_deg`), which is what keeps a diffusion bottleneck open
    rather than merely keeping the two metals apart.
    """

    def __init__(self, min_cation_dist: float = 3.5, min_bridge_angle_deg: float = 130.0):
        super().__init__()
        self.min_cation_dist = min_cation_dist
        # theta >= min_bridge_angle  <=>  cos(theta) <= cos(min_bridge_angle),
        # since cos is decreasing on [0, 180].
        self.max_bridge_cos = math.cos(math.radians(min_bridge_angle_deg))

    def forward(self, graph: PhysicsGraph) -> Optional[torch.Tensor]:
        """Returns the scalar connectivity loss, or None if no anion bridges two cations."""
        bonds = _cation_anion_bonds(graph)
        if bonds is None:
            return None
        centers, neighbors, vecs = bonds

        repulsions = []
        bridge_penalties = []
        weights = []

        # ponytail: python loop over bridging anions, same cost argument as the
        # geometry loss above. The pair enumeration within each anion is
        # vectorised, which is where the combinatorics actually live.
        for anion in torch.unique(neighbors):
            bond_ids = torch.nonzero(neighbors == anion, as_tuple=True)[0]
            if bond_ids.numel() < 2:
                continue

            pairs = torch.combinations(bond_ids, r=2)
            v1 = vecs[pairs[:, 0]]
            v2 = vecs[pairs[:, 1]]

            separation = (v1 - v2).norm(dim=-1)
            # Two bonds from the same cation image to the same anion cannot happen,
            # but a cation bridging to its own periodic image can give a near-zero
            # separation that would blow up the unit vectors below.
            valid = separation > 1e-3
            if not valid.any():
                continue
            pairs, v1, v2, separation = pairs[valid], v1[valid], v2[valid], separation[valid]

            cos_bridge = torch.nn.functional.cosine_similarity(v1, v2, dim=-1, eps=1e-8)

            repulsions.append(torch.relu(self.min_cation_dist - separation).square())
            bridge_penalties.append(torch.relu(cos_bridge - self.max_bridge_cos).square())
            # A pair is only as trustworthy as the noisier of its two cations.
            pair_weight = torch.minimum(
                graph.node_weight[centers[pairs[:, 0]]], graph.node_weight[centers[pairs[:, 1]]]
            )
            weights.append(pair_weight)

        if not repulsions:
            return None

        weight = torch.cat(weights)
        return _weighted_mean(torch.cat(repulsions), weight) + _weighted_mean(
            torch.cat(bridge_penalties), weight
        )


def polyhedral_guidance_grad(
    *,
    frac_coords: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    num_atoms: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    node_weight: torch.Tensor,
    geometry_loss: nn.Module,
    connectivity_loss: nn.Module,
    geometry_weight: float = 1.0,
    connectivity_weight: float = 1.0,
) -> Optional[torch.Tensor]:
    """Gradient of the polyhedral losses with respect to fractional coordinates.

    Used to steer reverse diffusion at sampling time, where no training loss is
    available. Runs under `enable_grad` because sampling is wrapped in
    `torch.no_grad`, and detaches its input so the guidance never leaks into any
    surrounding graph.

    Returns None when the structure yields no scorable polyhedron, in which case
    the caller should leave the score untouched.
    """
    with torch.enable_grad():
        coords = frac_coords.detach().clone().requires_grad_(True)
        graph = build_physics_graph(
            frac_coords=coords,
            cell=cell.detach(),
            atomic_numbers=atomic_numbers,
            num_atoms=num_atoms,
            batch_idx=batch_idx,
            num_graphs=num_graphs,
            node_weight=node_weight,
        )
        if graph is None:
            return None

        total = None
        for weight, loss_fn in ((geometry_weight, geometry_loss), (connectivity_weight, connectivity_loss)):
            if weight == 0.0:
                continue
            loss = loss_fn(graph)
            if loss is None:
                continue
            term = weight * loss
            total = term if total is None else total + term

        if total is None:
            return None

        (grad,) = torch.autograd.grad(total, coords, allow_unused=True)

    if grad is None:
        return None
    return grad.detach().to(frac_coords.dtype)
