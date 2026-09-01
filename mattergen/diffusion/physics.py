# Physics-informed guidance: shared first-shell periodic graph.
"""First-shell PBC neighbour graph used by the physics-informed auxiliary losses.

The auxiliary losses (bond-valence redox window, octahedral geometry) are
distance-based, so they need a *short-range* graph: the GemNet message-passing
graph uses a 7 A cutoff with 50 neighbours, which pulls in second and third
coordination shells and makes bond-valence sums and O-TM-O angles meaningless.
We therefore build a separate ~3.2 A graph here.

Both the distances and the bond vectors returned are differentiable with respect
to the fractional coordinates handed in, so gradients reach the denoiser.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from mattergen.common.utils.data_utils import get_pbc_distances, radius_graph_pbc

# ponytail: 3.2 A / 20 neighbours covers the first anion shell of every cathode
# chemistry in the training set. Widen only if a target chemistry has genuinely
# longer first-shell bonds.
DEFAULT_CUTOFF = 3.2
DEFAULT_MAX_NEIGHBORS = 20


@dataclass
class PhysicsGraph:
    """A short-range periodic bond graph.

    Attributes:
        atomic_numbers: (num_nodes,) atomic number per node.
        center: (num_edges,) index of the atom each bond is measured *from*.
        neighbor: (num_edges,) index of the atom each bond points *to*.
        dist: (num_edges,) bond length in Angstrom, differentiable.
        vec: (num_edges, 3) center -> neighbor displacement, differentiable.
        batch_idx: (num_nodes,) which structure in the batch each node belongs to.
        num_graphs: number of structures in the batch.
        node_weight: (num_nodes,) how much each atom's physics penalty counts.
            Used to fade guidance out at high diffusion noise, where the denoised
            geometry is not yet meaningful.
    """

    atomic_numbers: torch.Tensor
    center: torch.Tensor
    neighbor: torch.Tensor
    dist: torch.Tensor
    vec: torch.Tensor
    batch_idx: torch.Tensor
    num_graphs: int
    node_weight: torch.Tensor


def build_physics_graph(
    frac_coords: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    num_atoms: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    node_weight: Optional[torch.Tensor] = None,
    cutoff: float = DEFAULT_CUTOFF,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
) -> Optional[PhysicsGraph]:
    """Build the first-shell bond graph for a batch of (possibly denoised) structures.

    Args:
        frac_coords: (num_nodes, 3) fractional coordinates. Gradients flow through these.
        cell: (num_graphs, 3, 3) lattice matrices.
        atomic_numbers: (num_nodes,) atomic numbers.
        num_atoms: (num_graphs,) atoms per structure.
        batch_idx: (num_nodes,) structure index per node.
        num_graphs: number of structures in the batch.
        node_weight: (num_nodes,) optional per-atom weight for the physics
            penalties. Defaults to all ones.
        cutoff: neighbour cutoff in Angstrom.
        max_neighbors: maximum neighbours retained per atom.

    Returns:
        A PhysicsGraph, or None if the batch yields no bonds inside the cutoff.
    """
    # The periodic-graph helpers are hardcoded to float32 internally, and the
    # exponential in the bond-valence sum is unstable in fp16, so run the whole
    # physics path in float32. `.float()` keeps the autograd connection intact.
    frac_coords = frac_coords.float()
    cell = cell.float()

    lattice_nodes = torch.repeat_interleave(cell, num_atoms, dim=0)
    cart_coords = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)

    # Neighbour discovery is a discrete operation; run it detached. The distances
    # below are then recomputed from the live coordinates so gradients survive.
    with torch.no_grad():
        edge_index, to_jimages, num_bonds = radius_graph_pbc(
            cart_coords=cart_coords.detach(),
            lattice=cell.detach(),
            num_atoms=num_atoms,
            radius=cutoff,
            max_num_neighbors_threshold=max_neighbors,
        )

    if edge_index.numel() == 0:
        return None

    out = get_pbc_distances(
        cart_coords,
        edge_index,
        cell,
        to_jimages,
        num_atoms,
        num_bonds,
        coord_is_cart=True,
        return_distance_vec=True,
    )

    # get_pbc_distances computes pos[edge_index[0]] - pos[edge_index[1]], so the
    # vector points from edge_index[1] towards edge_index[0].
    return PhysicsGraph(
        atomic_numbers=atomic_numbers,
        center=edge_index[1],
        neighbor=edge_index[0],
        dist=out["distances"],
        vec=out["distance_vec"],
        batch_idx=batch_idx,
        num_graphs=num_graphs,
        node_weight=(
            torch.ones_like(cart_coords[:, 0]) if node_weight is None else node_weight.float()
        ),
    )
