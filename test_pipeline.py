"""Self-checks for the three cathode-pipeline features.

Feature 1: hierarchical generation - deterministic Li placement into a host framework.
Feature 2: physics-informed guidance - redox / octahedral losses on denoised geometry.
Feature 3: polyhedral awareness - corner-sharing connectivity, tetrahedral shape targets,
           and reverse-diffusion steering.

Run directly:  python test_pipeline.py
"""

import json
import os

import torch
from pymatgen.core import Lattice, Structure

from mattergen.diffusion.physics import build_physics_graph
from mattergen.diffusion.polyhedra import (
    PolyhedralConnectivityLoss,
    PolyhedralGeometryLoss,
    polyhedral_guidance_grad,
)
from mattergen.diffusion.redox import DifferentiableRedoxLoss
from mattergen.li_placer import ANIONS, PhysicsInformedLiPlacer

OctahedralGeometryLoss = PolyhedralGeometryLoss

MANIFEST_PATH = os.path.join("data", "delithiated_manifest.json")
HOSTS_DIR = os.path.join("data", "delithiated_hosts")


def _octahedral_nio(distortion: float = 0.0) -> tuple:
    """Rock-salt-like NiO3 cell: one Ni octahedrally coordinated across the PBC.

    Args:
        distortion: fractional-coordinate offset applied to one anion to break
            the ideal 90/180 degree angles.

    Returns:
        (frac_coords, cell, atomic_numbers, num_atoms, batch_idx)
    """
    frac = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Ni
            [0.5 + distortion, distortion, 0.0],  # O
            [0.0, 0.5, 0.0],  # O
            [0.0, 0.0, 0.5],  # O
        ],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 4.2
    atomic_numbers = torch.tensor([28, 8, 8, 8], dtype=torch.long)
    num_atoms = torch.tensor([4], dtype=torch.long)
    batch_idx = torch.zeros(4, dtype=torch.long)
    return frac, cell, atomic_numbers, num_atoms, batch_idx


def _graph(distortion: float = 0.0, requires_grad: bool = False, node_weight=None):
    frac, cell, z, num_atoms, batch_idx = _octahedral_nio(distortion)
    if requires_grad:
        frac = frac.clone().requires_grad_(True)
    graph = build_physics_graph(
        frac_coords=frac,
        cell=cell,
        atomic_numbers=z,
        num_atoms=num_atoms,
        batch_idx=batch_idx,
        num_graphs=1,
        node_weight=node_weight,
    )
    return frac, graph


def check_physics_graph():
    """A perfect octahedron must produce six first-shell Ni-O bonds at a/2."""
    _, graph = _graph()
    assert graph is not None, "no bonds found inside the physics cutoff"

    is_ni_center = graph.atomic_numbers[graph.center] == 28
    ni_bonds = graph.dist[is_ni_center]
    assert ni_bonds.numel() == 6, f"expected 6 Ni-O bonds, got {ni_bonds.numel()}"
    assert torch.allclose(
        ni_bonds, torch.full_like(ni_bonds, 2.1), atol=1e-4
    ), f"Ni-O bond lengths wrong: {ni_bonds.tolist()}"
    print(f"  physics graph: 6 Ni-O bonds at {ni_bonds[0]:.3f} A")


def check_octahedral_loss():
    """The geometry loss must vanish for an ideal octahedron and grow when distorted."""
    loss_fn = OctahedralGeometryLoss()

    _, ideal_graph = _graph(distortion=0.0)
    ideal = loss_fn(ideal_graph)
    assert ideal is not None, "loss returned None for a coordinated cation"
    assert ideal.item() < 1e-6, f"ideal octahedron should score ~0, got {ideal.item():.6f}"

    _, bent_graph = _graph(distortion=0.08)
    bent = loss_fn(bent_graph)
    assert bent.item() > ideal.item(), "distorted cell must be penalised more than the ideal one"
    print(f"  octahedral loss: ideal={ideal.item():.2e}  distorted={bent.item():.4f}")


def check_redox_loss():
    """Bond valence sums must be finite, and the loss must reject a stretched cell."""
    loss_fn = DifferentiableRedoxLoss()

    _, graph = _graph()
    bvs = loss_fn.bond_valence_sums(graph)
    ni_bvs = bvs[graph.atomic_numbers == 28].item()
    assert 1.0 < ni_bvs < 3.0, f"Ni bond valence sum out of physical range: {ni_bvs:.3f}"

    loss = loss_fn(graph)
    assert loss is not None, "loss returned None despite a redox-active metal"
    assert torch.isfinite(loss), "redox loss is not finite"
    print(f"  redox loss: Ni BVS={ni_bvs:.3f}  loss={loss.item():.4f}")


def check_no_redox_metal_is_skipped():
    """A cell with no redox-active metal yields no redox loss rather than a bogus one."""
    frac = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64)
    graph = build_physics_graph(
        frac_coords=frac,
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 3.0,
        atomic_numbers=torch.tensor([3, 8], dtype=torch.long),  # Li, O
        num_atoms=torch.tensor([2], dtype=torch.long),
        batch_idx=torch.zeros(2, dtype=torch.long),
        num_graphs=1,
    )
    assert graph is not None
    assert DifferentiableRedoxLoss()(graph) is None, "Li-O cell must not produce a redox loss"
    print("  redox loss correctly skipped for a cell with no redox-active metal")


def check_noise_fade():
    """Guidance must vanish where the per-atom weight is zero (high diffusion noise)."""
    for name, loss_fn in (
        ("redox", DifferentiableRedoxLoss()),
        ("octahedral", OctahedralGeometryLoss()),
    ):
        _, full = _graph(distortion=0.08, node_weight=torch.ones(4))
        _, faded = _graph(distortion=0.08, node_weight=torch.zeros(4))
        assert loss_fn(full).item() > 0.0, f"{name} loss should be non-zero at full weight"
        assert loss_fn(faded).item() == 0.0, f"{name} loss should vanish at zero weight"
    print("  both losses vanish at zero node weight and are active at full weight")


def check_gradients_flow():
    """The whole point of the fix: both losses must reach the coordinates."""
    for name, loss_fn in (
        ("redox", DifferentiableRedoxLoss()),
        ("octahedral", OctahedralGeometryLoss()),
    ):
        frac, graph = _graph(distortion=0.05, requires_grad=True)
        loss = loss_fn(graph)
        assert loss is not None, f"{name} loss returned None"
        loss.backward()
        assert frac.grad is not None, f"{name} loss produced no gradient"
        grad_norm = frac.grad.norm().item()
        assert grad_norm > 0.0, f"{name} loss gradient is identically zero"
        print(f"  {name} loss gradient norm w.r.t. coordinates: {grad_norm:.4f}")


def check_redox_capacity():
    """Li capacity must follow charge balance, and non-cathodes must be rejected."""
    placer = PhysicsInformedLiPlacer()

    # CoO2 host: Co goes 4+ -> 3+, so exactly one Li per Co.
    coo2 = Structure(
        Lattice.cubic(4.0),
        ["Co", "O", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
    )
    capacity = placer.calculate_redox_capacity(coo2)
    assert capacity == 1, f"CoO2 should accept 1 Li, got {capacity}"

    # No redox-active metal: not a cathode host.
    mgo = Structure(Lattice.cubic(4.0), ["Mg", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    assert placer.calculate_redox_capacity(mgo) == 0, "MgO must be rejected"
    assert placer.place_lithium(mgo) is None, "MgO must not be lithiated"
    print(f"  redox capacity: CoO2 -> {capacity} Li,  MgO -> rejected")


def check_unknown_element_rejected():
    """An element with no tabulated oxidation state must reject the host, not default to +2."""
    placer = PhysicsInformedLiPlacer()

    # Ta is in neither REDOX_WINDOW nor FIXED_OXIDATION. Booking it as +2 instead
    # of its real +5 inflated the capacity of hosts like TaNbS6.
    tas = Structure(
        Lattice.cubic(5.0),
        ["Ta", "Nb", "S", "S", "S", "S"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.25],
            [0.75, 0.25, 0.75],
        ],
    )
    assert placer.calculate_redox_capacity(tas) == 0, "host with untabulated Ta must be rejected"
    assert placer.place_lithium(tas) is None, "host with untabulated Ta must not be lithiated"
    print("  host containing untabulated Ta correctly rejected")


def check_anion_dimer_detection():
    """A bonded anion pair carries 2 fewer electrons than two isolated anions."""
    placer = PhysicsInformedLiPlacer()

    # Two O at 1.45 A: one peroxide unit, not two oxide ions.
    lat = Lattice.cubic(8.0)
    peroxide = Structure(
        lat, ["Ni", "O", "O"], [[0.0, 0.0, 0.0], [0.4, 0.5, 0.5], [0.4 + 1.45 / 8.0, 0.5, 0.5]]
    )
    assert placer.count_anion_dimers(peroxide) == 1, "O-O at 1.45 A must count as one dimer"

    # Same cell, oxygens pulled apart: two independent oxide ions.
    oxide = Structure(lat, ["Ni", "O", "O"], [[0.0, 0.0, 0.0], [0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
    assert placer.count_anion_dimers(oxide) == 0, "O at 3.2 A apart must not count as a dimer"

    # The dimer removes 2 from the anion charge the metals must balance, so the
    # peroxide cell accepts fewer Li than the oxide one.
    assert placer.calculate_redox_capacity(peroxide) < placer.calculate_redox_capacity(oxide), (
        "peroxide must have lower capacity than the same cell with isolated oxide ions"
    )

    # A linear S3 chain is one dimer plus a lone S, never two dimers.
    s3 = Structure(
        lat,
        ["V", "S", "S", "S"],
        [[0.0, 0.0, 0.0], [0.4, 0.5, 0.5], [0.65, 0.5, 0.5], [0.9, 0.5, 0.5]],
    )
    assert placer.count_anion_dimers(s3) == 1, "each anion may be matched at most once"
    print(
        f"  dimers: peroxide=1 (capacity {placer.calculate_redox_capacity(peroxide)}), "
        f"oxide=0 (capacity {placer.calculate_redox_capacity(oxide)}), S3 chain=1"
    )


def check_overoxidized_host_rejected():
    """A host no oxidation state can balance must be rejected, not clamped to full capacity."""
    placer = PhysicsInformedLiPlacer()
    lat = Lattice.cubic(9.0)

    # MnO4 came out of a real sampling run. Neutrality needs Mn(8+); the window
    # tops out at 4+. Before the feasibility guard this reported 1 Li, because
    # capacity was clamped to the electrons Mn can accept and never checked
    # against the charge the host actually demands.
    mno4 = Structure(
        lat,
        ["Mn", "O", "O", "O", "O"],
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2], [0.8, 0.0, 0.0]],
    )
    assert placer.calculate_redox_capacity(mno4) == 0, "MnO4 needs Mn(8+) and must be rejected"
    assert placer.place_lithium(mno4) is None, "MnO4 must not be lithiated"

    # MnO2 is the same chemistry at a charge Mn(4+) can carry, and must survive.
    mno2 = Structure(lat, ["Mn", "O", "O"], [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.8, 0.0, 0.0]])
    assert placer.calculate_redox_capacity(mno2) > 0, "MnO2 is balanceable and must be accepted"

    print("  MnO4 (needs Mn 8+) rejected, MnO2 accepted")


def check_li_cation_separation():
    """The steric knobs must hold physical values, and a closed host must be rejected."""
    placer = PhysicsInformedLiPlacer()
    assert placer.min_tm_dist >= 2.4, "Li-cation floor is a Li-anion number, too permissive"
    assert placer.min_coordination >= 4, "Li needs at least a tetrahedron of anions"

    # Charge-balanceable (Mn2O4 wants 2 Li) but far too sparse for a real Li site:
    # with only 4 O in 512 A^3 every Voronoi void is 2-3 coordinate. The placer must
    # say so rather than relax its cutoffs until something fits.
    sparse = Structure(
        Lattice.cubic(8.0),
        ["Mn", "Mn", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
            [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.75, 0.0, 0.0], [0.0, 0.75, 0.0],
        ],
    )
    assert placer.calculate_redox_capacity(sparse) > 0, "Mn2O4 is charge-balanceable"
    assert placer.place_lithium(sparse) is None, "host with only low-CN voids must be rejected"

    print("  cation floor 2.40 A, min CN 4, undercoordinated host rejected")


def check_placement_on_real_host():
    """End-to-end Li insertion on a host from the dataset, if it is available."""
    if not os.path.exists(MANIFEST_PATH):
        print(f"  skipped: {MANIFEST_PATH} not found")
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    placer = PhysicsInformedLiPlacer()
    for record in manifest:
        cif_path = record["cif_path"]
        if not os.path.exists(cif_path):
            cif_path = os.path.join(HOSTS_DIR, os.path.basename(cif_path))
        if not os.path.exists(cif_path):
            continue

        host = Structure.from_file(cif_path)
        lithiated = placer.place_lithium(host)
        if lithiated is None:
            continue

        n_li = sum(1 for s in lithiated if s.specie.symbol == "Li")
        assert n_li > 0
        assert len(lithiated) == len(host) + n_li
        # Every inserted Li must respect the steric floors and be properly coordinated.
        for site in lithiated[len(host):]:
            dists = [
                (lithiated.lattice.get_distance_and_image(site.frac_coords, o.frac_coords)[0], o)
                for o in host
            ]
            nearest = min(d for d, _ in dists)
            assert nearest >= placer.min_anion_dist - 1e-6, f"Li too close to framework: {nearest:.2f} A"

            cations = [d for d, o in dists if o.specie.symbol not in ANIONS]
            assert min(cations) >= placer.min_tm_dist - 1e-6, f"Li only {min(cations):.2f} A from a cation"

            cn = sum(
                1 for d, o in dists
                if o.specie.symbol in ANIONS and placer.min_anion_dist <= d <= placer.max_anion_dist
            )
            assert cn >= placer.min_coordination, f"Li in a CN {cn} pocket"
        print(f"  {record['host_formula']} -> {lithiated.composition.reduced_formula} ({n_li} Li)")
        return

    print("  skipped: no host CIF from the manifest was lithiated")


def check_cycling_math():
    """Voltage sign/scale and volume change must come out right on known numbers."""
    from mattergen.cycling_screen import average_voltage, min_li_clearance, volume_change_pct

    # Host at -10 eV, one Li metal atom at -1.9 eV; lithiating releases 3.5 eV.
    voltage = average_voltage(
        energy_lithiated=-10.0 - 1.9 - 3.5, energy_host=-10.0, num_li=1, energy_li_metal=-1.9
    )
    assert abs(voltage - 3.5) < 1e-9, f"voltage should be 3.5 V, got {voltage}"
    # Endothermic insertion means no usable cathode: voltage must go negative.
    assert average_voltage(-10.0 - 1.9 + 0.5, -10.0, 1, -1.9) < 0.0

    # Lithiation expands the cell, so the sign is positive and relative to the host.
    assert abs(volume_change_pct(100.0, 107.0) - 7.0) < 1e-9
    assert volume_change_pct(100.0, 93.0) < 0.0

    # One Li at the body centre of a 4 A cube of framework atoms: clearance = a*sqrt(3)/2.
    cell = Structure(
        Lattice.cubic(4.0), ["Ni", "Li"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    clearance = min_li_clearance(cell)
    assert abs(clearance - 4.0 * 3**0.5 / 2) < 1e-6, f"clearance wrong: {clearance}"
    # No Li means nothing to measure, not a crash.
    assert min_li_clearance(Structure(Lattice.cubic(4.0), ["Ni"], [[0, 0, 0]])) is None
    print(f"  cycling math: V=3.5 V, dV=+7.0%, clearance={clearance:.3f} A")


def check_delithiation_ladder():
    """Every partial state must appear, in order, with the framework untouched."""
    from mattergen.cycling_screen import delithiation_ladder, framework_only, li_removal_order

    # Mn framework plus 3 Li, two of them deliberately crowded together.
    lithiated = Structure(
        Lattice.cubic(8.0),
        ["Mn", "O", "Li", "Li", "Li"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.20, 0.0, 0.0],  # crowded pair
            [0.28, 0.0, 0.0],
            [0.0, 0.6, 0.3],  # isolated
        ],
    )

    # The isolated Li has the largest nearest-neighbour distance, so it goes last.
    order = li_removal_order(lithiated)
    assert len(order) == 3, f"expected 3 Li in the removal order, got {order}"
    assert order[-1] == 4, f"the isolated Li must be removed last, order was {order}"

    rungs = delithiation_ladder(lithiated)
    assert [n for n, _ in rungs] == [3, 2, 1, 0], f"ladder rungs wrong: {[n for n, _ in rungs]}"
    for num_li, structure in rungs:
        assert sum(1 for s in structure if s.specie.symbol == "Li") == num_li
        # Delithiation removes Li and nothing else; the host must be identical.
        assert framework_only(structure).composition == framework_only(lithiated).composition
    print(f"  ladder: Li3 -> Li0 in {len(rungs)} rungs, framework preserved at each")


def _bridge_graph(m_separation: float, bridging_anions: int, box: float = 12.0):
    """Two metals bridged by 1 or 2 anions, isolated in a large box.

    One bridging anion, placed on the M-M axis, is a linear corner-sharing bridge.
    Two bridging anions placed off-axis force the metals close together, which is
    the edge-sharing motif that blocks Li diffusion.
    """
    cart = [[0.0, 0.0, 0.0], [m_separation, 0.0, 0.0]]
    species = [28, 28]
    if bridging_anions == 1:
        cart.append([m_separation / 2.0, 0.0, 0.0])
        species.append(8)
    else:
        offset = 1.5
        cart += [[m_separation / 2.0, offset, 0.0], [m_separation / 2.0, -offset, 0.0]]
        species += [8, 8]

    frac = torch.tensor(cart, dtype=torch.float64) / box
    n = len(species)
    return build_physics_graph(
        frac_coords=frac,
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * box,
        atomic_numbers=torch.tensor(species, dtype=torch.long),
        num_atoms=torch.tensor([n], dtype=torch.long),
        batch_idx=torch.zeros(n, dtype=torch.long),
        num_graphs=1,
    )


def check_connectivity_prefers_corner_sharing():
    """Edge-sharing polyhedra must be penalised far more than a linear corner bridge."""
    loss_fn = PolyhedralConnectivityLoss()

    corner = loss_fn(_bridge_graph(m_separation=4.0, bridging_anions=1))
    edge = loss_fn(_bridge_graph(m_separation=2.9, bridging_anions=2))

    assert corner is not None and edge is not None, "bridging anion produced no cation pair"
    assert corner.item() < 1e-6, f"linear corner bridge should score ~0, got {corner.item():.4f}"
    assert edge.item() > corner.item(), "edge-sharing must be penalised above corner-sharing"
    print(f"  connectivity: corner={corner.item():.2e}  edge-sharing={edge.item():.4f}")


def check_bridge_angle_penalty():
    """A corner bridge must be penalised once its M-O-M angle closes below the cutoff."""
    loss_fn = PolyhedralConnectivityLoss(min_bridge_angle_deg=130.0)

    # Both metals 4.2 A apart, but the bridging anion pushed off-axis so the
    # M-O-M angle closes to well under 130 degrees.
    frac = torch.tensor(
        [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [2.1, 2.0, 0.0]], dtype=torch.float64
    ) / 12.0
    graph = build_physics_graph(
        frac_coords=frac,
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 12.0,
        atomic_numbers=torch.tensor([28, 28, 8], dtype=torch.long),
        num_atoms=torch.tensor([3], dtype=torch.long),
        batch_idx=torch.zeros(3, dtype=torch.long),
        num_graphs=1,
    )
    bent = loss_fn(graph)
    straight = loss_fn(_bridge_graph(m_separation=4.2, bridging_anions=1))

    assert bent is not None and bent.item() > 0.0, "narrow bridge must be penalised"
    assert straight.item() < bent.item(), "a 180 degree bridge must beat a narrow one"
    print(f"  bridge angle: linear={straight.item():.2e}  bent={bent.item():.4f}")


def check_tetrahedral_target():
    """A four-coordinate centre must be scored against 109.47 degrees, not 90/180."""
    loss_fn = PolyhedralGeometryLoss()

    # Ideal tetrahedron: anions on alternating cube corners around the cation.
    box = 12.0
    d = 1.0
    corners = [[d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]]
    frac = torch.tensor([[0.0, 0.0, 0.0]] + corners, dtype=torch.float64) / box
    graph = build_physics_graph(
        frac_coords=frac,
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * box,
        atomic_numbers=torch.tensor([14, 8, 8, 8, 8], dtype=torch.long),
        num_atoms=torch.tensor([5], dtype=torch.long),
        batch_idx=torch.zeros(5, dtype=torch.long),
        num_graphs=1,
    )
    loss = loss_fn(graph)
    assert loss is not None, "tetrahedral centre produced no loss"
    assert loss.item() < 1e-6, f"ideal tetrahedron should score ~0, got {loss.item():.6f}"
    print(f"  tetrahedral target: ideal SiO4 scores {loss.item():.2e}")


def check_guidance_gradient():
    """Sampling-time steering must return a usable gradient under no_grad."""
    frac, cell, z, num_atoms, batch_idx = _octahedral_nio(distortion=0.06)

    with torch.no_grad():  # sampling runs inside no_grad; the helper must still work
        grad = polyhedral_guidance_grad(
            frac_coords=frac,
            cell=cell,
            atomic_numbers=z,
            num_atoms=num_atoms,
            batch_idx=batch_idx,
            num_graphs=1,
            node_weight=torch.ones(4),
            geometry_loss=PolyhedralGeometryLoss(),
            connectivity_loss=PolyhedralConnectivityLoss(),
        )

    assert grad is not None, "guidance returned no gradient"
    assert grad.shape == frac.shape, f"gradient shape {tuple(grad.shape)} != coords"
    assert torch.isfinite(grad).all(), "guidance gradient is not finite"
    assert grad.norm().item() > 0.0, "guidance gradient is identically zero"
    print(f"  guidance gradient norm: {grad.norm().item():.4f}")


def check_guidance_reduces_loss():
    """A step along the negative guidance gradient must improve the geometry."""
    geometry, connectivity = PolyhedralGeometryLoss(), PolyhedralConnectivityLoss()
    frac, cell, z, num_atoms, batch_idx = _octahedral_nio(distortion=0.06)

    def total(coords):
        graph = build_physics_graph(
            frac_coords=coords,
            cell=cell,
            atomic_numbers=z,
            num_atoms=num_atoms,
            batch_idx=batch_idx,
            num_graphs=1,
            node_weight=torch.ones(4),
        )
        terms = [fn(graph) for fn in (geometry, connectivity)]
        return sum(t for t in terms if t is not None)

    before = total(frac)
    grad = polyhedral_guidance_grad(
        frac_coords=frac,
        cell=cell,
        atomic_numbers=z,
        num_atoms=num_atoms,
        batch_idx=batch_idx,
        num_graphs=1,
        node_weight=torch.ones(4),
        geometry_loss=geometry,
        connectivity_loss=connectivity,
    )
    after = total(frac - 0.01 * grad / grad.norm())

    assert after < before, f"guidance step made geometry worse: {before:.5f} -> {after:.5f}"
    print(f"  guidance step: loss {before:.5f} -> {after:.5f}")


def check_connectivity_noise_fade():
    """Connectivity guidance must vanish at zero node weight, like the other terms."""
    loss_fn = PolyhedralConnectivityLoss()
    graph = _bridge_graph(m_separation=2.9, bridging_anions=2)

    active = loss_fn(graph)
    graph.node_weight = torch.zeros_like(graph.node_weight)
    faded = loss_fn(graph)

    assert active.item() > 0.0, "connectivity loss should be non-zero at full weight"
    assert faded.item() == 0.0, "connectivity loss should vanish at zero weight"
    print("  connectivity loss vanishes at zero node weight")


def main():
    checks = [
        ("physics graph", check_physics_graph),
        ("octahedral loss", check_octahedral_loss),
        ("redox loss", check_redox_loss),
        ("redox loss guard", check_no_redox_metal_is_skipped),
        ("noise fade", check_noise_fade),
        ("gradient flow", check_gradients_flow),
        ("redox capacity", check_redox_capacity),
        ("unknown element", check_unknown_element_rejected),
        ("anion dimers", check_anion_dimer_detection),
        ("overoxidized host", check_overoxidized_host_rejected),
        ("Li-cation separation", check_li_cation_separation),
        ("corner vs edge sharing", check_connectivity_prefers_corner_sharing),
        ("bridge angle", check_bridge_angle_penalty),
        ("tetrahedral target", check_tetrahedral_target),
        ("guidance gradient", check_guidance_gradient),
        ("guidance improves geometry", check_guidance_reduces_loss),
        ("connectivity noise fade", check_connectivity_noise_fade),
        ("cycling math", check_cycling_math),
        ("delithiation ladder", check_delithiation_ladder),
        ("real host placement", check_placement_on_real_host),
    ]
    for name, check in checks:
        print(f"[{name}]")
        check()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
