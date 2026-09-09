# Run-4 screening: structure / composition validity and novelty.
"""Minimal validity and novelty metrics for generated structures.

Faithful reimplementation of the subset of upstream MatterGen's evaluation
stack that Paper 1's Run 4 needs, without the LMDB reference-dataset
machinery (novelty here is computed against a local directory of CIFs, e.g.
the 4544 delithiated training hosts):

- `is_structure_valid`: no two atoms closer than `min_dist_threshold` (CDVAE
  convention, 0.5 A).
- `is_smact_valid`: composition can be charge balanced with physically
  sensible oxidation states (SMACT + Pauling electronegativity test), ported
  from upstream's evaluation/metrics/structure.py.
- `novelty_report`: per-structure novelty vs a reference set via pymatgen's
  StructureMatcher, pre-filtered by reduced formula so 200 x 4544 collapses
  to a few hundred matcher calls.
"""

import itertools
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import smact
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Structure
from smact.screening import pauling_test


def min_pairwise_distance(structure: Structure) -> float:
    """Shortest interatomic distance in the cell, PBC-aware.

    Uses pymatgen's min-image distance matrix; N is at most a few dozen for
    generated cathode cells, so the O(N^2) cost is irrelevant.
    """
    dm = structure.distance_matrix
    n = len(structure)
    if n < 2:
        return float("nan")
    # Mask the diagonal (self-distances of 0).
    mask = ~np.eye(n, dtype=bool)
    return float(dm[mask].min())


def is_structure_valid(structure: Structure, min_dist_threshold: float = 0.5) -> bool:
    """True if no two atoms are closer than `min_dist_threshold` Angstrom."""
    return min_pairwise_distance(structure) >= min_dist_threshold


def smact_validity(
    comp: tuple,
    count: tuple,
    use_pauling_test: bool = True,
    include_alloys: bool = True,
    include_cutoff: bool = False,
) -> bool:
    """SMACT charge-balance + electronegativity validity (upstream port)."""
    assert len(comp) == len(count)
    elem_symbols = tuple([str(Element.from_Z(Z=elem)) for elem in comp])  # type: ignore
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    n_comb = np.prod([len(ls) for ls in ox_combos])
    if n_comb > 1e6 and include_cutoff:
        return True
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # smact >=4 returns a plain list; older versions returned (list, bool)
        cn_e = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_OK = True
                if electroneg_OK:
                    return True
            else:
                return True
    return False


def is_smact_valid(structure: Structure) -> bool:
    """True if the structure's composition passes the SMACT check."""
    elem_counter = Counter(structure.atomic_numbers)
    composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    elems, counts = list(zip(*composition))
    counts = np.array(counts)
    counts = counts / np.gcd.reduce(counts)
    comps: tuple = tuple(np.array(counts).astype("int"))
    try:
        return smact_validity(comp=elems, count=comps, use_pauling_test=True, include_alloys=True)
    except UnicodeDecodeError:
        # HOTFIX (upstream): a decode error sometimes occurs on the first call
        return smact_validity(comp=elems, count=comps, use_pauling_test=True, include_alloys=True)


def load_reference_formulas(reference_dir) -> Dict[str, int]:
    """Reduced formula -> count of reference structures with that formula."""
    from pathlib import Path

    formulas: Dict[str, int] = {}
    for cif in sorted(Path(reference_dir).glob("*.cif")):
        try:
            s = Structure.from_file(str(cif))
            f = s.composition.reduced_formula
            formulas[f] = formulas.get(f, 0) + 1
        except Exception:
            continue
    return formulas


def load_reference_structures(reference_dir) -> Dict[str, List[Structure]]:
    """Reduced formula -> reference Structures. Built once per session and
    shared across arms (4544 CIFs take a couple of minutes to parse)."""
    from pathlib import Path

    ref_by_formula: Dict[str, List[Structure]] = {}
    for cif in sorted(Path(reference_dir).glob("*.cif")):
        try:
            s = Structure.from_file(str(cif))
        except Exception:
            continue
        ref_by_formula.setdefault(s.composition.reduced_formula, []).append(s)
    return ref_by_formula


def novelty_report(
    generated: List[Structure],
    reference: Dict[str, List[Structure]] | str | Path,
    matcher: Optional[StructureMatcher] = None,
) -> List[Dict]:
    """Per-structure novelty vs a reference set.

    `reference` is either a formula -> Structures mapping (from
    load_reference_structures) or a directory of CIFs. A structure is *not
    novel* if its reduced formula appears in the reference set and it matches
    at least one same-formula reference under StructureMatcher. ponytail: no
    anonymous-composition fallback; a different-composition structure is
    novel by definition here.
    """
    from pathlib import Path

    if matcher is None:
        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=False)

    if isinstance(reference, (str, Path)):
        ref_by_formula = load_reference_structures(reference)
    else:
        ref_by_formula = reference

    records = []
    for i, s in enumerate(generated):
        formula = s.composition.reduced_formula
        candidates = ref_by_formula.get(formula, [])
        matched = False
        for ref in candidates:
            if matcher.fit(s, ref):
                matched = True
                break
        records.append(
            {
                "index": i,
                "formula": formula,
                "n_reference_same_formula": len(candidates),
                "novel": not matched,
            }
        )
    return records
