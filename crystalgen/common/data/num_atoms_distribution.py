# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

NUM_ATOMS_DISTRIBUTIONS = {
    "ALEX_MP_20": {
        1: 0.0002303828963737732,
        2: 0.002804088967292211,
        3: 0.019342289742695216,
        4: 0.1636343889258233,
        5: 0.04668051158167732,
        6: 0.07808005476530565,
        7: 0.027247714272549548,
        8: 0.1150400537121267,
        9: 0.048984340545415055,
        10: 0.12620539622566992,
        11: 0.03577352703049611,
        12: 0.14591300741832927,
        13: 0.0060031200426537475,
        14: 0.028628366058675234,
        15: 0.02022761830161729,
        16: 0.04473213051520198,
        17: 0.0013033089566287742,
        18: 0.038699389814443035,
        19: 0.0070135136024644384,
        20: 0.04345679662456145,
    }
}

# Cell-size extrapolation probes. The base checkpoint never saw a cell with more
# than 20 atoms, but the num_atoms dependence enters the corruption process
# analytically -- the lattice prior is centred on num_atoms/limit_density and the
# fractional-coordinate noise is scaled by num_atoms**(-1/3) -- so the model may
# extrapolate. These two distributions exist to measure whether it does: sample
# both, and compare novelty and validity of LARGE against SMALL as the control.
#
# ponytail: flat distributions, not fitted histograms. They are a diagnostic, not
# a sampling target. Fit a real >20 histogram from Alexandria if this ships.
NUM_ATOMS_DISTRIBUTIONS["TEST_SMALL"] = {n: 1 / 7 for n in (8, 10, 12, 14, 16, 18, 20)}
NUM_ATOMS_DISTRIBUTIONS["TEST_LARGE"] = {n: 1 / 6 for n in (22, 24, 26, 28, 30, 32)}
