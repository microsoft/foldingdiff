"""
Hand-recorded constants
"""


class CathCanonicalTraining:
    """
    Values calculated from CATH canonical angles/dihedrals/distances
    dataset, randomized training split. These are recorded by running
    the training script on the full dataset with 5 features and manually
    recording the values.
    """

    ft_mean_var = {
        "angles": {
            "bond_dist": (1.3032870292663574, 0.03523094952106476),
            "phi": (-1.308528184890747, 0.7981446385383606),
            "psi": (0.6201950311660767, 2.3921289443969727),
            "omega": (0.37990081310272217, 9.198516845703125),
            "tau": (1.940384030342102, 0.0025286211166530848),
        }
    }
