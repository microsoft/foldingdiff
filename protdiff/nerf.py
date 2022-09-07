"""
https://benjamin-computer.medium.com/protein-loops-in-tensorflow-a-i-bio-part-2-f1d802ef8300
"""

import numpy as np
import math, itertools


class NeRF(object):
    def __init__(self):
        # TODO - PROLINE has different lengths which we should take into account
        # TODO - A_TO_C angle differs by +/- 5 degrees
        # bond_lengths = { "N_TO_A" : 1.4615, "PRO_N_TO_A" : 1.353, "A_TO_C" : 1.53, "C_TO_N" : 1.325 }
        self.bond_lengths = {"N_TO_A": 1.4615, "A_TO_C": 1.53, "C_TO_N": 1.325}
        self.bond_angles = {
            "A_TO_C": math.radians(109),
            "C_TO_N": math.radians(115),
            "N_TO_A": math.radians(121),
        }
        self.bond_order = ["C_TO_N", "N_TO_A", "A_TO_C"]

    def _next_data(self, key):
        """Loop over our bond_angles and bond_lengths"""
        ff = itertools.cycle(self.bond_order)
        for item in ff:
            if item == key:
                next_key = next(ff)
                break
        return (self.bond_angles[next_key], self.bond_lengths[next_key], next_key)

    def _place_atom(
        self, atom_a, atom_b, atom_c, bond_angle, torsion_angle, bond_length
    ):
        """Given the three previous atoms, the required angles and the bond
        lengths, place the next atom. Angles are in radians, lengths in angstroms."""
        # TODO - convert to sn-NeRF
        ab = np.subtract(atom_b, atom_a)
        bc = np.subtract(atom_c, atom_b)
        bcn = bc / np.linalg.norm(bc)
        R = bond_length
        # numpy is row major
        d = np.array(
            [
                -R * math.cos(bond_angle),
                R * math.cos(torsion_angle) * math.sin(bond_angle),
                R * math.sin(torsion_angle) * math.sin(bond_angle),
            ]
        )
        n = np.cross(ab, bcn)
        n = n / np.linalg.norm(n)
        nbc = np.cross(n, bcn)
        m = np.array(
            [[bcn[0], nbc[0], n[0]], [bcn[1], nbc[1], n[1]], [bcn[2], nbc[2], n[2]]]
        )
        d = m.dot(d)
        d = d + atom_c
        return d

    def compute_positions(self, torsions, input_is_degrees:bool = False):
        """Call this function with a set of torsions (including omega) in degrees."""
        atoms = [[0, -1.355, 0], [0, 0, 0], [1.4466, 0.4981, 0]]
        if input_is_degrees:
            torsions = list(map(math.radians, torsions))
        key = "C_TO_N"
        angle = self.bond_angles[key]
        length = self.bond_lengths[key]
        for torsion in torsions:
            atoms.append(
                self._place_atom(
                    atoms[-3], atoms[-2], atoms[-1], angle, torsion, length
                )
            )
            (angle, length, key) = self._next_data(key)
        return np.array(atoms)


if __name__ == "__main__":
    nerf = NeRF()
    print("3NH7_1 - using real omega")
    torsions = [
        142.951668191667,
        173.2,
        -147.449854444109,
        137.593755455898,
        -176.98,
        -110.137784727015,
        138.084240732612,
        162.28,
        -101.068226849313,
        -96.1690297398444,
        167.88,
        -78.7796836206707,
        -44.3733790929788,
        175.88,
        -136.836113196726,
        164.182984866024,
        -172.22,
        -63.909882696529,
        143.817250526837,
        168.89,
        -144.50345668635,
        158.70503596547,
        175.87,
        -96.842536650294,
        103.724939588454,
        -172.34,
        -85.7345901579845,
        -18.1379473766538,
        -172.98,
        -150.084356709565,
    ]
    atoms0 = nerf.compute_positions(torsions)
    print(len(atoms0))
    for atom in atoms0:
        print(atom)
