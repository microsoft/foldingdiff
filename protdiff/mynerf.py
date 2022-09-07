"""
NERF!

References:
https://benjamin-computer.medium.com/protein-loops-in-tensorflow-a-i-bio-part-2-f1d802ef8300
https://www.biotite-python.org/examples/gallery/structure/peptide_assembly.html
"""
import os
import itertools
from typing import *

import numpy as np


N_CA_LENGTH = 1.46
CA_C_LENGTH = 1.54
C_N_LENGTH = 1.34
C_O_LENGTH = 1.43

# Taken from initial coords from 1CRN, which is a THR
N_INIT = np.array([17.047, 14.099, 3.625])
CA_INIT = np.array([16.967, 12.784, 4.338])
C_INIT = np.array([15.685, 12.755, 5.133])


class NERFBuilder:
    """
    Builder for NERF
    """

    def __init__(
        self,
        phi_dihedrals: np.ndarray,
        psi_dihedrals: np.ndarray,
        omega_dihedrals: np.ndarray,
        bond_len_n_ca: Union[float, np.ndarray] = N_CA_LENGTH,
        bond_len_ca_c: Union[float, np.ndarray] = CA_C_LENGTH,
        bond_len_c_n: Union[float, np.ndarray] = C_N_LENGTH,
        bond_angle_n_ca: Union[float, np.ndarray] = 121 / 180 * np.pi,
        bond_angle_ca_c: Union[float, np.ndarray] = 109 / 180 * np.pi,
        bond_angle_c_n: Union[float, np.ndarray] = 115 / 180 * np.pi,
        init_coords: np.ndarray = [N_INIT, CA_INIT, C_INIT],
    ) -> None:
        self.phi = phi_dihedrals.squeeze()
        self.psi = psi_dihedrals.squeeze()
        self.omega = omega_dihedrals.squeeze()

        # We start with coordinates for N --> CA --> C so the next atom we add
        # is the next N. Therefore, the first angle we need is the
        self.bond_lengths = {
            ("C", "N"): bond_len_c_n,
            ("N", "CA"): bond_len_n_ca,
            ("CA", "C"): bond_len_ca_c,
        }
        self.bond_angles = {
            ("C", "N"): bond_angle_c_n,
            ("N", "CA"): bond_angle_n_ca,
            ("CA", "C"): bond_angle_ca_c,
        }
        if init_coords is not None:
            self.init_coords = [c for c in init_coords]
        else:
            raise NotImplementedError

        self.bonds = itertools.cycle(self.bond_angles.keys())

    def build(self):
        """Build out the molecule"""
        retval = self.init_coords.copy()

        # The first value of phi at the N terminus is not defined
        # The last value of psi and omega at the C terminus are not defined
        for i, (phi, psi, omega) in enumerate(
            zip(self.phi[1:], self.psi[:-1], self.omega[:-1])
        ):
            # Procedure
            # Place the next N atom, which requires the C-N bond length/angle, and the psi dihedral
            n_coords = place_dihedral(
                retval[-3],
                retval[-2],
                retval[-1],
                bond_angle=self.bond_angles[("C", "N")],
                bond_length=self.bond_lengths[("C", "N")],
                torsion_angle=psi,
            )
            retval.append(n_coords)
            # Place the alpha carbon, which requires the N-CA bond length/angle, and the omega dihedral
            ca_coords = place_dihedral(
                retval[-3],
                retval[-2],
                retval[-1],
                bond_angle=self.bond_angles[("N", "CA")],
                bond_length=self.bond_angles[("N", "CA")],
                torsion_angle=omega,
            )
            retval.append(ca_coords)
            # Place the carbon, which requires the the CA-C bond length/angle, and the phi dihedral
            c_coords = place_dihedral(
                retval[-3],
                retval[-2],
                retval[-1],
                bond_angle=self.bond_angles[("CA", "C")],
                bond_length=self.bond_angles[("CA", "C")],
                torsion_angle=phi,
            )
            retval.append(c_coords)

        return np.array(retval)


def place_dihedral(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_angle: float,
    bond_length: float,
    torsion_angle: float,
) -> np.ndarray:
    """
    Place the point d such that the bond angle, length, and torsion angle are satisfied
    with the series a, b, c, d.
    """
    assert a.ndim == b.ndim == c.ndim == 1
    unit_vec = lambda x: x / np.linalg.norm(x)
    ab = b - a
    bc = unit_vec(c - b)
    d = np.array(
        [
            -bond_length * np.cos(bond_angle),
            bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
            bond_length * np.sin(torsion_angle) * np.sin(bond_angle),
        ]
    )
    n = unit_vec(np.cross(ab, bc))
    nbc = np.cross(n, bc)
    m = np.stack([bc, nbc, n]).T
    d = m.dot(d)
    return d + c


def main():
    """On the fly testing"""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    source = PDBFile.read(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/1CRN.pdb")
    )
    source_struct = source.get_structure()
    # print(source_struct[0])
    phi, psi, omega = struc.dihedral_backbone(source_struct)

    builder = NERFBuilder(phi, psi, omega)
    built = builder.build()
    print(built)
    print(built.shape)


if __name__ == "__main__":
    main()
