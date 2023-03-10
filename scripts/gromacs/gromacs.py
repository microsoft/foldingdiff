"""
Script to run GROMACS on an input file

Take the 20 amino acids, build a skeleton of each side chain
Map the coordinates into the glycine

Don't even need a great "starting structure" - can be in the general proximity
GROMACS can do no hydrogen
pdb2gmx ignore h to add them in
"""

import os
import sys
import socket
import argparse
import tempfile
import logging
import shlex
import subprocess
import shutil

GRO_FILE_DIR = os.path.join(os.path.dirname(__file__), "mdp")


def run_gromacs(
    pdb_file: str,
    outdir: str = os.getcwd(),
    gmx: str = "gmx",
    gro_file_dir: str = GRO_FILE_DIR,
    n_threads: int = 8,
) -> float:
    """
    Run GROMACS on a PDB file
    """
    assert os.path.isfile(pdb_file), f"File {pdb_file} not found! (pwd: {os.getcwd()})"
    gro_file = os.path.join(outdir, os.path.basename(pdb_file).replace(".pdb", ".gro"))
    # pdb2gmx = f"gmx pdb2gmx -f {pdb_file} -o {gro_file} -ff 6 -water tip3p"
    # Puts it in a GMX format, add water and force field
    # AMBER/CHARM most common for protein and protein folding.
    # A force field defines all forces/energies interacting on
    # a given atom.
    pdb2gmx = f"{gmx} pdb2gmx -f {pdb_file} -o {gro_file} -water tip3p"
    logging.debug(f"pdb2gmx cmd: {pdb2gmx}")
    p = subprocess.Popen(shlex.split(pdb2gmx), stdin=subprocess.PIPE)
    p.communicate(input="6".encode())

    # gen box - put this in a water solvent 'in a box' - 1nm around the system
    box_file = os.path.join(outdir, "box.gro")
    box_cmd = f"{gmx} editconf -f {gro_file} -o {box_file} -c -d 1"
    subprocess.call(shlex.split(box_cmd))

    # solvate - add water
    solvate_cmd = (
        f"{gmx} solvate -cp {box_file} -o solv.gro -cs spc216.gro -p topol.top"
    )
    logging.debug(f"solvate cmd: {solvate_cmd}")
    subprocess.call(shlex.split(solvate_cmd))

    # add ions - add counter postive and negative ions to make
    # the box "neutral"
    ions_cmd = (
        f"{gmx} grompp -f {gro_file_dir}ions.mdp -c solv.gro -o ions.tpr -p topol.top"
    )
    logging.debug(f"ions cmd: {ions_cmd}")
    subprocess.call(shlex.split(ions_cmd))

    genion_cmd = f"{gmx} genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral"
    logging.debug(f"genion cmd: {genion_cmd}")
    p = subprocess.Popen(shlex.split(genion_cmd), stdin=subprocess.PIPE)
    p.communicate(input="13".encode())

    # Energy minimization - remove unfavorable contacts
    # like making sure nothing is overlapping; nothing should
    # change too much
    em_cmd = (
        f"{gmx} grompp -f {gro_file_dir}minim.mdp -c ions.gro -o em.tpr -p topol.top"
    )
    logging.debug(f"EM cmd: {em_cmd}")
    subprocess.call(shlex.split(em_cmd))

    mdrun_cmd = f"{gmx} mdrun -ntmpi 1 -ntomp {n_threads} -deffnm em"
    logging.debug(f"mdrun cmd: {mdrun_cmd}")
    subprocess.call(shlex.split(mdrun_cmd))

    # NVT - equilibrate the system at constant volume and temperature
    # come to "room temperature"
    grompp_cmd = f"{gmx} grompp -f {gro_file_dir}nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr"
    subprocess.call(shlex.split(grompp_cmd))
    nvt_cmd = f"{gmx} mdrun -ntmpi 1 -ntomp {n_threads - 1} -nb gpu -pin on -deffnm nvt"
    subprocess.call(shlex.split(nvt_cmd))

    # NPT
    grompp_cmd = (
        f"{gmx} grompp -f {gro_file_dir}npt.mdp -c nvt.gro -o npt.tpr -p topol.top"
    )
    subprocess.call(shlex.split(grompp_cmd))
    npt_cmd = f"{gmx} mdrun -ntmpi 1 -ntomp {n_threads - 1} -nb gpu -pin on -deffnm npt"
    subprocess.call(shlex.split(npt_cmd))

    # Production run
    grompp_cmd = f"{gmx} grompp -f {gro_file_dir}md.mdp -c npt.gro -t npt.cpt -p topol.top -o prod.tpr"
    subprocess.call(shlex.split(grompp_cmd))
    prod_cmd = (
        f"{gmx} mdrun -ntmpi 1 -ntomp {n_threads - 1} -nb gpu -pin on -deffnm prod"
    )
    subprocess.call(shlex.split(prod_cmd))

    # Produce a PDB of final structure
    pdb_cmd = f"{gmx} editconf -f prod.gro -o prod.pdb"
    subprocess.call(shlex.split(pdb_cmd))

    # Read energy and return
    return read_energy("prod.edr", gmx=gmx)


def read_energy(
    energy_edr_file: str,
    gmx: str = "gmx",
) -> float:
    """
    Read energy from GROMACS energy file
    """
    assert os.path.isfile(energy_edr_file), f"File {energy_edr_file} not found"
    cmd = f"{gmx} energy -f {energy_edr_file} -o energy.xvg"
    p = subprocess.Popen(
        shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    stdout = p.communicate(input="11\n\n".encode())[0].decode().split("\n")
    potential_lines = [l for l in stdout if l.startswith("Potential")]
    assert len(potential_lines) == 1, "Unexpected number of potential lines"
    energy = float(potential_lines[0].split()[1])
    return energy


def build_parser():
    """Build basic CLI parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pdb_file", help="PDB file to run GROMACS on")
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=os.getcwd(),
        help="Directory to write output",
    )
    parser.add_argument("--copyall", action="store_true", help="Copy all GROMACS files")
    parser.add_argument(
        "--gmxbin", type=str, default=shutil.which("gmx"), help="GROMACS binary"
    )
    parser.add_argument(
        "--mdp", type=str, default=GRO_FILE_DIR, help="MDP file directory"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Threads (minimum 2)"
    )
    return parser


def main():
    """Run script"""
    args = build_parser().parse_args()
    logging.info(f"Running under Python {sys.version} in {socket.gethostname()}")
    assert os.path.isdir(args.outdir), f"Directory {args.outdir} not found"
    assert args.gmxbin is not None
    args.pdb_file = os.path.abspath(args.pdb_file)
    # Run in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        energy = run_gromacs(
            args.pdb_file,
            tmpdir,
            gmx=args.gmxbin,
            gro_file_dir=args.mdp,
            n_threads=args.threads,
        )
        for file in os.listdir(tmpdir):
            logging.debug(f"GROMACS file: {file}")
            if args.copyall:
                shutil.copy(file, args.outdir)
            elif file.startswith("prod"):
                shutil.copy(file, args.outdir)
    logging.info(f"{args.pdb_file} energy: {energy:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
