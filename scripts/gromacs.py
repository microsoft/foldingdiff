"""
Script to run GROMACS on an input file

Take the 20 amino acids, build a skeleton of each side chain
Map the coordinates into the glycine

Don't even need a great "starting structure" - can be in the general proximity
GROMACS can do no hydrogen
pdb2gmx ignore h to add them in
"""

import os, sys
import tempfile
import logging
import shlex
import subprocess
import shutil

GRO_FILE_DIR = "/home/wukevin/software/md_template/mdp/"


def run_gromacs(pdb_file: str, outdir: str = os.getcwd()) -> float:
    """
    Run GROMACS on a PDB file
    """
    gro_file = os.path.join(outdir, os.path.basename(pdb_file).replace(".pdb", ".gro"))
    # pdb2gmx = f"gmx pdb2gmx -f {pdb_file} -o {gro_file} -ff 6 -water tip3p"
    # Puts it in a GMX format, add water and force field
    # AMBER/CHARM most common for protein and protein folding.
    # A force field defines all forces/energies interacting on
    # a given atom.
    pdb2gmx = f"gmx pdb2gmx -f {pdb_file} -o {gro_file} -water tip3p"
    logging.debug(f"pdb2gmx cmd: {pdb2gmx}")
    p = subprocess.Popen(shlex.split(pdb2gmx), stdin=subprocess.PIPE)
    p.communicate(input="6".encode())

    # gen box - put this in a water solvent 'in a box' - 1nm around the system
    box_file = os.path.join(outdir, "box.gro")
    box_cmd = f"gmx editconf -f {gro_file} -o {box_file} -c -d 1"
    subprocess.call(shlex.split(box_cmd))

    # solvate - add water
    solvate_cmd = f"gmx solvate -cp {box_file} -o solv.gro -cs spc216.gro -p topol.top"
    logging.debug(f"solvate cmd: {solvate_cmd}")
    subprocess.call(shlex.split(solvate_cmd))

    # add ions - add counter postive and negative ions to make
    # the box "neutral"
    ions_cmd = (
        f"gmx grompp -f {GRO_FILE_DIR}ions.mdp -c solv.gro -o ions.tpr -p topol.top"
    )
    logging.debug(f"ions cmd: {ions_cmd}")
    subprocess.call(shlex.split(ions_cmd))

    genion_cmd = (
        f"gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral"
    )
    logging.debug(f"genion cmd: {genion_cmd}")
    p = subprocess.Popen(shlex.split(genion_cmd), stdin=subprocess.PIPE)
    p.communicate(input="13".encode())

    # Energy minimization - remove unfavorable contacts
    # like making sure nothing is overlapping; nothing should
    # change too much
    em_cmd = f"gmx grompp -f {GRO_FILE_DIR}minim.mdp -c ions.gro -o em.tpr -p topol.top"
    logging.debug(f"em cmd: {em_cmd}")
    subprocess.call(shlex.split(em_cmd))

    mdrun_cmd = "gmx mdrun -deffnm em"
    logging.debug(f"mdrun cmd: {mdrun_cmd}")
    subprocess.call(shlex.split(mdrun_cmd))

    # NVT - equilibrate the system at constant volume and temperature
    # come to "room temperature"
    grompp_cmd = f"gmx grompp -f {GRO_FILE_DIR}nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr"
    subprocess.call(shlex.split(grompp_cmd))
    nvt_cmd = "gmx mdrun -deffnm nvt"
    subprocess.call(shlex.split(nvt_cmd))

    # NPT
    grompp_cmd = (
        f"gmx grompp -f {GRO_FILE_DIR}npt.mdp -c nvt.gro -o npt.tpr -p topol.top"
    )
    subprocess.call(shlex.split(grompp_cmd))
    npt_cmd = "gmx mdrun -ntmpi 1 -ntomp 32 -pin on -deffnm npt"
    subprocess.call(shlex.split(npt_cmd))

    # Production run
    grompp_cmd = f"gmx grompp -f {GRO_FILE_DIR}md.mdp -c npt.gro -t npt.cpt -p topol.top -o prod.tpr"
    subprocess.call(shlex.split(grompp_cmd))
    prod_cmd = "gmx mdrun -ntmpi 1 -ntomp 32 -pin on -deffnm prod"
    subprocess.call(shlex.split(prod_cmd))

    # Read energy and return
    return read_energy("prod.edr")


def read_energy(energy_edr_file: str) -> float:
    """
    Read energy from GROMACS energy file
    """
    cmd = f"gmx energy -f {energy_edr_file} -o energy.xvg"
    p = subprocess.Popen(
        shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    stdout = p.communicate(input="11\n\n".encode())[0].decode().split("\n")
    potential_lines = [l for l in stdout if l.startswith("Potential")]
    assert len(potential_lines) == 1, "Unexpected number of potential lines"
    energy = float(potential_lines[0].split()[1])
    return energy


def main():
    """Run script"""
    assert shutil.which("gmx") is not None, "GROMACS not found in PATH"
    orig_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        energy = run_gromacs(sys.argv[1], tmpdir)
        files = os.listdir(tmpdir)
        for file in files:
            logging.info(f"GROMACS file: {file}")
            if file.endswith(".edr"):
                shutil.copy(file, orig_dir)
    logging.info(f"{sys.argv[1]} energy: {energy:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
