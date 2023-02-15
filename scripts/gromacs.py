"""
Script to run GROMACS on an input file
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
    pdb2gmx = f"gmx pdb2gmx -f {pdb_file} -o {gro_file} -water tip3p"
    logging.debug(f"pdb2gmx cmd: {pdb2gmx}")
    # subprocess.run(shlex.split(pdb2gmx), check=True, stdin="6 1")
    p = subprocess.Popen(shlex.split(pdb2gmx), stdin=subprocess.PIPE)
    p.communicate(input="6".encode())

    # gen box
    box_file = os.path.join(outdir, "box.gro")
    box_cmd = f"gmx editconf -f {gro_file} -o {box_file} -c -d 1"
    subprocess.call(shlex.split(box_cmd))

    # solvate
    solvate_cmd = f"gmx solvate -cp {box_file} -o solv.gro -cs spc216.gro -p topol.top"
    logging.debug(f"solvate cmd: {solvate_cmd}")
    subprocess.call(shlex.split(solvate_cmd))

    # add ions
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

    # Energy minimization
    em_cmd = f"gmx grompp -f {GRO_FILE_DIR}minim.mdp -c ions.gro -o em.tpr -p topol.top"
    logging.debug(f"em cmd: {em_cmd}")
    subprocess.call(shlex.split(em_cmd))

    mdrun_cmd = "gmx mdrun -deffnm em"
    logging.debug(f"mdrun cmd: {mdrun_cmd}")
    subprocess.call(shlex.split(mdrun_cmd))

    # NVT
    grompp_cmd = "gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr"
    subprocess.call(shlex.split(grompp_cmd))
    nvt_cmd = "gmx mdrun -deffnm nvt"
    subprocess.call(shlex.split(nvt_cmd))

    # Read in the energy file and get energy
    energy_file = os.path.join(outdir, "em.log")
    with open(energy_file) as f:
        for line in f:
            if "Potential Energy" in line:
                energy = float(line.split()[3])
                break
    return energy


def main():
    """Run script"""
    assert shutil.which("gmx") is not None, "GROMACS not found in PATH"
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        energy = run_gromacs(sys.argv[1], tmpdir)
    logging.info(f"{sys.argv[1]} energy: {energy:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
