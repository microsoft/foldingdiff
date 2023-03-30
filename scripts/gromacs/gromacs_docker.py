"""
Simple script to wrap the gromacs.py file in an easy to run docker container
to avoid all the messiness of trying to do mounting and stuff.

Usage: python gromacs_docker.py <input_file> <output_dir>
"""
import os
import logging
import shutil
import tempfile
import subprocess
import argparse


def build_parser():
    """Build a basic CLI parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="Input file to run GROMACS on")
    parser.add_argument("output_dir", help="Output dir to write output files to")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    return parser


def run_gromacs_in_docker(fname: str, out_dir: str, gpu: int = 0):
    """
    Run gromacs in docker
    """
    assert os.path.isfile(fname), f"Input file {fname} not found"
    assert shutil.which("nvidia-docker")
    out_dir = os.path.abspath(out_dir)
    fname = os.path.abspath(fname)
    bname = os.path.splitext(os.path.basename(fname))[0]

    orig_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        logging.info(f"Running {fname} via docker in temporary directory {tmpdir}")
        assert not os.listdir(tmpdir)
        os.chdir(tmpdir)
        # Copy the file into the directory
        shutil.copy(fname, tmpdir)
        # Build and run the command
        # https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#i-have-multiple-gpu-devices-how-can-i-isolate-them-between-my-containers
        cmd = f"nvidia-docker run -it --rm -e NVIDIA_VISIBLE_DEVICES={gpu} -v {tmpdir}:/host_pwd --workdir /host_pwd wukevin:gromacs-latest {os.path.basename(fname)}"
        logging.info(f"Running command: {cmd}")
        with open(os.path.join(out_dir, f"{bname}.gromacs.stdout"), "wb") as stdout:
            with open(os.path.join(out_dir, f"{bname}.gromacs.stderr"), "wb") as stderr:
                subprocess.call(cmd, shell=True, stdout=stdout, stderr=stderr)

        for src_fname in os.listdir(tmpdir):
            dest_fname = (
                src_fname
                if src_fname.startswith(bname)
                else ".".join([bname, src_fname])
            )
            logging.info(f"Copying {src_fname} to {dest_fname} in {out_dir}")
            shutil.copy(
                os.path.join(tmpdir, src_fname), os.path.join(out_dir, dest_fname)
            )
    os.chdir(orig_dir)  # Restore directory


def main():
    """Run script"""
    args = build_parser().parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    for fname in [os.path.abspath(f) for f in args.input_file]:
        run_gromacs_in_docker(fname, args.output_dir, gpu=args.gpu)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
