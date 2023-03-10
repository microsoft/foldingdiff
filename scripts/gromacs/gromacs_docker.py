"""
Simple script to wrap the gromacs.py file in an easy to run docker container
to avoid all the messiness of trying to do mounting and stuff.

Usage: python gromacs_docker.py <input_file> <output_dir>
"""
import os
import shutil
import tempfile
import subprocess
import argparse


def build_parser():
    """Build a basic CLI parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input file to run GROMACS on")
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
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        # Copy the file into the directory
        shutil.copy(fname, tmpdir)
        # Build and run the command
        # https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#i-have-multiple-gpu-devices-how-can-i-isolate-them-between-my-containers
        cmd = f"nvidia-docker run -it --rm -e NVIDIA_VISIBLE_DEVICES={gpu} -v {tmpdir}:/host_pwd --workdir /host_pwd wukevin:gromacs-latest {os.path.basename(fname)}"
        subprocess.call(cmd, shell=True)

        for fname in os.listdir(tmpdir):
            shutil.copy(fname, out_dir)


def main():
    """Run script"""
    args = build_parser().parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    run_gromacs_in_docker(args.input_file, args.output_dir, gpu=args.gpu)


if __name__ == "__main__":
    main()
