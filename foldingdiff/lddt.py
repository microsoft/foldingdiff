"""
Code for computing lDDT scores.
"""

import os, sys
from pathlib import Path
import subprocess
import tempfile
import json

IMAGE = "2d07309e7a56"  # Docker image from https://git.scicore.unibas.ch/schwede/openstructure/container_registry/

DOCKER_OST = Path(os.path.realpath(__file__)).parent.parent / "scripts/run_docker_ost"
assert DOCKER_OST.exists(), f"Cannot find docker wrapper script {DOCKER_OST}"


def lddt(query: Path, ref: Path) -> float:
    """Compute the lDDT between query and reference structures."""
    with tempfile.NamedTemporaryFile(dir=os.getcwd()) as outfile:
        cmd = f"{DOCKER_OST} {IMAGE} compare-structures -m {query} -r {ref} --lddt -o {os.path.basename(outfile.name)}"
        subprocess.call(cmd, shell=True)

        # outfile.seek(0)
        data = json.loads(outfile.read().decode("utf-8"))

    if "lddt" in data:
        return data["lddt"]
    return -1.0


if __name__ == "__main__":
    print(lddt(Path(sys.argv[1]), Path(sys.argv[2])))
