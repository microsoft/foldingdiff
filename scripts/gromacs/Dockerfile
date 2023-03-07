# Useful links:
# https://github.com/docker/buildx/issues/476
# Sometimes need to run sudo service docker restart to build correctly
# https://stackoverflow.com/questions/50309605/reading-input-files-with-docker

# Example command:
# nvidia-docker run -it --rm -v ${PWD}:/host_pwd --workdir /host_pwd wukevin:gromacs-latest generated_0_proteinmpnn_residues_0.pdb

FROM nvcr.io/hpc/gromacs:2022.3

# Should already come with python3; just copy in files
COPY gromacs.py /usr/local/bin/gromacs.py
COPY mdp /usr/local/bin/mdp

ENTRYPOINT [ "python3", "/usr/local/bin/gromacs.py", "--gmxbin", "/usr/local/gromacs/avx2_256/bin/gmx", "--mdp", "/usr/local/bin/mdp/"]