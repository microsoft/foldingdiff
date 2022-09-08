# cd into data directory and execute shell script
# md5 cbb61487871603181d5e38e1ebff486a
wget -P cath http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl
# md5 0b84c1c522e51ebfe97e92e5cf0f36cc
wget -P cath http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json

# download the source files
wget -P cath ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz

# Check md5sums
md5sum -c cath_checksums.md5 cath/*

# cd into the cath directory and untar the file
cd cath
tar -xzf cath-dataset-nonredundant-S40.pdb.tgz