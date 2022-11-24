# cd into data directory and execute shell script
wget -P cath ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz

# Check md5sums
md5sum -c cath_checksums.md5

# cd into the cath directory and untar the file
cd cath
tar -xzf cath-dataset-nonredundant-S40.pdb.tgz