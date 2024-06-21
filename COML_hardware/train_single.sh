#!/bin/bash
#SBATCH --job-name train_pnorm_single
#SBATCH -o %j.log

# Initialize the module command first source
source /etc/profile

# Load modules
module load anaconda/2023a

python train_z_up_kR.py "$@"
