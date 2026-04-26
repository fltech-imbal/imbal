#!/bin/bash

#SBATCH --job-name SEP-EC-decoupled-RI
#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --mem=16GB

#SBATCH --time=12:00:00
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1

#SBATCH --error=slurm-out/sep-ec-decoupled-ri-%J.err 
#SBATCH --output=slurm-out/sep-ec-decoupled-ri-%J.out

module load mpich

echo "Starting at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is $(pwd)"

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate imbal-env

python sep_ec_decoupled_ri.py
