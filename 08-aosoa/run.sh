#!/bin/bash
#SBATCH --job-name=ceci-gpu-training
#SBATCH --partition=gpu          # Partition name
#SBATCH --nodes=1                # Total number of nodes
#SBATCH --ntasks-per-node=1      # Number of MPI tasks per node
#SBATCH --gpus-per-node=1        # Number of requested GPUs per GPU node
#SBATCH --cpus-per-task=1        # Number of OpenMP thread per MPI task
#SBATCH --time=0-00:05:00        # Run time (d-hh:mm:ss)
#SBATCH --account=cecigpu        # Project for billing

ml --force purge
ml Python
ml SciPy-bundle
ml UCX-CUDA

srun ncu -o profile --set full -f ./main --profile
srun main
srun python plot.py
