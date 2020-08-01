#! /bin/bash

#SBATCH --job-name=qhat
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --partition=std
#SBATCH --time=24:00:00
#SBATCH --array=1-141
#SBATCH --output=/rstorage/james/qhat/slurm-%A_%a.out

srun process_analysis.sh $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
