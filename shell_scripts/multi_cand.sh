#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=5
#SBATCH --partition=gpu
#SBATCH --mem=64gb

#SBATCH --time=00:05:00

#SBATCH --job-name=multi_cand_eval
#SBATCH --output=hpglogs/multi_cand_eval_%j.out
#SBATCH --error=hpglogs/multi_cand_eval_%j.err

module load conda
conda activate eccentric_env

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python /HR/eval_scripts/multi_cand_eval.py