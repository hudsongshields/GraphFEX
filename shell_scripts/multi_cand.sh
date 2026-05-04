#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=2
#SBATCH --partition=hpg-b200
#SBATCH --mem=20gb

#SBATCH --time=12:00:00

#SBATCH --job-name=multi_cand_eval
#SBATCH --output=hpglogs/multi_cand_eval_%j.out
#SBATCH --error=hpglogs/multi_cand_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hudsonshields@ufl.edu

cd /blue/chunmei.wang/hudsonshields/GraphFEX

module load conda
conda activate eccentric_env

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python -u HR/scripts/eval_controller.py
