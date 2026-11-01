#!/bin/bash
#SBATCH --job-name=graphfex-hr-maximin
#SBATCH --output=NumericalExperiments/HR/slurm/%x_%j.out
#SBATCH --error=NumericalExperiments/HR/slurm/%x_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xingjianxu@ufl.edu
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00

# Submit from the GraphFEX repository root:
#   sbatch NumericalExperiments/HR/slurm/maximin_operator_search.slurm

set -euo pipefail
module purge || true
module load conda || true
conda activate xxjan

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python NumericalExperiments/HR/scripts/maximin_operator_search.py \
  --snr 30 \
  --num-groups 4 \
  --fex-epochs 8 \
  --cands-per-epoch 4 \
  --limit-hours 9.75 \
  --device cuda \
  --seed 42 \
  --summary "NumericalExperiments/HR/slurm/maximin_operator_search_${SLURM_JOB_ID:-local}_summary.txt"
