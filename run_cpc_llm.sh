#!/bin/bash
#SBATCH --job-name=cpc_llm
#SBATCH -A ai4dd
#SBATCH -p ai-normal
#SBATCH -q ai_normal
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-gpu=1

ALPHA=${1:-0.8}
INITIAL_SEED=${2:-0}
LAST_SEED=${3:-0}


source .venv/bin/activate

cpc-llm --config-name=cpc_llm conformal_policy_control.alpha=${ALPHA} initial_seed=${INITIAL_SEED} last_seed=${LAST_SEED}
