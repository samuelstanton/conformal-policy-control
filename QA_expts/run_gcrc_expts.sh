#!/bin/sh
#SBATCH -t 24:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=parallel
#SBATCH --gpus-per-task=0
source .venv/bin/activate
python run_gcrc_expts.py --method_names monotized_losses_crc ltt gcrc --score_names logprobs # logprobs frequency selfevals
