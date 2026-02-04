#!/bin/sh
#SBATCH -t 24:00:00 
#SBATCH --mem=16G
source .venv/bin/activate
python run_gcrc_expts.py --method_names monotized_losses_crc ltt gcrc --score_names logprobs # logprobs frequency selfevals
