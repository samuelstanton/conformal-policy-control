#!/bin/sh
#SBATCH -t 24:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=parallel
#SBATCH --gpus-per-task=0
source ../.venv/bin/activate
%load_ext autoreload
%autoreload 2
%matplotlib inline
python run_gcrc_expts.py
