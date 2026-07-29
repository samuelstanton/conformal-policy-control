#!/bin/bash

# ALPHAS accepts one or more alpha values, space- and/or comma-separated, e.g.:
#   bash submit_cpc_llm_expts.sh 0.8 0 29
#   bash submit_cpc_llm_expts.sh "0.4,0.6,0.8,1.0" 0 29
#   bash submit_cpc_llm_expts.sh "0.4 0.6 0.8 1.0" 0 29
ALPHAS_RAW=${1:-"0.4,0.6,0.8,1.0"}
INITIAL_SEED=${2:-0}
LAST_SEED=${3:-29}

IFS=', ' read -r -a ALPHAS <<< "${ALPHAS_RAW}"

for (( seed=INITIAL_SEED; seed<=LAST_SEED; seed++ )); do
    for ALPHA in "${ALPHAS[@]}"; do
        sbatch --job-name="a${ALPHA}_s${seed}_cpc_llm" run_cpc_llm.sh ${ALPHA} ${seed} ${seed}
        sleep 0.1
    done
done
