# infer/

Sequence generation and sampling under different policy parameterizations (safe, unconstrained, CPC-constrained).

## Key files

- **rejection_sampling.py** — `generate_proposals_for_AR_sampling()`: generates proposals from a specified policy, computes likelihoods under all models, constrains likelihoods per CPC bounds, and performs acceptance-rejection or independent Metropolis-Hastings sampling. `accept_reject_sample_and_get_likelihoods()`: wrapper that handles the full AR pipeline including likelihood computation.
- **iterative_generation2.py** — `run_iterative_generation()`: multi-temperature sequential generation. Loads a ModelClient, generates sequences at varying temperatures for diversity, parses and scores outputs. Handles checkpoint resumption.
- **generation_utils.py** — `get_temperatures()`: adaptively scales generation temperature based on previous iteration's Hamming distance to ensure sufficient exploration.

## Sampling algorithms

The rejection sampling supports two modes:
- **Accept-reject**: accepts proposal x with probability min(1, pi_constrained(x) / (M * pi_proposal(x)))
- **Independent Metropolis-Hastings**: keeps previous sample if proposal is rejected, with acceptance ratio based on target/proposal likelihood ratios
