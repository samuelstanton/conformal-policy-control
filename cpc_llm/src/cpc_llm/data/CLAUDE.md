# data/

Dataset generation, formatting, and splitting for training and calibration.

## Key files

- **combine_and_split.py** — `train_cal_split_gen_outputs()`: splits generated samples into calibration (for CPC) and training (for policy improvement) sets. `combine_new_with_old_datasets()`: mixes current round data with historical data, controlled by `proportion_of_old_data`. `append_df_len_to_fp()`: utility to embed DataFrame length into filenames.
- **select.py** — `get_seeds_from_training_data()`: selects next-round prompts by mixing old seeds with new best-scoring sequences. Supports "best_scoring" and "uniform" sampling.
- **synthetic_dataset_formatter.py** — `find_minimal_edit_pairs()`: finds nearest-neighbor particle pairs with different scores (using PyNNDescent) for preference learning. Also has filtering helpers: `filter_by_score()`, `abs_subtract_replace_infs()`, `find_dense_pairs()`, `find_preference_pairs()`.
- **synthetic_dataset_generator.py** — `init_test_fns()`: initializes Ehrlich/RoughMtFuji test functions from the holo library. `filter_sequences()`: selects diverse subsets via ranked FFT.
- **synthetic_dataset_lib.py** — Low-level formatting: `format_instruction_tuning()`, `format_plain()`, `format_fewshot()`, `format_edit_pairs()`, `format_preference_pairs()`.

## Data structures

- Particles: JSON-serialized integer lists (discrete optimization variables)
- Scores: float evaluation values from test functions
- Preference pairs: dicts with `higher_score_particle`, `lower_score_particle`, and their scores
