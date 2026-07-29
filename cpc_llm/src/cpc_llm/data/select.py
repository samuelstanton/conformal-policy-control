import json
import numpy as np
import os
import pandas as pd
import torch
from omegaconf import DictConfig
from ..infrastructure.file_handler import LocalOrS3Client
from ..data_contracts import (
    CHOSEN,
    CHOSEN_SCORE,
    HIGHER_SCORE,
    HIGHER_SCORE_PARTICLE,
    LOWER_SCORE,
    LOWER_SCORE_PARTICLE,
    PROMPT,
    PROMPT_SCORE,
    SCORE,
)
from .synthetic_dataset_lib import ranked_fft


def get_seeds_from_training_data(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    prev_seeds_fp: str,
    curr_training_data_fp: str,
    output_dir: str,
    sample_size: int,
    sampling_method: str = "best_scoring",
    higher_score_particle_field: str = HIGHER_SCORE_PARTICLE,
    lower_score_particle_field: str = LOWER_SCORE_PARTICLE,
    lower_score_field: str = LOWER_SCORE,
    higher_score_field: str = HIGHER_SCORE,
    pi_optimizer_name: str = "sft",
    setting: str = "",
    random_seed: int = 0,
    first_iter: bool = False,
) -> str:
    """Select seed sequences from training data for the next CPC round.

    Mixes best-scoring or uniformly sampled sequences from the current
    training data with historical seeds (controlled by
    ``cfg.proportion_of_old_seeds``).

    Args:
        cfg: Hydra config with ``overwrite_seeds_flag`` and
            ``proportion_of_old_seeds``.
        fs: File system client (local or S3).
        prev_seeds_fp: Path to seeds from the previous iteration.
        curr_training_data_fp: Path to current round's training JSONL.
        output_dir: Directory to write the selected seeds file.
        sample_size: Total number of seeds to select.
        sampling_method: ``"best_scoring"``, ``"uniform"``, or
            ``"ranked_fft"`` (ranked farthest-first traversal, which balances
            best-scoring selection with diversity).
        higher_score_particle_field: Column name for the higher-score particle.
        lower_score_particle_field: Column name for the lower-score particle.
        lower_score_field: Column name for the lower score value.
        higher_score_field: Column name for the higher score value.
        pi_optimizer_name: Optimizer type (``"sft"``, ``"dpo"``, etc.).
        setting: Optional setting string for output filename.
        random_seed: Random seed for reproducibility.
        first_iter: Whether this is the first iteration (no historical data).

    Returns:
        Path to the output seeds JSONL file.
    """
    output_fp = os.path.join(
        output_dir, f"seeds_from_{os.path.basename(curr_training_data_fp)}"
    )

    if len(setting) > 0:
        output_fp = os.path.join(
            os.path.dirname(output_fp), f"{setting}_{os.path.basename(output_fp)}"
        )

    if not cfg.overwrite_seeds_flag and fs.exists(output_fp):
        return output_fp

    else:
        if not output_dir.startswith("s3://"):
            os.makedirs(output_dir, exist_ok=True)
        if not first_iter:
            ## If not first iteration: read, prepare, and get new sample sizes for historical data
            prev_seeds_df = pd.read_json(prev_seeds_fp, orient="records", lines=True)
            hist_sample_size = int(sample_size * cfg.proportion_of_old_seeds)
            curr_sample_size = sample_size - hist_sample_size
        else:
            curr_sample_size = sample_size

        curr_train_df = pd.read_json(
            curr_training_data_fp, orient="records", lines=True
        )

        if len(curr_train_df) == 0:
            prev_seeds_df.to_json(output_fp, orient="records", lines=True)
            return output_fp

        elif len(curr_train_df) < curr_sample_size:
            curr_sample_size = len(curr_train_df)
            hist_sample_size = sample_size - curr_sample_size

        curr_train_df = curr_train_df.loc[
            curr_train_df[lower_score_particle_field]
            .astype(str)
            .drop_duplicates()
            .index
        ]

        if sampling_method == "best_scoring":
            if not first_iter:
                prev_seeds_df = prev_seeds_df.sort_values(by=[SCORE], ascending=True)[
                    :hist_sample_size
                ]
            curr_train_df = curr_train_df.sort_values(
                by=[lower_score_field], ascending=True
            )[:curr_sample_size]

        elif sampling_method == "uniform":
            if not first_iter:
                prev_seeds_df = prev_seeds_df.sample(
                    n=min(len(prev_seeds_df), hist_sample_size),
                    random_state=random_seed,
                )
            curr_train_df = curr_train_df.sample(
                n=min(len(curr_train_df), curr_sample_size), random_state=random_seed
            )

        elif sampling_method == "ranked_fft":

            def _hamming_distance(x, y):
                return (x != y).sum().item()

            def _build_library_and_scores(df, particle_col, score_col):
                """Parse particles into a fixed-length LongTensor library.

                Mirrors the particle_for_scoring logic of
                parse_particle_and_score_permissive: JSON-decode, validate
                integer values, then pad (wrap mode) or truncate to the target
                dimension inferred from the first valid particle.

                Returns:
                    library: LongTensor of shape (n_valid, dim)
                    scores: FloatTensor of shape (n_valid,)
                    valid_positions: list of iloc positions in df that parsed
                        successfully, for mapping ranked_fft indices back to df
                """

                def _parse_raw(p):
                    if isinstance(p, str):
                        try:
                            particle = json.loads(p)
                        except (ValueError, TypeError):
                            return None
                    else:
                        particle = p
                    if not isinstance(particle, list) or len(particle) == 0:
                        return None
                    try:
                        if any(int(x) != x for x in particle):
                            return None
                        return [int(x) for x in particle]
                    except (ValueError, TypeError, OverflowError):
                        return None

                raw_parsed = [_parse_raw(p) for p in df[particle_col]]
                valid_raw = [p for p in raw_parsed if p is not None]
                if not valid_raw:
                    raise ValueError(
                        f"No parseable particles in column '{particle_col}' "
                        "for ranked_fft."
                    )
                dim = len(valid_raw[0])

                valid_positions = []
                valid_particles = []
                for i, p in enumerate(raw_parsed):
                    if p is None:
                        continue
                    if len(p) != dim:
                        p = np.pad(
                            p, (0, max(0, dim - len(p))), mode="wrap"
                        )[:dim].tolist()
                    valid_positions.append(i)
                    valid_particles.append(p)

                library = torch.LongTensor(valid_particles)
                scores = torch.FloatTensor(
                    df[score_col].iloc[valid_positions].tolist()
                )
                finite_mask = torch.isfinite(scores)
                finite_indices = finite_mask.nonzero(as_tuple=True)[0].tolist()
                library = library[finite_indices]
                scores = scores[finite_indices]
                valid_positions = [valid_positions[i] for i in finite_indices]

                # Restrict to the top-scoring fraction (lower score is better).
                top_fraction = cfg.get("ranked_fft_top_fraction", 0.1)
                n_keep = max(1, int(np.ceil(len(scores) * top_fraction)))
                top_indices = torch.argsort(scores)[:n_keep].tolist()
                library = library[top_indices]
                scores = scores[top_indices]
                valid_positions = [valid_positions[i] for i in top_indices]
                return library, scores, valid_positions

            curr_library, curr_scores, curr_valid_pos = _build_library_and_scores(
                curr_train_df, lower_score_particle_field, lower_score_field
            )
            
            curr_indices = ranked_fft(
                curr_library,
                curr_scores,
                n=curr_sample_size,
                descending=False,
                distance_fn=_hamming_distance,
            )
            curr_train_df = curr_train_df.iloc[
                [curr_valid_pos[i] for i in curr_indices.tolist()]
            ]

            if not first_iter:
                prev_library, prev_scores, prev_valid_pos = _build_library_and_scores(
                    prev_seeds_df, higher_score_particle_field, SCORE
                )
                prev_indices = ranked_fft(
                    prev_library,
                    prev_scores,
                    n=hist_sample_size,
                    descending=False,
                    distance_fn=_hamming_distance,
                )
                prev_seeds_df = prev_seeds_df.iloc[
                    [prev_valid_pos[i] for i in prev_indices.tolist()]
                ]

        else:
            raise ValueError(f"Unknown sampling method '{sampling_method}.'")

        ## Reformat seeds selected from current training data
        curr_train_df_selected = curr_train_df[
            [lower_score_particle_field, lower_score_field]
        ]
        curr_train_df_selected = curr_train_df_selected.rename(
            columns={
                lower_score_particle_field: higher_score_particle_field,
                lower_score_field: SCORE,
            }
        )

        if pi_optimizer_name == "dpo":
            curr_train_df_selected = curr_train_df_selected.rename(
                columns={
                    higher_score_particle_field: PROMPT,
                    lower_score_particle_field: CHOSEN,
                    higher_score_field: PROMPT_SCORE,
                    lower_score_field: CHOSEN_SCORE,
                }
            )

        if not first_iter:
            train_df_selected = pd.concat([prev_seeds_df, curr_train_df_selected])
        else:
            train_df_selected = curr_train_df_selected

        train_df_selected.to_json(output_fp, orient="records", lines=True)

        return output_fp
