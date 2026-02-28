import os
import pandas as pd
from omegaconf import DictConfig
from ..infrastructure.file_handler import LocalOrS3Client


def get_seeds_from_training_data(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    prev_seeds_fp: str,
    curr_training_data_fp: str,
    output_dir: str,
    sample_size: int,
    sampling_method: str = "best_scoring",
    higher_score_particle_field: str = "higher_score_particle",
    lower_score_particle_field: str = "lower_score_particle",
    lower_score_field: str = "lower_score",
    higher_score_field: str = "higher_score",
    pi_optimizer_name: str = "sft",
    setting: str = "",
    random_seed: int = 0,
    first_iter: bool = False,
) -> str:

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
        if not first_iter:
            ## If not first iteration: read, prepare, and get new sample sizes for historical data
            prev_seeds_df = pd.read_json(prev_seeds_fp, orient="records", lines=True)
            # hist_train_df = hist_train_df.loc[hist_train_df[lower_score_particle_field].astype(str).drop_duplicates().index]
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
                prev_seeds_df = prev_seeds_df.sort_values(by=["score"], ascending=True)[
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

        else:
            raise ValueError(f"Unknown sampling method '{sampling_method}.'")

        ## Reformat seeds selected from current training data
        curr_train_df_selected = curr_train_df[
            [lower_score_particle_field, lower_score_field]
        ]
        curr_train_df_selected = curr_train_df_selected.rename(
            columns={
                lower_score_particle_field: higher_score_particle_field,
                lower_score_field: "score",
            }
        )

        if pi_optimizer_name == "dpo":
            curr_train_df_selected = curr_train_df_selected.rename(
                columns={
                    higher_score_particle_field: "prompt",
                    lower_score_particle_field: "chosen",
                    higher_score_field: "prompt_score",
                    lower_score_field: "chosen_score",
                }
            )
            # if not first_iter:
            #     prev_seeds_df = prev_seeds_df.rename(columns={higher_score_particle_field : 'prompt', lower_score_particle_field: 'chosen', higher_score_field : 'prompt_score', lower_score_field: 'chosen_score'})

        if not first_iter:
            train_df_selected = pd.concat([prev_seeds_df, curr_train_df_selected])
        else:
            train_df_selected = curr_train_df_selected

        train_df_selected.to_json(output_fp, orient="records", lines=True)

        return output_fp
