import pandas as pd

from omegaconf import OmegaConf
from cpc_llm.data.synthetic_dataset_formatter import (
    find_dense_pairs,
    find_preference_pairs,
    filter_infeasible_examples,
)


class TestFindDensePairs:
    def test_x_threshold(self):
        cfg = OmegaConf.create(
            {
                "score_lower_threshold": -0.5,
                "n": None,
                "n_neighbors": 3,
                "distance_metric": "hamming",
                "dist_x_threshold": 2 / 7,
                "dist_y_threshold": None,
                "max_proportion_infeasible": None,
            }
        )

        library = [
            [1, 2, 3, 4, 5, 6, 7],
            [1, 3, 3, 4, 5, 6, 7],
            [1, 3, 3, 4, 7, 7, 8],
            [5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 7, 5],
        ]
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        df = pd.DataFrame({"particle": library, "score": scores})

        outputs = find_dense_pairs(cfg, df)
        expected_outputs = [
            {
                "lower_score_particle": [1, 2, 3, 4, 5, 6, 7],
                "lower_score": "0.100",
                "higher_score_particle": [1, 3, 3, 4, 5, 6, 7],
                "higher_score": "0.200",
            },
            {
                "lower_score_particle": [5, 5, 5, 5, 5, 5, 5],
                "lower_score": "0.400",
                "higher_score_particle": [5, 5, 5, 5, 5, 7, 5],
                "higher_score": "0.500",
            },
        ]
        assert outputs == expected_outputs

    def test_xy_thresholds(self):
        cfg = OmegaConf.create(
            {
                "score_lower_threshold": -0.5,
                "n": None,
                "n_neighbors": 3,
                "distance_metric": "hamming",
                "dist_x_threshold": 2 / 7,
                "dist_y_threshold": 0.1,
                "max_proportion_infeasible": None,
            }
        )

        library = [
            [1, 2, 3, 4, 5, 6, 7],
            [1, 3, 3, 4, 5, 6, 7],
            [1, 3, 3, 4, 7, 7, 8],
            [5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 7, 5],
        ]
        scores = [0.1, 0.2, 0.3, 0.4, 0.55]
        df = pd.DataFrame({"particle": library, "score": scores})

        outputs = find_dense_pairs(cfg, df)
        expected_outputs = [
            {
                "lower_score_particle": [1, 2, 3, 4, 5, 6, 7],
                "lower_score": "0.100",
                "higher_score_particle": [1, 3, 3, 4, 5, 6, 7],
                "higher_score": "0.200",
            },
        ]
        assert outputs == expected_outputs

    def test_lower_score_threshold(self):
        cfg = OmegaConf.create(
            {
                "score_lower_threshold": 0.1,
                "n": None,
                "n_neighbors": 3,
                "distance_metric": "hamming",
                "dist_x_threshold": 2 / 7,
                "dist_y_threshold": None,
                "max_proportion_infeasible": None,
            }
        )

        library = [
            [1, 2, 3, 4, 5, 6, 7],
            [1, 3, 3, 4, 5, 6, 7],
            [1, 3, 3, 4, 7, 7, 8],
            [5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 7, 5],
        ]
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        df = pd.DataFrame({"particle": library, "score": scores})

        outputs = find_dense_pairs(cfg, df)
        expected_outputs = [
            {
                "lower_score_particle": [5, 5, 5, 5, 5, 5, 5],
                "lower_score": "0.400",
                "higher_score_particle": [5, 5, 5, 5, 5, 7, 5],
                "higher_score": "0.500",
            },
        ]
        assert outputs == expected_outputs


class TestFilterInfeasibleExamples:
    def test_downsample_infeasible(self):
        ds = [
            {"higher_score": "inf"},
            {"higher_score": "inf"},
            {"higher_score": "inf"},
            {"higher_score": "inf"},
            {"higher_score": "inf"},
            {"higher_score": "0.25"},
            {"higher_score": "0.25"},
        ]
        cfg = OmegaConf.create(
            {
                "seed": 0,
                "max_proportion_infeasible": 2 / 3,
            }
        )
        out_ds = filter_infeasible_examples(cfg, ds)
        out_num_infs = len([d for d in out_ds if d["higher_score"] == "inf"])
        assert out_num_infs == 4

    def test_no_filter_needed(self):
        ds = [
            {"higher_score": "inf"},
            {"higher_score": "0.25"},
            {"higher_score": "0.25"},
        ]
        cfg = OmegaConf.create(
            {
                "seed": 0,
                "max_proportion_infeasible": 2 / 3,
            }
        )
        out_ds = filter_infeasible_examples(cfg, ds)
        out_num_infs = len([d for d in out_ds if d["higher_score"] == "inf"])
        assert out_num_infs == 1


class TestFindPreferencePairs:
    def test_basic(self):
        cfg = OmegaConf.create(
            {
                "score_lower_threshold": -0.5,
                "n": None,
                "n_neighbors": 4,
                "distance_metric": "hamming",
                "dist_x_threshold": 1.0,
                "max_proportion_infeasible": None,
                "seed": 0,
            }
        )

        library = [
            [1, 2, 3, 4, 5, 6, 7],
            [1, 3, 3, 4, 5, 6, 7],
            [1, 3, 3, 4, 7, 7, 8],
            [5, 5, 5, 5, 5, 5, 5],
        ]
        scores = [0.1, 0.2, 0.2, 0.4]
        df = pd.DataFrame({"particle": library, "score": scores})
        outputs = find_preference_pairs(cfg, df)
        expected_outputs = [
            {
                "prompt": [1, 3, 3, 4, 5, 6, 7],
                "prompt_score": "0.200",
                "chosen": [1, 2, 3, 4, 5, 6, 7],
                "chosen_score": "0.100",
                "rejected": [1, 3, 3, 4, 5, 6, 7],
                "rejected_score": "0.200",
            },
            {
                "prompt": [1, 3, 3, 4, 5, 6, 7],
                "prompt_score": "0.200",
                "chosen": [1, 2, 3, 4, 5, 6, 7],
                "chosen_score": "0.100",
                "rejected": [1, 3, 3, 4, 7, 7, 8],
                "rejected_score": "0.200",
            },
            {
                "prompt": [1, 3, 3, 4, 5, 6, 7],
                "prompt_score": "0.200",
                "chosen": [1, 2, 3, 4, 5, 6, 7],
                "chosen_score": "0.100",
                "rejected": [5, 5, 5, 5, 5, 5, 5],
                "rejected_score": "0.400",
            },
            {
                "prompt": [1, 3, 3, 4, 7, 7, 8],
                "prompt_score": "0.200",
                "chosen": [1, 2, 3, 4, 5, 6, 7],
                "chosen_score": "0.100",
                "rejected": [1, 3, 3, 4, 7, 7, 8],
                "rejected_score": "0.200",
            },
            {
                "prompt": [1, 3, 3, 4, 7, 7, 8],
                "prompt_score": "0.200",
                "chosen": [1, 2, 3, 4, 5, 6, 7],
                "chosen_score": "0.100",
                "rejected": [1, 3, 3, 4, 5, 6, 7],
                "rejected_score": "0.200",
            },
            {
                "prompt": [1, 3, 3, 4, 7, 7, 8],
                "prompt_score": "0.200",
                "chosen": [1, 2, 3, 4, 5, 6, 7],
                "chosen_score": "0.100",
                "rejected": [5, 5, 5, 5, 5, 5, 5],
                "rejected_score": "0.400",
            },
            {
                "prompt": [5, 5, 5, 5, 5, 5, 5],
                "prompt_score": "0.400",
                "chosen": [1, 2, 3, 4, 5, 6, 7],
                "chosen_score": "0.100",
                "rejected": [5, 5, 5, 5, 5, 5, 5],
                "rejected_score": "0.400",
            },
            {
                "prompt": [5, 5, 5, 5, 5, 5, 5],
                "prompt_score": "0.400",
                "chosen": [1, 3, 3, 4, 5, 6, 7],
                "chosen_score": "0.200",
                "rejected": [5, 5, 5, 5, 5, 5, 5],
                "rejected_score": "0.400",
            },
            {
                "prompt": [5, 5, 5, 5, 5, 5, 5],
                "prompt_score": "0.400",
                "chosen": [1, 3, 3, 4, 7, 7, 8],
                "chosen_score": "0.200",
                "rejected": [5, 5, 5, 5, 5, 5, 5],
                "rejected_score": "0.400",
            },
        ]
        assert len(outputs) == len(expected_outputs)
        for o in expected_outputs:
            assert o in outputs, f"Cannot find {o} in outputs:\n{outputs}"
