import logging

import pandas as pd

from omegaconf import OmegaConf
from cpc_llm.data.synthetic_dataset_formatter import (
    find_dense_pairs,
    find_preference_pairs,
    filter_infeasible_examples,
)
from cpc_llm.data_contracts import (
    CHOSEN,
    CHOSEN_SCORE,
    HIGHER_SCORE,
    HIGHER_SCORE_PARTICLE,
    LOWER_SCORE,
    LOWER_SCORE_PARTICLE,
    PARTICLE,
    PROMPT,
    PROMPT_SCORE,
    REJECTED,
    REJECTED_SCORE,
    SCORE,
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
        df = pd.DataFrame({PARTICLE: library, SCORE: scores})

        outputs = find_dense_pairs(cfg, df)
        expected_outputs = [
            {
                LOWER_SCORE_PARTICLE: [1, 2, 3, 4, 5, 6, 7],
                LOWER_SCORE: "0.100",
                HIGHER_SCORE_PARTICLE: [1, 3, 3, 4, 5, 6, 7],
                HIGHER_SCORE: "0.200",
            },
            {
                LOWER_SCORE_PARTICLE: [5, 5, 5, 5, 5, 5, 5],
                LOWER_SCORE: "0.400",
                HIGHER_SCORE_PARTICLE: [5, 5, 5, 5, 5, 7, 5],
                HIGHER_SCORE: "0.500",
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
        df = pd.DataFrame({PARTICLE: library, SCORE: scores})

        outputs = find_dense_pairs(cfg, df)
        expected_outputs = [
            {
                LOWER_SCORE_PARTICLE: [1, 2, 3, 4, 5, 6, 7],
                LOWER_SCORE: "0.100",
                HIGHER_SCORE_PARTICLE: [1, 3, 3, 4, 5, 6, 7],
                HIGHER_SCORE: "0.200",
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
        df = pd.DataFrame({PARTICLE: library, SCORE: scores})

        outputs = find_dense_pairs(cfg, df)
        expected_outputs = [
            {
                LOWER_SCORE_PARTICLE: [5, 5, 5, 5, 5, 5, 5],
                LOWER_SCORE: "0.400",
                HIGHER_SCORE_PARTICLE: [5, 5, 5, 5, 5, 7, 5],
                HIGHER_SCORE: "0.500",
            },
        ]
        assert outputs == expected_outputs


class TestFilterInfeasibleExamples:
    def test_downsample_infeasible(self):
        ds = [
            {HIGHER_SCORE: "inf"},
            {HIGHER_SCORE: "inf"},
            {HIGHER_SCORE: "inf"},
            {HIGHER_SCORE: "inf"},
            {HIGHER_SCORE: "inf"},
            {HIGHER_SCORE: "0.25"},
            {HIGHER_SCORE: "0.25"},
        ]
        cfg = OmegaConf.create(
            {
                "seed": 0,
                "max_proportion_infeasible": 2 / 3,
            }
        )
        out_ds = filter_infeasible_examples(cfg, ds)
        out_num_infs = len([d for d in out_ds if d[HIGHER_SCORE] == "inf"])
        assert out_num_infs == 4

    def test_no_filter_needed(self):
        ds = [
            {HIGHER_SCORE: "inf"},
            {HIGHER_SCORE: "0.25"},
            {HIGHER_SCORE: "0.25"},
        ]
        cfg = OmegaConf.create(
            {
                "seed": 0,
                "max_proportion_infeasible": 2 / 3,
            }
        )
        out_ds = filter_infeasible_examples(cfg, ds)
        out_num_infs = len([d for d in out_ds if d[HIGHER_SCORE] == "inf"])
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
        df = pd.DataFrame({PARTICLE: library, SCORE: scores})
        outputs = find_preference_pairs(cfg, df)
        expected_outputs = [
            {
                PROMPT: [1, 3, 3, 4, 5, 6, 7],
                PROMPT_SCORE: "0.200",
                CHOSEN: [1, 2, 3, 4, 5, 6, 7],
                CHOSEN_SCORE: "0.100",
                REJECTED: [1, 3, 3, 4, 5, 6, 7],
                REJECTED_SCORE: "0.200",
            },
            {
                PROMPT: [1, 3, 3, 4, 5, 6, 7],
                PROMPT_SCORE: "0.200",
                CHOSEN: [1, 2, 3, 4, 5, 6, 7],
                CHOSEN_SCORE: "0.100",
                REJECTED: [1, 3, 3, 4, 7, 7, 8],
                REJECTED_SCORE: "0.200",
            },
            {
                PROMPT: [1, 3, 3, 4, 5, 6, 7],
                PROMPT_SCORE: "0.200",
                CHOSEN: [1, 2, 3, 4, 5, 6, 7],
                CHOSEN_SCORE: "0.100",
                REJECTED: [5, 5, 5, 5, 5, 5, 5],
                REJECTED_SCORE: "0.400",
            },
            {
                PROMPT: [1, 3, 3, 4, 7, 7, 8],
                PROMPT_SCORE: "0.200",
                CHOSEN: [1, 2, 3, 4, 5, 6, 7],
                CHOSEN_SCORE: "0.100",
                REJECTED: [1, 3, 3, 4, 7, 7, 8],
                REJECTED_SCORE: "0.200",
            },
            {
                PROMPT: [1, 3, 3, 4, 7, 7, 8],
                PROMPT_SCORE: "0.200",
                CHOSEN: [1, 2, 3, 4, 5, 6, 7],
                CHOSEN_SCORE: "0.100",
                REJECTED: [1, 3, 3, 4, 5, 6, 7],
                REJECTED_SCORE: "0.200",
            },
            {
                PROMPT: [1, 3, 3, 4, 7, 7, 8],
                PROMPT_SCORE: "0.200",
                CHOSEN: [1, 2, 3, 4, 5, 6, 7],
                CHOSEN_SCORE: "0.100",
                REJECTED: [5, 5, 5, 5, 5, 5, 5],
                REJECTED_SCORE: "0.400",
            },
            {
                PROMPT: [5, 5, 5, 5, 5, 5, 5],
                PROMPT_SCORE: "0.400",
                CHOSEN: [1, 2, 3, 4, 5, 6, 7],
                CHOSEN_SCORE: "0.100",
                REJECTED: [5, 5, 5, 5, 5, 5, 5],
                REJECTED_SCORE: "0.400",
            },
            {
                PROMPT: [5, 5, 5, 5, 5, 5, 5],
                PROMPT_SCORE: "0.400",
                CHOSEN: [1, 3, 3, 4, 5, 6, 7],
                CHOSEN_SCORE: "0.200",
                REJECTED: [5, 5, 5, 5, 5, 5, 5],
                REJECTED_SCORE: "0.400",
            },
            {
                PROMPT: [5, 5, 5, 5, 5, 5, 5],
                PROMPT_SCORE: "0.400",
                CHOSEN: [1, 3, 3, 4, 7, 7, 8],
                CHOSEN_SCORE: "0.200",
                REJECTED: [5, 5, 5, 5, 5, 5, 5],
                REJECTED_SCORE: "0.400",
            },
        ]
        assert len(outputs) == len(expected_outputs)
        for o in expected_outputs:
            assert o in outputs, f"Cannot find {o} in outputs:\n{outputs}"


class TestFindPreferencePairsNeighborhood:
    """Pairing is gated by the approximate neighbor search, not just distance.

    A strictly-better particle that is an outlier in edit space never enters any
    other particle's ``n_neighbors`` list, so no (prompt, chosen, rejected)
    triple can form even with a permissive ``dist_x_threshold``. This is what
    empties the preference dataset at smoke-test scale, where only a handful of
    generations are feasible.
    """

    @staticmethod
    def _df():
        library = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 1, 2, 2, 2],
            [9, 9, 9, 9, 9, 9, 9, 9],  # only better score, far from everything
        ]
        scores = [0.2, 0.2, 0.2, 0.2, 0.1]
        return pd.DataFrame({PARTICLE: library, SCORE: scores})

    @staticmethod
    def _cfg(n_neighbors: int, dist_x_threshold: float):
        return OmegaConf.create(
            {
                "score_lower_threshold": None,
                "n": None,
                "n_neighbors": n_neighbors,
                "distance_metric": "hamming",
                "dist_x_threshold": dist_x_threshold,
                "max_proportion_infeasible": None,
                "seed": 0,
            }
        )

    def test_outlier_best_particle_yields_no_pairs(self, caplog):
        with caplog.at_level(logging.WARNING):
            outputs = find_preference_pairs(self._cfg(3, 1.0), self._df())
        assert outputs == []
        assert "No preference pairs found" in caplog.text

    def test_full_neighborhood_recovers_pairs(self):
        outputs = find_preference_pairs(self._cfg(5, 1.0), self._df())
        assert len(outputs) > 0
        assert all(o[CHOSEN_SCORE] == "0.100" for o in outputs)

    def test_distance_threshold_still_applies(self):
        # Neighbor list covers everything, but the outlier is further away than
        # the threshold allows, so it is still excluded.
        assert find_preference_pairs(self._cfg(5, 0.5), self._df()) == []
