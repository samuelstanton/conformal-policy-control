import heapq
import json
import logging
import numpy as np
import random
import torch

from holo.test_functions.closed_form import Ehrlich
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..data_contracts import CHOSEN, INPUT, PARTICLE, REJECTED, SCORE, TARGET


def format_instruction_tuning(
    X: torch.Tensor,
    scores: torch.Tensor,
    test_fn: Ehrlich,
    include_constraints: bool = True,
) -> List[Dict[str, str]]:
    assert scores.shape == (X.shape[0], 1)
    output_examples = []
    test_fn_str = repr(test_fn)
    if include_constraints:
        constraints_str = f"{test_fn_str}\n"
    else:
        constraints_str = ""
    for x, score in zip(X, scores):
        ex_dict = {
            INPUT: f"{constraints_str}Score: {score.item():.2f}\nSolution:\n",
            TARGET: f"{x.tolist()}",
        }
        output_examples.append(ex_dict)
    return output_examples


def format_plain(input_dicts: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Formats particles as JSON lists. No further formatting.
    """
    output_examples = []
    for d in input_dicts:
        ex_dict = {k: v for k, v in d.items() if k not in [PARTICLE, SCORE]}
        if isinstance(d[PARTICLE], torch.Tensor):
            ex_dict[PARTICLE] = f"{json.dumps(d[PARTICLE].tolist())}"
        elif isinstance(d[PARTICLE], Iterable):
            ex_dict[PARTICLE] = f"{json.dumps(list(d[PARTICLE]))}"
        else:
            raise ValueError(f"Unrecognized particle type: {type(d[PARTICLE])}")
        if isinstance(d[SCORE], torch.Tensor):
            ex_dict[SCORE] = f"{d[SCORE].item():.3f}"
        elif isinstance(d[SCORE], float):
            ex_dict[SCORE] = f"{d[SCORE]:.3f}"
        else:
            raise ValueError(f"Unrecognized score type: {type(d[SCORE])}")
        output_examples.append(ex_dict)
    return output_examples


def format_fewshot(
    X: torch.Tensor,
    scores: torch.Tensor,
    ks: List[
        int
    ],  # a list of the options for no. of few-shot examples to include in each prompt
    allow_same_score: bool = True,  # Whether the target score can be the same as the score of a few-shot example
    seed: int = 0,
) -> List[Dict[str, str]]:
    random.seed(seed)
    output_examples = []
    for i in range(X.shape[0]):
        particle = X[i]
        score = scores[i]
        k = random.sample(ks, k=1)[0]  # num. of few-shot examples to use
        # get valid indexes for selecting few-shot examples from -- not including current example and particles with the same score
        if not allow_same_score:
            valid_idxs = torch.nonzero(scores != score).squeeze(-1).tolist()
            if i in valid_idxs:
                valid_idxs.remove(i)
        else:
            valid_idxs = [j for j in range(len(scores)) if j != i]
        if len(valid_idxs) < k:
            logging.error(
                f"Not enough examples to select {k} in-context examples for. Skipping to next example..."
            )
            continue
        few_shot_examples_str = ""
        if k > 0:
            few_shot_example_idxs = random.sample(valid_idxs, k=k)
            few_shot_examples_X = X[few_shot_example_idxs]
            few_shot_examples_scores = scores[few_shot_example_idxs]
            few_shot_examples_str += "\n\n".join(
                [
                    f"Score: {ex_score.item():.3f}\nParticle: {json.dumps([int(x) for x in ex_particle.tolist()])}"
                    for ex_particle, ex_score in zip(
                        few_shot_examples_X, few_shot_examples_scores
                    )
                ]
            )
            few_shot_examples_str += "\n\n"
        input_str = f"{few_shot_examples_str}Score: {score.item():.3f}\nParticle: "
        output_examples.append(
            {
                INPUT: input_str,
                TARGET: f"{json.dumps([int(x) for x in particle.tolist()])}",
            }
        )
    return output_examples


EDIT_INSTRUCTION = """
Given the following constraints, score (lower score is better), and original input, 
edit the original input to decrease the score.
"""


def format_edit_pairs(
    X: torch.Tensor,
    X_scores: torch.Tensor,
    X_prime: torch.Tensor,
    X_prime_scores: torch.Tensor,
    test_fn: Ehrlich,
) -> List[Dict[str, str]]:
    assert X.shape == X_prime.shape
    assert X_scores.shape == (X.shape[0], 1)
    output_examples = []
    test_fn_str = repr(test_fn)

    for x, x_score, x_prime, x_prime_score in zip(X, X_scores, X_prime, X_prime_scores):
        # choose which one is the original version and which is the edited (improved) version
        if x_score.item() < x_prime_score.item():
            orig = x
            edited = x_prime
            orig_score = x_score
        else:
            orig = x_prime
            edited = x
            orig_score = x_prime_score
        ex_dict = {
            INPUT: f"{EDIT_INSTRUCTION}\nConstraints:\n{test_fn_str}\n"
            + f"Original input:\n{orig.tolist()}\nScore: {orig_score.item():.2f}\nEdited:\n",
            TARGET: f"{edited.tolist()}",
        }
        output_examples.append(ex_dict)
    return output_examples


def format_preference_pairs(
    X: torch.Tensor,
    X_scores: torch.Tensor,
    X_prime: torch.Tensor,
    X_prime_scores: torch.Tensor,
    test_fn: Ehrlich,
) -> List[Dict[str, str]]:
    """Format pairs from X and X_prime into preference-tuning dataset."""
    output_examples = []
    test_fn_str = repr(test_fn)
    for x, x_score, x_prime, x_prime_score in zip(X, X_scores, X_prime, X_prime_scores):
        if x_score.item() < x_prime_score.item():
            y_pair = (x, x_prime)
            y_scores = (x_score, x_prime_score)
        else:
            y_pair = (x_prime, x)
            y_scores = (x_prime_score, x_score)
        ex_dict = {
            INPUT: f"Constraints:\n{test_fn_str}\nSolution:\n",
            CHOSEN: y_pair[0].tolist(),
            "chosen_score": y_scores[0].item(),
            REJECTED: y_pair[1].tolist(),
            "rejected_score": y_scores[1].item(),
        }
        output_examples.append(ex_dict)
    return output_examples


def ranked_fft(
    library: torch.Tensor,
    ranking_scores: Optional[torch.Tensor] = None,
    n: int = 2,
    descending: bool = False,
    distance_fn: Callable = None,
) -> torch.Tensor:
    """
    Farthest-first traversal of a library of strings.
    If `ranking_scores` is provided, the scores are used to pick the starting point and break ties.
    Args:
        library: A tensor of shape (N,d) where N is the number of sequences and d is the dimension of each sequence.
        ranking_scores: A tensor of shape (N,) containing the ranking scores of the sequences in the library.
        n: The number of sequences to return.
        descending: If True, then higher ranking score is better. If False, then lower ranking score is better.
        distance_fn: The distance function to use. If None, then uses Euclidean norm.
    Returns:
        A tensor of shape (n,) containing the indices of the selected sequences.
    """
    if ranking_scores is None:
        ranking_scores = torch.zeros(library.shape[0])
        remaining_indices = list(range(library.shape[0]))
    else:
        if descending:
            ranking_scores = -ranking_scores
        remaining_indices = list(np.argsort(ranking_scores))

    library_len = library.shape[0]
    if n > library_len:
        logging.warning(
            f"n={n} is greater than library length ({library_len}). Returning original dataset indices."
        )
        return torch.LongTensor(range(library.shape[0]))

    selected = [remaining_indices.pop(0)]

    if n == 1:
        return torch.tensor(selected)

    if distance_fn is None:

        def distance_fn(x, y):
            return torch.norm(x - y, p=2).item()

    pq = []
    # First pass through library
    for index in tqdm(remaining_indices, desc="First pass dataset sketching"):
        # Pushing with heapq, negate dist to simulate max-heap with min-heap
        (
            heapq.heappush(
                pq,
                (
                    -distance_fn(library[index], library[selected[0]]),
                    ranking_scores[index],
                    index,
                    1,
                ),
            ),
        )

    for _ in tqdm(range(1, n), desc="Dataset sketching"):
        while True:
            try:
                neg_dist, score, idx, num_checked = heapq.heappop(pq)
            except IndexError as e:
                print(f"Error while popping: {e}")
                print(f"Len selected: {len(selected)}")
                print(f"n: {n}")
                print(f"num_checked: {num_checked}")
                raise e

            # Check if the top of the heap has been checked against all currently selected sequences
            if num_checked < len(selected):
                min_dist = min(
                    distance_fn(library[idx], library[selected[i]])
                    for i in range(num_checked, len(selected))
                )
                min_dist = min(min_dist, -neg_dist)
                heapq.heappush(pq, (-min_dist, score, idx, len(selected)))
            else:
                selected.append(idx)
                break

    return torch.tensor(selected)
