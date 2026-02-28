import numpy as np

import logging
import sys

from typing import Any
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def prepare_grid(
    cfg: DictConfig,
    V: Any,  ## 1-D np array, lik-ratio values (unsorted) to process into grid
    n_grid: int = 50,  ## int, approximately how many values want to have in resulting grid
    proposal: str = "unconstrained",  ## str, 'unconstrained' or 'safe' to indicate prop dist
) -> Any:
    """Sort and coarsen grid of lik-ratio values to search over"""

    G = np.sort(
        np.unique(V)
    )  ## Want to search in increasing order for CPC (safest to most aggressive)

    ## Coarsen grid to approximately n_grid elements
    n_curr = len(G)
    k = max(int(n_curr / n_grid), 1)
    G = G[::k]

    if proposal == "unconstrained":
        ## For unconstrained, ensure also consider unconstrained policy in grid (np.inf)
        G = np.concatenate((G, [np.inf]))

    elif proposal == "safe":
        ## For safe, ensure that include minimum positive float value
        G = np.concatenate(([sys.float_info.min], G))

    elif proposal == "mixed":
        ## For mixed, ensure that include minimum positive float value as well as np.inf
        G = np.concatenate(([sys.float_info.min], G, [np.inf]))

    else:
        raise ValueError(f"unrecognized proposal name : {proposal}")

    return G
