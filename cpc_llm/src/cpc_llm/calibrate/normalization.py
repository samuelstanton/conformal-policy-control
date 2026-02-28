import numpy as np

import logging


logger = logging.getLogger(__name__)


def importance_weighted_monte_carlo_integration(
    LRs_unconstrained_over_safe,  ## 1D numpy array
    beta_t,  ## float
    proposal: str = "safe",  ## "unconstrained"
):

    if proposal == "unconstrained":
        ## If beta_t >= 1: Assume proposal is unconstrained
        return np.mean(np.minimum(beta_t / LRs_unconstrained_over_safe, 1))

    elif proposal == "safe":
        ## Else, beta_t < 1: Assume proposal is safe
        return np.mean(np.minimum(LRs_unconstrained_over_safe, beta_t))
    else:
        raise ValueError(f"Unrecognized proposal name : {proposal}")


def iwmci_overlap_est(
    LRs_unconstrained_over_safe,  ## 1D numpy array
    unconstrained_liks,
    safe_liks,
    beta_t,  ## float
    psi_t,
    # intersection_target: str = "safe", ## The policy that want to compute intersection (w constrained policy) for
    proposal: str = "safe",
):

    if proposal not in ["safe", "unconstrained"]:
        raise ValueError(f"proposal name not recognized : {proposal}")

    constrained_density_est = np.minimum(
        safe_liks * (beta_t / psi_t), unconstrained_liks / psi_t
    )

    if proposal == "unconstrained":
        ## If beta_t >= 1: Assume proposal is unconstrained
        # constrained_density_est = np.minimum(safe_liks * (beta_t / psi_t), unconstrained_liks / psi_t)
        return np.mean(
            np.minimum(constrained_density_est, unconstrained_liks) / unconstrained_liks
        )

    elif proposal == "safe":
        return np.mean(np.minimum(constrained_density_est, safe_liks) / safe_liks)

    else:
        raise ValueError(f"Unrecognized proposal name : {proposal}")
