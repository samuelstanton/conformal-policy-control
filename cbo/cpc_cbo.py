#!/usr/bin/env python3
"""Conformal Policy Control (CPC) + Constrained BO on the Townsend problem.

Runs multi-trial constrained Bayesian optimization experiments 
(CPC+CBO vs. CBO-without-CPC vs. random baselines) 
and saves averaged objective / constraint-violation plots.

Hyperparameters such as ``n_trials``, ``n_iterations`` and ``temperature`` are
exposed as command-line arguments. Example:

    python cpc_cbo.py --n-trials 20 --temperature 0.1 --alphas 1.0 0.4 --n-iterations 10 --output-dir results

(Note: In the above command, including "--alphas 1.0" is the CBO baseline without CPC.)
Some of this code was based on the following notebook: https://botorch.org/docs/v0.16.1/notebooks_community/clf_constrained_bo
"""

import argparse
import csv
import os
import sys
from functools import partial

import matplotlib

matplotlib.use("Agg")  # headless-friendly; plots are saved to disk, not shown
import matplotlib.pyplot as plt
import numpy as np
import torch

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.exceptions.errors import ModelFittingError
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Global configuration.
#
# These module-level names are read inside the functions below (mirroring the
# original notebook). ``main()`` overrides them from the parsed CLI arguments
# before running any experiments.
# --------------------------------------------------------------------------- #
torch.set_default_dtype(torch.float64)
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

GAUSSIAN_STD_DIV_FACTOR = 6.5
GAUSSIAN_MEAN = torch.tensor([0, -1.0], **tkwargs)
NUM_CANDIDATE_SAMPLES_CPC_TEST_WEIGHT = 1000 # Num (unlabeled) samples used to estimate test pt weight in cpc search
NUM_CANDIDATE_SAMPLES = 5000 # Num samples used for estimating norm constant, pool for optimized policy, etc
MAX_ITERATIONS = 1000
TEMPERATURE = 0.1
BATCH_SIZE = 4
NUM_RESTARTS = 10
RAW_SAMPLES = 512
CPC_ALPHA = 1.0


# --------------------------------------------------------------------------- #
# Problem definition.
# --------------------------------------------------------------------------- #
class Townsend:
    def __init__(self):
        self.dim = 2
        # Domain box from the canonical Chebfun definition of the Townsend
        # (constrained) problem: chebfun2(@(x,y) ..., [-3 3 -3 3]), i.e. both
        # coordinates range over [-3, 3].
        # (See https://www.chebfun.org/examples/opt/ConstrainedOptimization.html)
        self.lower = torch.tensor([-3.0, -3.0], **tkwargs)
        self.upper = torch.tensor([3.0, 3.0], **tkwargs)
        # botorch normalize/unnormalize expect bounds of shape (2, dim) with
        # row 0 = per-dim lower and row 1 = per-dim upper.
        self.bounds = torch.stack([self.lower, self.upper])
        self._optimal_value = 2.024 #1.660
        self.name = "Townsend"

    def __call__(self, x):
        return self.objective(x)

    def is_feasible(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        t = torch.atan2(x1, x2)
        c = ((2 * torch.cos(t) - 0.5 * torch.cos(2 * t) - 0.25 * torch.cos(3 * t) - 0.125 * torch.cos(4 * t)) ** 2 + (2 * torch.sin(t)) ** 2 - x1 ** 2 - x2 ** 2)
        y_con = (c > 0).float()  # binarize the feasibility
        return y_con

    def objective(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        return torch.cos((x1 - 0.1) * x2) ** 2 + x1 * torch.sin(3 * x1 + x2)


# Instantiated in main() (and here as a default) so functions can reference it.
townsend = Townsend()


# --------------------------------------------------------------------------- #
# Data generation.
# --------------------------------------------------------------------------- #
def generate_initial_training_data(n_train, n_cal=None):
    # generate training data within the problem bounds
    train_x = draw_sobol_samples(bounds=townsend.bounds, n=n_train, q=1).squeeze(1)
    train_obj = townsend(train_x).unsqueeze(-1)
    train_con = townsend.is_feasible(train_x)
    
    return train_x, train_obj, train_con


def generate_initial_calibration_data(n_cal):
    # Draw the initial data from the (safer) Gaussian safe policy rather than
    # quasi-uniform Sobol points, so every method starts from a lower-risk dataset.
    cal_x, cal_obj, cal_con = generate_gaussian_samples(n=n_cal, mean=GAUSSIAN_MEAN)

    return cal_x, cal_obj, cal_con


# Gaussian Random Sampling Baseline (Safe Policy)
def generate_gaussian_samples(n, mean=GAUSSIAN_MEAN, std=None):
    """Generate random samples from a Gaussian distribution."""
    # Default mean: center of the search space
    if mean is None:
        mean = (townsend.bounds[0] + townsend.bounds[1]) / 2

    # Default std: range / GAUSSIAN_STD_DIV_FACTOR
    if std is None:
        range_per_dim = townsend.bounds[1] - townsend.bounds[0]
        std = range_per_dim / GAUSSIAN_STD_DIV_FACTOR

    # Sample from Gaussian
    samples = torch.randn(n, townsend.dim, **tkwargs) * std + mean

    # Clip to bounds to ensure validity
    train_x = torch.max(torch.min(samples, townsend.bounds[1]), townsend.bounds[0])

    # Evaluate objective and feasibility
    train_obj = townsend(train_x).unsqueeze(-1)
    train_con = townsend.is_feasible(train_x)

    return train_x, train_obj, train_con


# Gaussian PDF (for Safe Policy)
def gaussian_pdf(x, mean=GAUSSIAN_MEAN, std=None):
    """Evaluate the multivariate (diagonal) Gaussian PDF at point(s) x."""
    # Handle single point vs batch
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if mean is None:
        mean = (townsend.bounds[0] + townsend.bounds[1]) / 2

    if std is None:
        range_per_dim = townsend.bounds[1] - townsend.bounds[0]
        std = range_per_dim / GAUSSIAN_STD_DIV_FACTOR

    # Log probability for numerical stability
    log_pdf = torch.zeros(x.shape[0], **tkwargs)

    for dim in range(x.shape[1]):
        log_norm = -0.5 * torch.log(2 * torch.tensor(torch.pi, **tkwargs)) - torch.log(std[dim])
        standardized = (x[:, dim] - mean[dim]) / std[dim]
        log_exp = -0.5 * standardized ** 2
        log_pdf += log_norm + log_exp

    pdf_values = torch.exp(log_pdf)

    if squeeze_output:
        return pdf_values.squeeze()

    return pdf_values


# Uniform Random Sampling Baseline (Random Policy)
def generate_uniform_samples(n):
    """Draw ``n`` i.i.d. uniform samples over the original problem box.

    Returns points in the original space ``townsend.bounds`` (shape ``[n, dim]``).
    """
    lower = townsend.bounds[0]
    upper = townsend.bounds[1]
    u = torch.rand(n, townsend.dim, **tkwargs)
    return lower + (upper - lower) * u


def uniform_pdf_value():
    """Constant density of the uniform distribution over ``townsend.bounds``.

    Equals ``1 / volume`` where ``volume = prod(upper - lower)``.
    """
    lower = townsend.bounds[0]
    upper = townsend.bounds[1]
    volume = torch.prod(upper - lower)
    return 1.0 / volume




# --------------------------------------------------------------------------- #
# Surrogate models for CBO (constrained Bayesian optimization)
# --------------------------------------------------------------------------- #
class GP_vi(ApproximateGP, GPyTorchModel):
    def __init__(self, train_x, train_y):
        self.train_inputs = (train_x,)
        self.train_targets = train_y

        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution
        )
        super(GP_vi, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _log_constraint_fit_failure(train_x, train_con, err, attempt):
    """Print diagnostics that help explain a constraint-GP fitting failure.

    The two usual culprits are (a) degenerate feasibility labels and (b) an
    ill-conditioned covariance from clustered/near-duplicate inputs (the variational
    GP uses every training point as an inducing point), which produces a non-finite
    ELBO/gradient and an L-BFGS-B ABNORMAL termination.
    """
    with torch.no_grad():
        labels, counts = torch.unique(train_con, return_counts=True)
        n = train_x.shape[0]
        if n > 1:
            dists = torch.cdist(train_x, train_x)
            dists.fill_diagonal_(float("inf"))
            min_dist = dists.min().item()
        else:
            min_dist = float("nan")
        any_nan = bool(torch.isnan(train_x).any().item())

    print(
        f"[initialize_model] Constraint GP fit failed ({attempt}): {err}\n"
        f"    n_train={n}, labels={labels.tolist()}, counts={counts.tolist()}, "
        f"min_pairwise_dist(normalized)={min_dist:.3e}, any_nan_in_train_x={any_nan}",
        file=sys.stderr,
    )


def _fit_constraint_model(mll_con, train_x, train_con):
    """Fit the variational feasibility GP robustly.

    The normal path is unchanged: a single successful ``fit_gpytorch_mll`` call.
    Only failure modes are handled specially so that one bad iteration/seed does
    not abort an entire multi-trial run:

    * Degenerate labels (all feasible or all infeasible) -> nothing to learn, so
      skip the optimizer and keep the prior model.
    * ``ModelFittingError`` -> log diagnostics, retry once under increased Cholesky
      jitter, and if that still fails fall back to the unfitted (prior) model.

    Returns ``True`` if the model was successfully fit, ``False`` otherwise.
    """
    # Nothing to learn from a single class; fitting it is also a common NaN trigger.
    unique_labels = torch.unique(train_con)
    if unique_labels.numel() < 2:
        # breakpoint()
        print(
            f"[initialize_model] Constraint labels are degenerate "
            f"(all == {unique_labels.tolist()}); skipping constraint GP fit and "
            f"using the prior constraint model for this iteration.",
            file=sys.stderr,
        )
        return False

    try:
        fit_gpytorch_mll(mll_con)
        return True
    except ModelFittingError as err:
        _log_constraint_fit_failure(train_x, train_con, err, attempt="initial")

    # Retry once with extra jitter to survive ill-conditioned covariances.
    try:
        with gpytorch.settings.cholesky_jitter(1e-4):
            fit_gpytorch_mll(mll_con)
        print(
            "[initialize_model] Constraint GP fit succeeded on jitter retry.",
            file=sys.stderr,
        )
        return True
    except ModelFittingError as err:
        _log_constraint_fit_failure(train_x, train_con, err, attempt="jitter-retry")
        print(
            "[initialize_model] Constraint GP fit failed after retry; falling back "
            "to the unfitted prior constraint model for this iteration.",
            file=sys.stderr,
        )
        return False


def _dedupe_jitter_inputs(train_x, min_sep=1e-4, jitter=1e-4):
    """Nudge near-duplicate rows of ``train_x`` apart to keep the constraint GP
    well-conditioned.

    The variational feasibility GP uses every training input as an inducing point,
    so (near-)duplicate rows make the inducing covariance ``K_zz`` singular and the
    ELBO fit diverges (L-BFGS-B ABNORMAL). Only rows whose nearest neighbour is
    closer than ``min_sep`` are perturbed by a tiny Gaussian ``jitter``; when all
    points are well separated the inputs are returned unchanged, so normal runs are
    unaffected. Operates in the normalized [0, 1] input space.
    """
    x = train_x.clone()
    n = x.shape[0]
    if n < 2:
        return x

    dists = torch.cdist(x, x)
    dists.fill_diagonal_(float("inf"))  # ignore self-distances
    too_close = dists.min(dim=1).values < min_sep
    if too_close.any():
        noise = torch.randn_like(x) * jitter
        x = torch.where(too_close.unsqueeze(-1), x + noise, x)
        # keep the perturbed inducing points inside the normalized domain
        x = x.clamp(0.0, 1.0)
    return x


def initialize_model(train_x, train_obj, train_con):
    """Initialize the model for the problem."""
    train_x = normalize(train_x, bounds=townsend.bounds)

    model_obj = SingleTaskGP(
        train_X=train_x,
        train_Y=train_obj,
    ).to(**tkwargs)

    mll_obj = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    fit_gpytorch_mll(mll_obj)

    # The feasibility GP uses each training point as an inducing point, so break up
    # any near-duplicate inputs before fitting to avoid a singular K_zz (hypothesis 1).
    train_x_con = _dedupe_jitter_inputs(train_x)
    model_con = GP_vi(train_x_con, train_con).to(**tkwargs)
    mll_con = gpytorch.mlls.VariationalELBO(
        model_con.likelihood, model_con, num_data=train_con.size(0)
    )

    # make sure the GPyTorch model is in double precision
    model_con.double()
    mll_con.double()

    _fit_constraint_model(mll_con, train_x_con, train_con)
    model = ModelListGP(model_obj, model_con)

    return model



# --------------------------------------------------------------------------- #
# Acquisition helpers.
# --------------------------------------------------------------------------- #
def pass_obj(Z, X=None):
    """Directly pass the objective to the acquisition function."""
    return Z[..., 0]


def pass_con(Z, model_con, X=None):
    """Pass the constraint to the acquisition function."""
    y_con = Z[..., 1]  # get the constraint
    prob = model_con.likelihood(y_con).probs  # probability constraint satisfied
    return prob + 1e-8  # small value avoids log(0) as qLogEI is used


def optimize_acqf_and_get_observation(model, train_obj, train_con):
    """Optimizes the acquisition function, and returns a new candidate and observation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    train_obj = train_obj.to(device)
    train_con = train_con.to(device)

    # best_f is the best feasible objective value observed so far
    best_f = np.ma.masked_array(train_obj.cpu().numpy(), mask=~train_con.bool().cpu().numpy()).max().item()

    # standardize the training data
    standard_bounds = torch.stack([torch.zeros(townsend.dim), torch.ones(townsend.dim)])

    acqf = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024])),
        objective=GenericMCObjective(pass_obj),
        constraints=[partial(pass_con, model_con=model.models[1])],
        fat=[None],
    )

    # run the optimization function
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    # observe new values
    new_x = unnormalize(candidates.detach(), townsend.bounds)
    new_obj = townsend(new_x)
    new_con = townsend.is_feasible(new_x)


    return new_x, new_obj, new_con, acqf



# --------------------------------------------------------------------------- #
# Conformal policy control (CPC) machinery.
# --------------------------------------------------------------------------- #
def prepare_grid(V, n_grid=250):
    """Sort and coarsen grid of lik-ratio values to search over.

    Args:
        V: 1-D array (or tensor) of likelihood-ratio values (unsorted).
        n_grid: Approximate number of values in the resulting grid.

    Returns:
        Sorted, coarsened grid with appropriate boundary values appended.
    """
    if isinstance(V, torch.Tensor):
        V = V.detach().cpu().numpy()

    G = np.sort(
        np.unique(V)
    )  # search in increasing order for CPC (safest to most aggressive)

    # Coarsen grid to approximately n_grid elements
    n_curr = len(G)
    k = max(int(n_curr / n_grid), 1)
    G = G[::k]

    # Construct grid
    G = np.concatenate(([sys.float_info.min], G, [np.inf]))

    return G


def est_opt_norm_const_from_uniform_samples(acqf):
    """Estimate the normalization constant for the optimized (Boltzmann) policy.

    Uses a *uniform* proposal over the whole domain for importance sampling:
        Z = E_x~Unif[ w(x) / p_unif(x) ] = volume * E_x~Unif[ w(x) ],
    where ``w(x) = exp(acq(x) / TEMPERATURE)`` and ``p_unif = 1 / volume``.

    A uniform proposal covers the entire box, so this is an unbiased, bounded-
    variance estimate of the integral even when ``w`` is sharply peaked (``w`` is
    bounded on the box). This avoids the heavy-tailed importance weights of the
    previous safe-Gaussian estimator, which under-sampled the acquisition peak and
    gave a high-variance / biased ``Z`` (and hence a distorted ``opt_pdf``).
    """
    uniform_samples = generate_uniform_samples(NUM_CANDIDATE_SAMPLES)

    weights = opt_acq_boltzmann_weight(uniform_samples, acqf)  # [NUM_CANDIDATE_SAMPLES]
    uniform_pdf_val = uniform_pdf_value()

    if weights is not None and len(weights) > 0:
        opt_norm_const_est = torch.mean(weights / uniform_pdf_val)
    else:
        opt_norm_const_est = None

    return opt_norm_const_est


def opt_acq_boltzmann_weight(x, acqf):
    """Unnormalized Boltzmann weight of the optimized policy at point(s) ``x``.

    The optimized policy is the Gibbs/Boltzmann distribution induced by the
    temperature-scaled acquisition function:

        w(x) = exp(acq(x) / TEMPERATURE),   pi_opt(x) = w(x) / Z.

    Two conventions are enforced here so the whole pipeline stays consistent:

    * Non-negativity: ``qLogExpectedImprovement`` returns *log*-scale values that
      are frequently negative, so the raw linear form ``acq / TEMPERATURE`` is not
      a valid (non-negative) density. Exponentiating gives a non-negative weight
      and matches the softmax used in ``rejection_sampling``.
    * Coordinate space: ``x`` is expressed in the *original* problem space
      (``townsend.bounds``). The GP model / acquisition function are fit on
      normalized inputs, so ``x`` is normalized to the unit cube before ``acqf``
      is evaluated. All callers therefore pass original-space coordinates and the
      resulting policy is a density over the original space (matching the Gaussian
      safe policy, so likelihood ratios are between densities on the same space).
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x_norm = normalize(x, bounds=townsend.bounds)
    with torch.no_grad():
        acq_value = acqf(x_norm.unsqueeze(1))
    return torch.exp(acq_value / TEMPERATURE)


def opt_pdf(x, acqf, opt_norm_const_est):
    """Evaluate the optimized (Boltzmann) policy PDF at point(s) ``x``.

    ``x`` is in the original problem space; normalization and non-negativity are
    handled inside ``opt_acq_boltzmann_weight``.
    """
    return opt_acq_boltzmann_weight(x, acqf) / opt_norm_const_est


def draw_optimized_samples(acqf, opt_norm_const_est, n=1, oversample=8, m_safety=1.10,
                           max_iters=MAX_ITERATIONS):
    """Draw i.i.d. samples from the optimized (Boltzmann) policy via rejection sampling.

    Proposals are drawn i.i.d. uniformly over the original problem box
    ``townsend.bounds``. Because the proposal density is *constant*, the rejection
    envelope collapses to a single scalar ``M >= sup_x w(x)`` where
    ``w(x) = exp(acq(x) / TEMPERATURE)`` is the unnormalized Boltzmann target, and
    a candidate is accepted with probability ``w(x) / M``. (A uniform proposal
    removes the need to *divide by a varying* proposal density, but it does *not*
    remove the need for the envelope constant ``M``.)

    Notes:
    * i.i.d. ``torch.rand`` proposals are used rather than a (correlated) Sobol
      sequence, since rejection sampling only yields i.i.d. target draws when the
      proposals are i.i.d.
    * The unknown normalization constant ``Z`` cancels in rejection sampling, so
      ``opt_norm_const_est`` is not needed here; it is retained in the signature
      for interface consistency with the other samplers.
    """
    lower = townsend.bounds[0]
    upper = townsend.bounds[1]
    dim = townsend.dim

    def uniform_original(m):
        # i.i.d. uniform draws over the original problem box [lower, upper].
        return lower + (upper - lower) * torch.rand(m, dim, **tkwargs)

    # Estimate the envelope M >= sup_x w(x) from a large i.i.d. pilot sample, with a
    # safety margin (a finite pilot can under-estimate the true supremum).
    pilot = uniform_original(max(NUM_CANDIDATE_SAMPLES, 5 * n))
    M = opt_acq_boltzmann_weight(pilot, acqf).max() * m_safety
    if not torch.isfinite(M) or M <= 0:
        raise RuntimeError("Envelope constant estimation failed (M non-finite or <= 0).")

    optimized_samples = []
    n_iters = 0
    while len(optimized_samples) < n and n_iters < max_iters:
        remaining = n - len(optimized_samples)
        candidates = uniform_original(oversample * remaining)
        weights = opt_acq_boltzmann_weight(candidates, acqf)
        u = torch.rand(candidates.shape[0], **tkwargs)
        accepted = candidates[u <= weights / M]
        for x in accepted[:remaining]:
            optimized_samples.append(x.detach().unsqueeze(0))
        n_iters += 1

    return optimized_samples


def est_constrained_norm_const_from_safe_samples(LRs_opt_over_safe, beta):
    """Estimate the normalization constant for the constrained (clipped) policy.

    Uses safe samples to estimate: E_safe[min(p_opt(x)/p_safe(x), beta)]
    """
    constrained_norm_const_est = torch.mean(torch.minimum(LRs_opt_over_safe, torch.tensor(beta, **tkwargs)))
    return constrained_norm_const_est


def get_LRs_grid(acqf, opt_norm_const_est, n_grid=250):
    safe_samples = generate_gaussian_samples(n=NUM_CANDIDATE_SAMPLES, mean=GAUSSIAN_MEAN)[0]
    safe_pdf_vals = gaussian_pdf(safe_samples, mean=GAUSSIAN_MEAN)
    opt_pdf_vals = opt_pdf(safe_samples, acqf, opt_norm_const_est)
    LRs_opt_over_safe = opt_pdf_vals / safe_pdf_vals

    G = prepare_grid(LRs_opt_over_safe, n_grid=n_grid)
    return G


def mixture_pdf_from_densities_mat(constrained_densities_cal_test_all_steps, mixture_weights):
    """Combine per-step densities into a mixture PDF.

    constrained_densities_cal_test_all_steps : dim (n_cal + 1, T), columns t=0..T-1
    mixture_weights : dim (T), relative weights on each *prior* distribution.
                      Note: mixture_weights[0] = n_cal_initial
    """
    mixture_weights_normed = mixture_weights / torch.sum(mixture_weights)
    mixture_pdfs = constrained_densities_cal_test_all_steps @ mixture_weights_normed
    return mixture_pdfs


# Conformal policy control (CPC) search for risk-controlling policy, with policies indexed by likelihood-ratio bounds, beta.
def cpc_search(t,
               alpha,
               cal_x,
               cal_con,
               cal_con_liks,
               cal_uncon_liks,
               acqfs,
               opt_norm_const_ests,
               constrained_norm_const_ests,
               betas,
               mixture_weights,
               n_grid=250):
    """
    Conformal Policy Control (CPC) search for risk-controlling policy, with policies indexed by likelihood-ratio bounds, beta.
    """

    n_cal = cal_x.shape[0]

    # Unlabeled safe (Gaussian) samples: used to estimate the constrained normalization
    # constant psi(beta) = E_safe[min(LR_opt/safe, beta)] via importance sampling.
    # The integrand min(., beta) is bounded, so the safe proposal is well-behaved here.
    safe_samples = generate_gaussian_samples(n=NUM_CANDIDATE_SAMPLES_CPC_TEST_WEIGHT, mean=GAUSSIAN_MEAN)[0]
    safe_pdf_vals = gaussian_pdf(safe_samples, mean=GAUSSIAN_MEAN)
    opt_pdf_vals = opt_pdf(safe_samples, acqfs[-1], opt_norm_const_ests[-1])
    LRs_opt_over_safe = opt_pdf_vals / safe_pdf_vals

    # Unlabeled uniform proposal samples over the whole domain: used both to (i) construct the
    # beta grid and (ii) conservatively estimate the test-point weight (empirical
    # maximum of the conformal weight). Alternatively, could use safe or optimized samples
    # here (or a mixture of the two).
    # (Note: these are not used as proposals in rejection sampling at the later
    # inference stage, which are drawn separately---sorry for overloading terms.)
    prop_samples = generate_uniform_samples(NUM_CANDIDATE_SAMPLES_CPC_TEST_WEIGHT)
    prop_safe_pdf_vals = gaussian_pdf(prop_samples, mean=GAUSSIAN_MEAN)
    prop_opt_pdf_vals = opt_pdf(prop_samples, acqfs[-1], opt_norm_const_ests[-1])

    # Construct the grid of candidate beta values from the likelihood ratios observed
    # at the uniform samples (grid construction may use any sampling distribution).
    LRs_opt_over_uniform = prop_opt_pdf_vals / prop_safe_pdf_vals
    G = prepare_grid(LRs_opt_over_uniform, n_grid=n_grid)

    # For each proposal sample, compute constrained PDF values for past policies,
    # \pi_0, \pi_1^{(\beta_1)}, ..., \pi_{t-1}^{(\beta_{t-1})}. Later used to compute
    # mixture PDF values for past policies and then the conservative test-point
    # weight (via empirical maximum).
    prop_con_liks = torch.zeros(NUM_CANDIDATE_SAMPLES_CPC_TEST_WEIGHT, t, **tkwargs)
    prop_con_liks[:, 0] = prop_safe_pdf_vals
    for i in range(1, t - 1):
        prop_opt_pdf_vals_i = opt_pdf(prop_samples, acqfs[i], opt_norm_const_ests[i])
        prop_con_liks[:, i] = torch.minimum(prop_opt_pdf_vals_i, betas[i] * prop_con_liks[:, 0]) / constrained_norm_const_ests[i]


    beta_last = G[0]
    psi_last = G[0]

    # For each candidate beta value, compute conservative (weighted) empirical risk and check if it exceeds alpha.
    for beta in G:
        # Estimate normalization constant for the current constrained policy (i.e., for current beta).
        psi = est_constrained_norm_const_from_safe_samples(LRs_opt_over_safe, beta)

        # Compute current constrained PDF values for calibration and proposal samples.
        cal_constrained_pdf_vals = torch.minimum(cal_uncon_liks[:n_cal, t], beta * gaussian_pdf(cal_x, mean=GAUSSIAN_MEAN)) / psi
        prop_constrained_pdf_vals = torch.minimum(prop_opt_pdf_vals, beta * prop_safe_pdf_vals) / psi
        
        # Compute past-policy mixture PDFs for calibration and proposal samples.
        cal_con_liks_mat = cal_con_liks[:n_cal, :t].view(1, -1) if cal_con_liks[:n_cal, :t].dim() == 1 else cal_con_liks[:n_cal, :t]
        cal_constrained_pdf_vals_mat = cal_constrained_pdf_vals.view(-1, 1) if cal_constrained_pdf_vals.dim() == 1 else cal_constrained_pdf_vals
        cal_densities_mat = torch.cat((cal_con_liks_mat, cal_constrained_pdf_vals_mat), dim=1)
        cal_mixture_pdfs = mixture_pdf_from_densities_mat(cal_densities_mat, mixture_weights[:t + 1])

        prop_con_liks_mat = prop_con_liks[:, :t].view(1, -1) if prop_con_liks[:, :t].dim() == 1 else prop_con_liks[:, :t]
        prop_constrained_pdf_vals_mat = prop_constrained_pdf_vals.view(-1, 1) if prop_constrained_pdf_vals.dim() == 1 else prop_constrained_pdf_vals
        prop_densities_mat = torch.cat((prop_con_liks_mat, prop_constrained_pdf_vals_mat), dim=1)
        prop_mixture_pdfs = mixture_pdf_from_densities_mat(prop_densities_mat, mixture_weights[:t + 1])

        # Estimate conformal weights (as constrained PDF / mixture PDF) for calibration and proposal samples.
        w_cal = cal_constrained_pdf_vals / cal_mixture_pdfs
        w_prop = prop_constrained_pdf_vals / prop_mixture_pdfs

        # Conservatively estimate test point weight via empirical maximum.
        w_test = torch.max(w_prop)

        # Self-normalize the conformal weights for calibration and (hypothetical) test points.
        w_cal_test = torch.cat((w_cal, torch.tensor([w_test], **tkwargs)))
        w_cal_test_normalized = w_cal_test / torch.sum(w_cal_test)

        # Compute empirical risk for the infeasibility constraint as weighted sum of infeasibility indicators 
        # for calibration infeasibility labels and assuming the worst-case of the test point being infeasible.
        cal_test_infeasibility_indicators = torch.cat((1 - cal_con, torch.tensor([1], **tkwargs)))
        w_empirical_risk = torch.sum(w_cal_test_normalized * cal_test_infeasibility_indicators)

        # Check if empirical risk exceeds alpha. If so, return the last beta and psi values.
        if w_empirical_risk > alpha:
            return beta_last, psi_last
        else:
            beta_last = beta
            psi_last = psi

    # If all beta values have empirical risk less than or equal to alpha, then use optimized policy (i.e., beta = infinity).
    return np.inf, 1.0



# Rejection sampling to obtain new samples from the risk-controlling policy
def rejection_sampling(acqf, opt_norm_const_est, n_target=1, use_gaussian_clip=False, beta=1.0,
                       gaussian_mean=None, gaussian_std=None):
    """Sample new point(s) either by softmax sampling (beta=inf) or rejection sampling.

    Returns lists of length ``n_target``; each entry is a single point of shape
    ``[1, dim]`` (with matching scalar objective/constraint entries). Callers that
    want a batch of ``B`` points simply request ``n_target=B`` and concatenate.
    """
    accepted_samples = []

    n_iters = 0

    if beta == np.inf:
        optimized_samples = draw_optimized_samples(acqf, opt_norm_const_est, n_target)
        accepted_samples.extend(optimized_samples)

    else:

        if beta < 1.0:
            # Propose from safe policy

            while len(accepted_samples) < n_target and n_iters < MAX_ITERATIONS:
                safe_prop = generate_gaussian_samples(n=1, mean=GAUSSIAN_MEAN)[0]

                # Compute likelihood ratio
                LR_opt_over_safe = opt_pdf(safe_prop, acqf, opt_norm_const_est) / gaussian_pdf(safe_prop, mean=GAUSSIAN_MEAN)

                # Accept if uniform random draw is less than LR/beta
                if np.random.rand() <= LR_opt_over_safe / beta:
                    accepted_samples.append(safe_prop)

                n_iters += 1
                
        else:
            # Propose from optimized policy. First need uniform samples to get proposals from optimized policy:

            while len(accepted_samples) < n_target and n_iters < MAX_ITERATIONS:

                # Sample optimized proposals from candidate_samples according to optimized pdf
                # selected_idx = torch.multinomial(probs, num_samples=1)
                # opt_prop = unnormalize(candidate_samples[selected_idx].detach(), townsend.bounds) ## Proposal from optimized policy
                opt_prop = draw_optimized_samples(acqf, opt_norm_const_est, n=1)[0]
                # Compute likelihood ratio
                LR_opt_over_safe = opt_pdf(opt_prop, acqf, opt_norm_const_est) / gaussian_pdf(opt_prop, mean=GAUSSIAN_MEAN)

                # Accept if uniform random draw is less than LR/beta
                if np.random.rand() <= beta / LR_opt_over_safe:
                    accepted_samples.append(opt_prop)

                n_iters += 1
            

    if len(accepted_samples) < n_target:
        ## If still haven't drawn target number of samples (due to reaching max iterations), draw from safe policy
        while len(accepted_samples) < n_target:
            safe_sample = generate_gaussian_samples(n=1, mean=GAUSSIAN_MEAN)[0]
            accepted_samples.append(safe_sample)


    new_obj = [townsend(new_x) for new_x in accepted_samples]
    new_con = [townsend.is_feasible(new_x) for new_x in accepted_samples]

    return accepted_samples, new_obj, new_con


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #
def plot_townsend(ax):
    x = np.linspace(-3.0, 3.0, 100)
    y = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(x, y)

    obj = townsend(torch.tensor(np.stack([X, Y], axis=-1), **tkwargs)).cpu().numpy()
    con = townsend.is_feasible(torch.tensor(np.stack([X, Y], axis=-1), **tkwargs)).cpu().numpy()

    # mask out the constraint region < 0
    obj[con == 0] = np.nan

    c = ax.contourf(X, Y, obj, levels=20, cmap="Blues")

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("Townsend Problem")

    plt.colorbar(c, ax=ax, orientation="vertical")
    return ax


def plot_helper(model, train_x, new_x, acqf, axes):
    with torch.no_grad():
        x = np.linspace(-3.0, 3.0, 100)
        y = np.linspace(-3.0, 3.0, 100)
        X, Y = np.meshgrid(x, y)
        Z = torch.tensor(np.stack([X, Y], axis=-1)).to(**tkwargs)
        Z = normalize(Z, bounds=townsend.bounds)
        Z = Z.reshape(-1, 2).unsqueeze(1)
        # get the acquisition function value
        acq_values = acqf(Z).cpu().numpy()
        # get the constraint probability
        model_con = model.models[1]
        prob = model_con.likelihood(model_con(Z)).probs.cpu().numpy()
        # get the expected improvement value
        ei_values = model.models[0](Z).mean.cpu().numpy()

    plot_townsend(axes[0])

    c_acqf = axes[1].contourf(X, Y, acq_values.reshape(100, 100), levels=20, cmap="Blues")
    c_prob = axes[2].contourf(X, Y, prob.reshape(100, 100), levels=20, cmap="RdYlGn", vmin=0, vmax=1)
    c_ei = axes[3].contourf(X, Y, ei_values.reshape(100, 100), levels=20, cmap="Oranges")

    for ax in axes:
        ax.scatter(train_x[:, 0].cpu(), train_x[:, 1].cpu(), color="grey", label="Observations", alpha=0.5)
        ax.scatter(new_x[:, 0].cpu(), new_x[:, 1].cpu(), marker="*", color="red", label="New Point")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")

    axes[1].set_title("Acquisition Function")
    axes[2].set_title("Constraint Probability")
    axes[3].set_title("EI value")

    plt.colorbar(c_acqf, ax=axes[1])
    plt.colorbar(c_prob, ax=axes[2])
    plt.colorbar(c_ei, ax=axes[3])
    plt.tight_layout()


def _savefig_tight_subplot(ax, path, pad_inches=0.1, dpi=300):
    """Save just the region of ``ax`` (labels, title, legend included) to ``path``.

    ``Axes.get_window_extent`` only covers the bare axes frame, so cropping to
    it (as ``bbox_inches=`` requires) cuts off axis labels, titles, and
    legends. ``Axes.get_tightbbox`` additionally accounts for those
    decorations, so use it instead to compute the crop box.
    """
    parent_fig = ax.get_figure()
    parent_fig.canvas.draw()  # ensure text/legend layout is up to date before measuring
    renderer = parent_fig.canvas.get_renderer()
    tight_bbox = ax.get_tightbbox(renderer)
    extent = tight_bbox.transformed(parent_fig.dpi_scale_trans.inverted())
    parent_fig.savefig(path, bbox_inches=extent.padded(pad_inches), dpi=dpi)


def plot_averaged_results(aggregated_results, show_std=True, std_multiplier=1.0, output_dir=".",
                          include_gaussian=True):
    """Plot averaged results from multiple trials (mean with shaded SE bands).

    ``include_gaussian`` controls whether the "Gaussian Random Sampling" baseline
    is drawn alongside the uniform-random and BO curves (it is always computed
    and included in the printed summary / saved CSVs regardless of this flag).
    """
    BLUE = "#2166ac"
    RED = "#d73027"
    GREEN = "#1a9850"  # Gaussian Random

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    fig.suptitle(f"Temperature = {TEMPERATURE}", fontsize=18, fontweight="bold")

    label_fs = 16

    optimal = townsend._optimal_value
    bo_configs = aggregated_results["bo_configs"]
    baselines = aggregated_results["baselines"]
    n_trials = aggregated_results.get("n_trials", "N")

    n_iters = len(baselines["uniform"]["regrets_mean"])
    iterations = range(n_iters)
    iterations_per_round = range(n_iters - 1)  # One fewer for per-round metrics

    # Plot 1: Average Best Objective Value Over Time
    ax1 = axes[1]

    best_obj_uniform_mean = optimal - baselines["uniform"]["regrets_mean"]
    best_obj_uniform_std = baselines["uniform"]["regrets_std"]
    ax1.plot(iterations, best_obj_uniform_mean, linewidth=2.5,
             label="Uniform Random", color="gray", alpha=0.8, marker="s", markersize=8)
    if show_std:
        ax1.fill_between(iterations,
                         best_obj_uniform_mean - std_multiplier * best_obj_uniform_std / np.sqrt(n_trials),
                         best_obj_uniform_mean + std_multiplier * best_obj_uniform_std / np.sqrt(n_trials),
                         color="gray", alpha=0.15)

    if include_gaussian:
        best_obj_gaussian_mean = optimal - baselines["gaussian"]["regrets_mean"]
        best_obj_gaussian_std = baselines["gaussian"]["regrets_std"]
        ax1.plot(iterations, best_obj_gaussian_mean, linewidth=2.5,
                 label="Gaussian Random", color=GREEN, alpha=0.9, marker="^", markersize=8)
        if show_std:
            ax1.fill_between(iterations,
                             best_obj_gaussian_mean - std_multiplier * best_obj_gaussian_std / np.sqrt(n_trials),
                             best_obj_gaussian_mean + std_multiplier * best_obj_gaussian_std / np.sqrt(n_trials),
                             color=GREEN, alpha=0.2)

    for i, bo_config in enumerate(bo_configs):
        best_obj_mean = optimal - bo_config["regrets_mean"]
        best_obj_std = bo_config["regrets_std"]  # std of regret

        if bo_config["CPC"]:
            label = "CPC+CBO," + r"$\alpha$=" + f"{bo_config['alpha']}"
            color = BLUE
            marker = "o"
        else:
            label = "CBO (no CPC)"
            color = RED
            marker = "X"

        ax1.plot(iterations, best_obj_mean, color=color, linewidth=2.5, label=label, marker=marker, markersize=8)

        if show_std:
            ax1.fill_between(iterations,
                             best_obj_mean - std_multiplier * best_obj_std / np.sqrt(n_trials),
                             best_obj_mean + std_multiplier * best_obj_std / np.sqrt(n_trials),
                             color=color, alpha=0.2)

    # ax1.axhline(y=optimal, color="black", xmin=0, xmax=n_iters - 1, linestyle=":", alpha=0.75, label="True Optimum", linewidth=1.5)
    ax1.set_xlabel("Iteration", fontsize=label_fs)
    ax1.set_ylabel(r"Best Objective Value Found [$\rightarrow$]", fontsize=label_fs)
    ax1.set_title(f"Average Best Objective Value Over Time\n({n_trials} trials, shaded = ± SE)", fontsize=label_fs)
    ax1.legend(fontsize=label_fs)
    ax1.grid(True, alpha=0.3)

    _savefig_tight_subplot(
        ax1,
        os.path.join(output_dir, f"ConservativeOptExpts_Objective_{SETTING}.pdf"),
    )
    
    # Plot 2: Average Per-Round Constraint Violations
    ax2 = axes[0]

    violations_uniform_mean = baselines["uniform"]["violations_per_round_mean"]
    violations_uniform_std = baselines["uniform"]["violations_per_round_std"]
    ax2.plot(iterations_per_round, violations_uniform_mean, linewidth=2.5, marker="s",
             label="Uniform Random", color="gray", alpha=0.8, markersize=8)
    if show_std:
        ax2.fill_between(iterations_per_round,
                         violations_uniform_mean - std_multiplier * violations_uniform_std / np.sqrt(n_trials),
                         violations_uniform_mean + std_multiplier * violations_uniform_std / np.sqrt(n_trials),
                         color="gray", alpha=0.15)

    if include_gaussian:
        violations_gaussian_mean = baselines["gaussian"]["violations_per_round_mean"]
        violations_gaussian_std = baselines["gaussian"]["violations_per_round_std"]
        ax2.plot(iterations_per_round, violations_gaussian_mean, linewidth=2.5, marker="^",
                 label="Gaussian Random", color=GREEN, alpha=0.9, markersize=8)
        if show_std:
            ax2.fill_between(iterations_per_round,
                             violations_gaussian_mean - std_multiplier * violations_gaussian_std / np.sqrt(n_trials),
                             violations_gaussian_mean + std_multiplier * violations_gaussian_std / np.sqrt(n_trials),
                             color=GREEN, alpha=0.15)

    for i, bo_config in enumerate(bo_configs):
        violations_per_round_mean = bo_config["violations_per_round_mean"]
        violations_per_round_std = bo_config["violations_per_round_std"]

        if bo_config["CPC"]:
            label = "CPC+CBO," + r"$\alpha$=" + f"{bo_config['alpha']}"
            color = BLUE
            marker = "o"
        else:
            label = "CBO (no CPC)"
            color = RED
            marker = "X"

        ax2.plot(iterations_per_round, violations_per_round_mean, color=color, marker=marker,
                 linewidth=2.5, label=label, markersize=8)

        if show_std:
            ax2.fill_between(iterations_per_round,
                             violations_per_round_mean - std_multiplier * violations_per_round_std / np.sqrt(n_trials),
                             violations_per_round_mean + std_multiplier * violations_per_round_std / np.sqrt(n_trials),
                             color=color, alpha=0.2)

        if bo_config["CPC"]:
            ax2.axhline(y=bo_config["alpha"], color=color, xmin=0, xmax=n_iters - 1, linestyle="--", label=r"$\alpha$=" + f'{bo_config["alpha"]}', linewidth=2.5)

    ax2.set_xlabel("Iteration", fontsize=label_fs)
    ax2.set_ylabel(r"Constraint Violations per Round [$\leftarrow$]", fontsize=label_fs)
    ax2.set_title(f"Average Per-Round Constraint Violations\n({n_trials} trials, shaded = ± SE)", fontsize=label_fs)
    ax2.legend(fontsize=label_fs)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.05, 1.05])  # violations per round are 0 or 1

    _savefig_tight_subplot(
        ax2,
        os.path.join(output_dir, f"ConservativeOptExpts_Constraint_{SETTING}.pdf"),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"CPC_vs_CBO_{SETTING}.pdf"), dpi=300)
    plt.close(fig)

    # Print summary statistics
    print("\n" + "=" * 80)
    print(f"AVERAGED RESULTS ACROSS {n_trials} TRIALS")
    print("=" * 80)

    for bo_config in aggregated_results["bo_configs"]:
        if bo_config["CPC"]:
            method_name = f"BO with Gaussian Clip (β={bo_config['CPC']})"
        else:
            method_name = "BO (Probabilistic, no clip)"

        print(f"\n{method_name}:")
        print(f"  Best objective (mean): {bo_config['best_f_mean']:.6f}")
        print(f"  Final regret (mean ± std): {bo_config['regrets_mean'][-1]:.6f} ± {bo_config['regrets_std'][-1]:.6f}")
        print(f"  Final violations (mean ± std): {bo_config['violations_mean'][-1]:.2f} ± {bo_config['violations_std'][-1]:.2f}")
        print(f"  Avg violation rate: {bo_config['violations_per_round_mean'].mean():.3f}")
        

    print("\nUniform Random Sampling:")
    print(f"  Best objective (mean): {baselines['uniform']['best_f_mean']:.6f}")
    print(f"  Final regret (mean ± std): {baselines['uniform']['regrets_mean'][-1]:.6f} ± {baselines['uniform']['regrets_std'][-1]:.6f}")
    print(f"  Final violations (mean ± std): {baselines['uniform']['violations_mean'][-1]:.2f} ± {baselines['uniform']['violations_std'][-1]:.2f}")
    print(f"  Avg violation rate: {baselines['uniform']['violations_per_round_mean'].mean():.3f}")

    print("\nGaussian Random Sampling:")
    print(f"  Best objective (mean): {baselines['gaussian']['best_f_mean']:.6f}")
    print(f"  Final regret (mean ± std): {baselines['gaussian']['regrets_mean'][-1]:.6f} ± {baselines['gaussian']['regrets_std'][-1]:.6f}")
    print(f"  Final violations (mean ± std): {baselines['gaussian']['violations_mean'][-1]:.2f} ± {baselines['gaussian']['violations_std'][-1]:.2f}")
    print(f"  Avg violation rate: {baselines['gaussian']['violations_per_round_mean'].mean():.3f}")

    print(f"\nTrue Optimum: {townsend._optimal_value:.6f}")
    print("=" * 80)


def _config_label(bo_config):
    """Human-readable label matching the plot legends for a BO config."""
    if bo_config["CPC"]:
        return f"CPC+CBO,alpha={bo_config['alpha']}"
    return "CBO (no CPC)"


def save_averaged_results_to_csv(aggregated_results, output_dir="."):
    """Write the data underlying the summary figures to CSV files.

    Produces two long-format CSVs per temperature and per CPC alpha:
      * ``Objective_temp{T}_alpha{CPC_ALPHA}.csv`` -- data for the "best objective over time" panel.
      * ``Constraint_temp{T}_alpha{CPC_ALPHA}.csv`` -- data for the "per-round violations" panel.

    Both include the mean, std, and standard error (std / sqrt(n_trials)) so the
    shaded bands in the figures can be reproduced exactly.
    """
    optimal = townsend._optimal_value
    bo_configs = aggregated_results["bo_configs"]
    baselines = aggregated_results["baselines"]
    n_trials = aggregated_results.get("n_trials", 1)
    sqrt_n = np.sqrt(n_trials) if n_trials else 1.0

    # ----- Objective panel (best objective value found per iteration) -----
    obj_path = os.path.join(output_dir, f"Objective_{SETTING}.csv")
    with open(obj_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "temperature", "method", "iteration",
            "regret_mean", "regret_std",
            "best_obj_mean", "best_obj_se",
        ])

        def write_obj_rows(method, regrets_mean, regrets_std):
            best_obj_mean = optimal - np.asarray(regrets_mean)
            se = np.asarray(regrets_std) / sqrt_n
            for it in range(len(regrets_mean)):
                writer.writerow([
                    TEMPERATURE, method, it,
                    float(regrets_mean[it]), float(regrets_std[it]),
                    float(best_obj_mean[it]), float(se[it]),
                ])

        for name, key in [("Uniform Random", "uniform"), ("Gaussian Random", "gaussian")]:
            write_obj_rows(name, baselines[key]["regrets_mean"], baselines[key]["regrets_std"])
        for bo_config in bo_configs:
            write_obj_rows(_config_label(bo_config), bo_config["regrets_mean"], bo_config["regrets_std"])

    # ----- Constraint panel (per-round constraint violations) -----
    con_path = os.path.join(output_dir, f"Constraint_{SETTING}.csv")
    with open(con_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "temperature", "method", "iteration",
            "violations_per_round_mean", "violations_per_round_std",
            "violations_per_round_se",
        ])

        def write_con_rows(method, mean, std):
            mean = np.asarray(mean)
            se = np.asarray(std) / sqrt_n
            for it in range(len(mean)):
                writer.writerow([
                    TEMPERATURE, method, it,
                    float(mean[it]), float(std[it]), float(se[it]),
                ])

        for name, key in [("Uniform Random", "uniform"), ("Gaussian Random", "gaussian")]:
            write_con_rows(name, baselines[key]["violations_per_round_mean"], baselines[key]["violations_per_round_std"])
        for bo_config in bo_configs:
            write_con_rows(_config_label(bo_config), bo_config["violations_per_round_mean"], bo_config["violations_per_round_std"])

    print(f"Saved figure data to {obj_path} and {con_path}")
    return obj_path, con_path


# --------------------------------------------------------------------------- #
# Experiment drivers.
# --------------------------------------------------------------------------- #
def run_bo_with_baselines(n_iterations=25, init_training_data_size=10, init_calibration_data_size=10, alphas=[1.0, 0.3], verbose=True, plot_every=5,
                          use_gaussian_clip=False, n_grid=250, output_dir="."):
    """Run constrained BO with CPC together with two random baselines."""
    if init_training_data_size is None:
        init_training_data_size = 10
    
    if init_calibration_data_size is None:
        init_calibration_data_size = 10

    # Determine which BO configurations to run
    if alphas is not None:
        bo_configs = [{"CPC": True if alpha < 1.0 else False, "alpha": alpha} for alpha in alphas]
    else:
        bo_configs = [{"CPC": False, "alpha": None}]

    # Initialize baselines (shared across all BO configs)
    train_x_init, train_obj_init, train_con_init = generate_initial_training_data(n_train=init_training_data_size)
    cal_x_init, cal_obj_init, cal_con_init = generate_initial_calibration_data(n_cal=init_calibration_data_size)

    # Initialize uniform random baseline
    train_x_uniform = train_x_init.clone()
    train_obj_uniform = train_obj_init.clone()
    train_con_uniform = train_con_init.clone()

    # Initialize Gaussian random baseline
    train_x_gaussian = train_x_init.clone()
    train_obj_gaussian = train_obj_init.clone()
    train_con_gaussian = train_con_init.clone()

    # Storage for baselines
    optimal = torch.tensor(townsend._optimal_value, **tkwargs)
    regrets_uniform = []
    regrets_gaussian = []
    violations_uniform = []
    violations_gaussian = []

    # Storage for all BO configurations
    bo_results = {}
    for i, config in enumerate(bo_configs):
        key = f"CPC_{config['CPC']}"
        bo_results[key] = {
            "train_x": train_x_init.clone(),
            "train_obj": train_obj_init.clone(),
            "train_con": train_con_init.clone(),
            "cal_x": cal_x_init.clone(),
            "cal_obj": cal_obj_init.clone(),
            "cal_con": cal_con_init.clone(),
            "cal_con_liks": torch.zeros(init_calibration_data_size + BATCH_SIZE * n_iterations + 1, n_iterations + 1, **tkwargs),
            "cal_uncon_liks": torch.zeros(init_calibration_data_size + BATCH_SIZE * n_iterations + 1, n_iterations + 1, **tkwargs),
            "model": initialize_model(train_x_init, train_obj_init, train_con_init),
            "acqfs": [None],
            "opt_norm_const_ests": [1.0],
            "constrained_norm_const_ests": [1.0],
            "betas": [np.inf],
            "regrets": [],
            "violations": [],
            "CPC": config["CPC"],
            "alpha": config["alpha"],
        }
        initial_cal_data_safe_liks = gaussian_pdf(bo_results[key]["cal_x"], mean=GAUSSIAN_MEAN)
        for j in range(init_calibration_data_size):
            bo_results[key]["cal_con_liks"][j, 0] = initial_cal_data_safe_liks[j]
            bo_results[key]["cal_uncon_liks"][j, 0] = initial_cal_data_safe_liks[j]

    # Create vector of mixture weights. Each entry counts how many calibration
    # points were drawn from that policy: init_data_size from the initial safe
    # prior (column 0) and BATCH_SIZE from every subsequent per-iteration policy.
    mixture_weights = torch.ones(n_iterations + 1, **tkwargs) * BATCH_SIZE
    mixture_weights[0] = init_calibration_data_size

    for iteration in tqdm(range(n_iterations + 1)):
        # Compute regrets and violations for baselines
        best_f_uniform = np.ma.masked_array(train_obj_uniform.cpu().numpy(), mask=~train_con_uniform.bool().cpu().numpy()).max().item()
        regrets_uniform.append((optimal - best_f_uniform).item())
        violations_uniform.append((~train_con_uniform.bool()).sum().item())

        best_f_gaussian = np.ma.masked_array(train_obj_gaussian.cpu().numpy(), mask=~train_con_gaussian.bool().cpu().numpy()).max().item()
        regrets_gaussian.append((optimal - best_f_gaussian).item())
        violations_gaussian.append((~train_con_gaussian.bool()).sum().item())

        # Compute regrets and violations for each BO configuration
        for key, bo_data in bo_results.items():
            best_f_bo = np.ma.masked_array(bo_data["train_obj"].cpu().numpy(), mask=~bo_data["train_con"].bool().cpu().numpy()).max().item()
            bo_data["regrets"].append((optimal - best_f_bo).item())
            bo_data["violations"].append((~bo_data["train_con"].bool()).sum().item())

        if verbose:
            print(f"Iteration {iteration}:")
            for key, bo_data in bo_results.items():
                if bo_data["CPC"]:
                    label = f"CPC+CBO, alpha={bo_data['alpha']}"
                else:
                    label = "CBO (no CPC)"
                print(f"  log Regret {label:20s} = {torch.log(torch.tensor(bo_data['regrets'][-1])):.2f}, Violations = {bo_data['violations'][-1]}")
            print(f"  log Regret Uniform         = {torch.log(torch.tensor(regrets_uniform[-1])):.2f}, Violations = {violations_uniform[-1]}")
            print(f"  log Regret Gaussian        = {torch.log(torch.tensor(regrets_gaussian[-1])):.2f}, Violations = {violations_gaussian[-1]}")

        if iteration < n_iterations:  # Don't get new observations after last iteration
            for key, bo_data in bo_results.items():

                # Train: Create optimized policy / acquisition function
                best_f = np.ma.masked_array(bo_data["train_obj"].cpu().numpy(), mask=~bo_data["train_con"].bool().cpu().numpy()).max().item()

                acqf = qLogExpectedImprovement(
                    model=bo_data["model"],
                    best_f=best_f,
                    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024])),
                    objective=GenericMCObjective(pass_obj),
                    constraints=[partial(pass_con, model_con=bo_data["model"].models[1])],
                    fat=[None],
                )
                bo_data["acqfs"].append(acqf)
                opt_norm_const_est = est_opt_norm_const_from_uniform_samples(acqf)
                bo_data["opt_norm_const_ests"].append(opt_norm_const_est)

                cal_opt_pdf_vals = opt_pdf(bo_data["cal_x"], acqf, opt_norm_const_est)

                # Compute optimized likelihoods for new policy
                n_cal = bo_data["cal_x"].shape[0]
                for i in range(n_cal):
                    bo_data["cal_uncon_liks"][i, iteration + 1] = cal_opt_pdf_vals[i]

                # Calibration
                if bo_data["CPC"]:
                    beta_hat, psi_hat = cpc_search(t=iteration + 1,
                                                   alpha=bo_data["alpha"],
                                                   cal_x=bo_data["cal_x"],
                                                   cal_con=bo_data["cal_con"],
                                                   cal_con_liks=bo_data["cal_con_liks"],
                                                   cal_uncon_liks=bo_data["cal_uncon_liks"],
                                                   acqfs=bo_data["acqfs"],
                                                   opt_norm_const_ests=bo_data["opt_norm_const_ests"],
                                                   constrained_norm_const_ests=bo_data["constrained_norm_const_ests"],
                                                   betas=bo_data["betas"],
                                                   mixture_weights=mixture_weights,
                                                   n_grid=n_grid)
                    if verbose:
                        print(f"beta_hat : {beta_hat}")
                else:
                    beta_hat, psi_hat = np.inf, 1.0

                bo_data["betas"].append(beta_hat)

                # Estimate normalization constant for constrained policy
                bo_data["constrained_norm_const_ests"].append(psi_hat)

                # Generation
                # Draw 2 * BATCH_SIZE points: the first BATCH_SIZE become new
                # training points, the last BATCH_SIZE become new calibration points.
                new_x, new_obj, new_con = rejection_sampling(
                    acqf=acqf,
                    opt_norm_const_est=opt_norm_const_est,
                    n_target=2 * BATCH_SIZE,
                    beta=beta_hat if bo_data["CPC"] else np.inf,
                )
                new_x_train = torch.cat(new_x[:BATCH_SIZE], dim=0)
                new_obj_train = torch.cat(new_obj[:BATCH_SIZE], dim=0)
                new_con_train = torch.cat(new_con[:BATCH_SIZE], dim=0)
                new_x_cal = torch.cat(new_x[BATCH_SIZE:], dim=0)
                new_obj_cal = torch.cat(new_obj[BATCH_SIZE:], dim=0)
                new_con_cal = torch.cat(new_con[BATCH_SIZE:], dim=0)


                # Plotting (only for first BO config to avoid too many plots)
                if key == list(bo_results.keys())[0] and plot_every is not None and iteration % plot_every == 0:
                    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
                    plot_helper(bo_data["model"], bo_data["train_x"], new_x_train, acqf, axes)
                    if bo_data["CPC"]:
                        fig.suptitle("BO with CPC ", fontsize=12)
                    fig.savefig(os.path.join(output_dir, f"bo_iter{iteration}_{SETTING}.pdf"), dpi=200)
                    plt.close(fig)

                # Update BO training data
                bo_data["train_x"] = torch.cat([bo_data["train_x"], new_x_train])
                bo_data["train_obj"] = torch.cat([bo_data["train_obj"], new_obj_train.unsqueeze(-1)])
                bo_data["train_con"] = torch.cat([bo_data["train_con"], new_con_train])
                bo_data["model"] = initialize_model(bo_data["train_x"], bo_data["train_obj"], bo_data["train_con"])

                # Update BO calibration data
                bo_data["cal_x"] = torch.cat([bo_data["cal_x"], new_x_cal])
                bo_data["cal_obj"] = torch.cat([bo_data["cal_obj"], new_obj_cal.unsqueeze(-1)])
                bo_data["cal_con"] = torch.cat([bo_data["cal_con"], new_con_cal])
                n_cal = bo_data["cal_x"].shape[0]

                # Compute constrained likelihoods for the new policy
                cal_constrained_pdf_vals = torch.minimum(bo_data["cal_uncon_liks"][:n_cal, iteration + 1], torch.tensor(beta_hat, **tkwargs) * gaussian_pdf(bo_data["cal_x"], mean=GAUSSIAN_MEAN)) / psi_hat

                # Update constrained likelihoods for most recent policy/column
                for i in range(n_cal):
                    bo_data["cal_con_liks"][i, iteration + 1] = cal_constrained_pdf_vals[i]

                # Add constrained likelihoods for previous policies on each of the
                # BATCH_SIZE new calibration datapoints (rows n_cal-BATCH_SIZE .. n_cal-1).
                gaussian_liks_new = gaussian_pdf(new_x_cal, mean=GAUSSIAN_MEAN)  # [BATCH_SIZE]
                for j in range(0, iteration):
                    if j == 0:
                        for b in range(BATCH_SIZE):
                            row = n_cal - BATCH_SIZE + b
                            bo_data["cal_uncon_liks"][row, j] = gaussian_liks_new[b]
                            bo_data["cal_con_liks"][row, j] = gaussian_liks_new[b]
                    else:
                        test_opt_pdf_curr = opt_pdf(new_x_cal, bo_data["acqfs"][j], bo_data["opt_norm_const_ests"][j])  # [BATCH_SIZE]
                        con_liks_curr = torch.minimum(test_opt_pdf_curr, torch.tensor(beta_hat, **tkwargs) * gaussian_liks_new) / bo_data["constrained_norm_const_ests"][j]  # [BATCH_SIZE]
                        for b in range(BATCH_SIZE):
                            row = n_cal - BATCH_SIZE + b
                            bo_data["cal_uncon_liks"][row, j] = test_opt_pdf_curr[b]
                            bo_data["cal_con_liks"][row, j] = con_liks_curr[b]

            # Update uniform random baseline (must stay genuinely uniform, since
            # generate_initial_data now draws from the Gaussian safe policy).
            new_x_uniform = generate_uniform_samples(BATCH_SIZE)
            new_obj_uniform = townsend(new_x_uniform).unsqueeze(-1)
            new_con_uniform = townsend.is_feasible(new_x_uniform)
            train_x_uniform = torch.cat([train_x_uniform, new_x_uniform])
            train_obj_uniform = torch.cat([train_obj_uniform, new_obj_uniform])
            train_con_uniform = torch.cat([train_con_uniform, new_con_uniform])

            # Update Gaussian random baseline
            new_x_gaussian, new_obj_gaussian, new_con_gaussian = generate_gaussian_samples(n=BATCH_SIZE, mean=GAUSSIAN_MEAN)
            train_x_gaussian = torch.cat([train_x_gaussian, new_x_gaussian])
            train_obj_gaussian = torch.cat([train_obj_gaussian, new_obj_gaussian])
            train_con_gaussian = torch.cat([train_con_gaussian, new_con_gaussian])

    # Compile results
    results = {
        "regrets_uniform": regrets_uniform,
        "regrets_gaussian": regrets_gaussian,
        "violations_uniform": violations_uniform,
        "violations_gaussian": violations_gaussian,
        "train_x_uniform": train_x_uniform,
        "train_x_gaussian": train_x_gaussian,
        "train_obj_uniform": train_obj_uniform,
        "train_obj_gaussian": train_obj_gaussian,
        "train_con_uniform": train_con_uniform,
        "train_con_gaussian": train_con_gaussian,
        "best_f_uniform": best_f_uniform,
        "best_f_gaussian": best_f_gaussian,
        "bo_configs": [],  # List of BO configuration info,
        "betas_selected": bo_results["CPC_True"]["betas"]
    }

    # Add each BO configuration's results
    for key, bo_data in bo_results.items():
        best_f_bo = np.ma.masked_array(bo_data["train_obj"].cpu().numpy(), mask=~bo_data["train_con"].bool().cpu().numpy()).max().item()

        bo_result = {
            "key": key,
            "CPC": bo_data["CPC"],
            "alpha": bo_data["alpha"],
            "regrets": bo_data["regrets"],
            "violations": bo_data["violations"],
            "train_x": bo_data["train_x"],
            "train_obj": bo_data["train_obj"],
            "train_con": bo_data["train_con"],
            "best_f": best_f_bo,
        }
        results["bo_configs"].append(bo_result)

    # For backwards compatibility, if single BO config, also add old-style keys
    if len(bo_results) == 1:
        single_bo = results["bo_configs"][0]
        results["regrets_bo"] = single_bo["regrets"]
        results["violations_bo"] = single_bo["violations"]
        results["train_x_bo"] = single_bo["train_x"]
        results["train_obj_bo"] = single_bo["train_obj"]
        results["train_con_bo"] = single_bo["train_con"]
        results["best_f_bo"] = single_bo["best_f"]
        results["CPC"] = single_bo["CPC"]
        results["alpha"] = single_bo["alpha"]

    return results


def run_multiple_trials(n_trials=10, n_iterations=25, init_training_data_size=10, init_calibration_data_size=10,
                        alphas=[1.0, 0.3], use_gaussian_clip=False,
                        verbose=True, plot_every=None, seed_start=0, n_grid=250,
                        output_dir="."):
    """Run BO with baselines multiple times with different seeds and aggregate."""
    all_results = []

    for trial in range(n_trials):
        # Set random seed for reproducibility
        seed = seed_start + trial
        torch.manual_seed(seed)
        np.random.seed(seed)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Running Trial {trial+1}/{n_trials} (seed={seed})")
            print(f"{'='*70}")

        results = run_bo_with_baselines(
            n_iterations=n_iterations,
            init_training_data_size=init_training_data_size,
            init_calibration_data_size=init_calibration_data_size,
            alphas=alphas,
            verbose=False,  # Don't print iteration details for each trial
            plot_every=plot_every,
            use_gaussian_clip=use_gaussian_clip,
            n_grid=n_grid,
            output_dir=output_dir,
        )

        all_results.append(results)

        if verbose:
            print(f"\nTrial {trial+1} Summary:")
            if "bo_configs" in results:
                for bo_config in results["bo_configs"]:
                    if bo_config["CPC"]:
                        label = "CPC+CBO," + r"$\alpha$=" + f"{bo_config['alpha']}"
                    else:
                        label = "CBO (no CPC)"
                    print(f"  {label:25s}: Best = {bo_config['best_f']:.4f}, Violations = {bo_config['violations'][-1]}")

            else:
                print(f"  BO: Best = {results['best_f_bo']:.4f}, Violations = {results['violations_bo'][-1]}")
            print(f"  Uniform Random: Best = {results['best_f_uniform']:.4f}, Violations = {results['violations_uniform'][-1]}")
            print(f"  Gaussian Random: Best = {results['best_f_gaussian']:.4f}, Violations = {results['violations_gaussian'][-1]}")

            if True in [bo_config["CPC"] for bo_config in results["bo_configs"]]:
                print(f"\n  CPC selected betas : {results['betas_selected']}")
            

    aggregated = aggregate_trial_results(all_results)
    aggregated["n_trials"] = n_trials
    aggregated["seeds"] = list(range(seed_start, seed_start + n_trials))

    return aggregated


def aggregate_trial_results(all_results):
    """Aggregate results from multiple trials (mean, std across trials).

    Per-round constraint violations are reported as a *rate* in [0, 1]: the raw
    count of infeasible points added in a round is divided by the number of
    points drawn that round (``BATCH_SIZE``). E.g. 3 violations out of 4 sampled
    points contributes 0.75, not 3. This keeps the metric comparable across
    batch sizes (and within the [0, 1] axis limits used in the plots).
    """
    n_trials = len(all_results)

    # Number of new samples drawn per round, used to convert per-round violation
    # counts into per-round violation rates.
    samples_per_round = BATCH_SIZE

    # Determine BO configurations from first trial
    if "bo_configs" in all_results[0]:
        n_bo_configs = len(all_results[0]["bo_configs"])
        bo_config_template = all_results[0]["bo_configs"]
    else:
        n_bo_configs = 1
        bo_config_template = [{"CPC": all_results[0].get("use_gaussian_clip", False),
                               "alpha": all_results[0].get("alpha")}]

    aggregated = {
        "bo_configs": [],
        "baselines": {
            "uniform": {"regrets": [], "violations": [], "violations_per_round": []},
            "gaussian": {"regrets": [], "violations": [], "violations_per_round": []},
        },
    }

    # Aggregate each BO configuration
    for config_idx in range(n_bo_configs):
        regrets_all = []
        violations_all = []
        violations_per_round_all = []

        for trial_results in all_results:
            if "bo_configs" in trial_results:
                config = trial_results["bo_configs"][config_idx]
                regrets_all.append(config["regrets"])
                violations_all.append(config["violations"])

                violations_per_round = [config["violations"][i + 1] - config["violations"][i]
                                        for i in range(len(config["violations"]) - 1)]
                violations_per_round_all.append(violations_per_round)
            else:
                regrets_all.append(trial_results["regrets_bo"])
                violations_all.append(trial_results["violations_bo"])

                violations_per_round = [trial_results["violations_bo"][i + 1] - trial_results["violations_bo"][i]
                                        for i in range(len(trial_results["violations_bo"]) - 1)]
                violations_per_round_all.append(violations_per_round)

        regrets_array = np.array(regrets_all)  # [n_trials, n_iterations+1]
        violations_array = np.array(violations_all)
        # Convert per-round violation counts to per-round violation rates in [0, 1].
        violations_per_round_array = np.array(violations_per_round_all) / samples_per_round  # [n_trials, n_iterations]

        aggregated["bo_configs"].append({
            "CPC": bo_config_template[config_idx]["CPC"],
            "alpha": bo_config_template[config_idx]["alpha"],
            "regrets_mean": regrets_array.mean(axis=0),
            "regrets_std": regrets_array.std(axis=0),
            "violations_mean": violations_array.mean(axis=0),
            "violations_std": violations_array.std(axis=0),
            "violations_per_round_mean": violations_per_round_array.mean(axis=0),
            "violations_per_round_std": violations_per_round_array.std(axis=0),
            "best_f_mean": np.mean([np.ma.masked_array((trial_results["bo_configs"][config_idx]["train_obj"] if "bo_configs" in trial_results else trial_results["train_obj_bo"]).cpu().numpy(),
                                                       mask=~(trial_results["bo_configs"][config_idx]["train_con"] if "bo_configs" in trial_results else trial_results["train_con_bo"]).bool().cpu().numpy()).max().item()
                                    for trial_results in all_results]),
        })

    # Aggregate baselines
    for baseline_name, baseline_key in [("uniform", "uniform"), ("gaussian", "gaussian")]:
        regrets_all = [trial[f"regrets_{baseline_key}"] for trial in all_results]
        violations_all = [trial[f"violations_{baseline_key}"] for trial in all_results]

        violations_per_round_all = []
        for violations in violations_all:
            violations_per_round = [violations[i + 1] - violations[i] for i in range(len(violations) - 1)]
            violations_per_round_all.append(violations_per_round)

        regrets_array = np.array(regrets_all)
        violations_array = np.array(violations_all)
        # Convert per-round violation counts to per-round violation rates in [0, 1].
        violations_per_round_array = np.array(violations_per_round_all) / samples_per_round

        aggregated["baselines"][baseline_name] = {
            "regrets_mean": regrets_array.mean(axis=0),
            "regrets_std": regrets_array.std(axis=0),
            "violations_mean": violations_array.mean(axis=0),
            "violations_std": violations_array.std(axis=0),
            "violations_per_round_mean": violations_per_round_array.mean(axis=0),
            "violations_per_round_std": violations_per_round_array.std(axis=0),
            "best_f_mean": np.mean([np.ma.masked_array(trial[f"train_obj_{baseline_key}"].cpu().numpy(),
                                                       mask=~trial[f"train_con_{baseline_key}"].bool().cpu().numpy()).max().item()
                                    for trial in all_results]),
        }

    return aggregated


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CPC + Constrained BO experiments on the Townsend problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core experiment hyperparameters.
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of independent trials to average over.")
    parser.add_argument("--n-iterations", type=int, default=10,
                        help="Number of BO iterations per trial.")
    parser.add_argument("--temperature", "--temperatures", dest="temperatures",
                        type=float, nargs="+", default=[0.1],
                        help="Softmax temperature(s). One experiment is run per value.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.4],
                        help="Risk levels; alpha<1.0 enables CPC for that config.")
    parser.add_argument("--init-training-data-size", type=int, default=20,
                        help="Number of initial random training data samples per trial.")
    parser.add_argument("--init-calibration-data-size", type=int, default=20,
                        help="Number of initial random calibration data samples per trial.")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Starting random seed (trial t uses seed_start+t).")

    # Secondary knobs (previously module-level constants).
    parser.add_argument("--num-candidate-samples", type=int, default=1000,
                        help="Monte Carlo candidate samples for sampling/normalization.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of new training points added per iteration for baselines \
                        (same number of new calibration points are added each iteration).")
    parser.add_argument("--num-restarts", type=int, default=10,
                        help="Restarts for optimize_acqf (unused by default loop).")
    parser.add_argument("--raw-samples", type=int, default=512,
                        help="Raw samples for optimize_acqf (unused by default loop).")
    parser.add_argument("--gaussian-std-div-factor", type=float, default=6.5,
                        help="Divisor of the search range for the safe Gaussian std.")
    parser.add_argument("--gaussian-mean", type=float, default=-1.0,
                        help="Vertical axis Gaussian mean (for safe policy).")
    parser.add_argument("--n-grid", type=int, default=250,
                        help="Approximate size of the beta grid searched in cpc_search.")

    # Plotting / output / device.
    parser.add_argument("--plot-every", type=int, default=None,
                        help="Save per-iteration diagnostic plots every N iters (None to disable).")
    parser.add_argument("--std-multiplier", type=float, default=1.0,
                        help="Multiplier for the shaded standard-error bands.")
    parser.add_argument("--no-std", action="store_true",
                        help="Disable shaded standard-error bands in the summary plots.")
    parser.add_argument("--no-gaussian-baseline-plot", action="store_true",
                        help="Don't plot the Gaussian Random Sampling baseline (it is still "
                             "computed and included in the printed summary / saved CSVs).")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory where figures and result files are written.")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device override, e.g. 'cpu' or 'cuda:0' (default: auto).")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce per-trial console output.")

    return parser.parse_args(argv)


def main(argv=None):
    global TEMPERATURE, NUM_CANDIDATE_SAMPLES, NUM_CANDIDATE_SAMPLES_CPC_TEST_WEIGHT, BATCH_SIZE
    global NUM_RESTARTS, RAW_SAMPLES, CPC_ALPHA, GAUSSIAN_STD_DIV_FACTOR, GAUSSIAN_MEAN, SETTING, townsend

    args = parse_args(argv)

    # Apply device override (rebuild the problem so its bounds live on the device).
    if args.device is not None:
        tkwargs["device"] = torch.device(args.device)
    townsend = Townsend()

    # Push CLI hyperparameters into the module-level constants used by functions.
    NUM_CANDIDATE_SAMPLES = args.num_candidate_samples
    BATCH_SIZE = args.batch_size
    NUM_RESTARTS = args.num_restarts
    RAW_SAMPLES = args.raw_samples
    GAUSSIAN_STD_DIV_FACTOR = args.gaussian_std_div_factor
    GAUSSIAN_MEAN = torch.tensor([0, args.gaussian_mean], **tkwargs)
    CPC_ALPHA = args.alphas[-1]
    TEMPERATURE = args.temperatures[0]

    # SETTING = f"temp{TEMPERATURE}_alpha{CPC_ALPHA}_InitTrainN{args.init_training_data_size}_InitCalN{args.init_calibration_data_size}_nTrials{args.n_trials}_nIter{args.n_iterations}_batchSize{BATCH_SIZE}_GaussianStdDivFactor{GAUSSIAN_STD_DIV_FACTOR}_GaussianMean{GAUSSIAN_MEAN[-1]}"
    SETTING = f"temp{TEMPERATURE}_alpha{CPC_ALPHA}_nTrials{args.n_trials}"
    print(f"Setting: {SETTING}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {tkwargs['device']}")
    print(f"Temperatures: {args.temperatures}")
    print(f"n_trials={args.n_trials}, n_iterations={args.n_iterations}, alphas={args.alphas}")

    aggregated_by_temp = {}
    for temperature in args.temperatures:
        TEMPERATURE = temperature  # read as a global by the acquisition/sampling helpers
        print(f"\n{'#'*70}")
        print(f"# Running experiment at TEMPERATURE = {temperature}")
        print(f"{'#'*70}")

        aggregated = run_multiple_trials(
            n_trials=args.n_trials,
            n_iterations=args.n_iterations,
            init_training_data_size=args.init_training_data_size,
            init_calibration_data_size=args.init_calibration_data_size,
            alphas=args.alphas,
            verbose=not args.quiet,
            plot_every=args.plot_every,
            seed_start=args.seed_start,
            n_grid=args.n_grid,
            output_dir=args.output_dir,
        )

        plot_averaged_results(
            aggregated,
            show_std=not args.no_std,
            std_multiplier=args.std_multiplier,
            output_dir=args.output_dir,
            include_gaussian=not args.no_gaussian_baseline_plot,
        )

        save_averaged_results_to_csv(aggregated, output_dir=args.output_dir)

        results_path = os.path.join(args.output_dir, f"aggregated_{SETTING}.pt")
        torch.save(aggregated, results_path)
        print(f"Saved aggregated results to {results_path}")

        aggregated_by_temp[temperature] = aggregated

    print("\nDone.")
    return aggregated_by_temp


if __name__ == "__main__":
    main()
