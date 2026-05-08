"""
Example 2: bimodal Gaussian mixture posterior.

This example shows a limitation of local random-walk Metropolis:
the chain may struggle to move between separated modes.

Run:

    examples/example_02_bimodal_gaussian.py

To see the implications of the initial point - 
Change the starting point x0 and reduce number of burn-in samples
You should see that it fails to identify the bimodal behaviour. 
You have tinker with it a bit. 
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "src"

sys.path.insert(0, str(MODULE_DIR))

from sampler import AdaptiveMetropolisSampler
from diagnostics import MCMCDiagnostics
from plotting import MCMCPlotter


Array = np.ndarray


def logsumexp(a: Array) -> float:
    """Stable log(sum(exp(a)))."""
    a = np.asarray(a, dtype=float)
    amax = np.max(a)
    return float(amax + np.log(np.sum(np.exp(a - amax))))


def make_log_gaussian(mu: Array, cov: Array):
    """
    Return a function evaluating log N(theta | mu, cov).
    """

    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    dim = len(mu)
    inv_cov = np.linalg.inv(cov)
    sign, logdet_cov = np.linalg.slogdet(cov)

    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    def log_gaussian(theta: Array) -> float:
        theta = np.asarray(theta, dtype=float)
        r = theta - mu
        return float(
            -0.5
            * (
                dim * np.log(2.0 * np.pi)
                + logdet_cov
                + r @ inv_cov @ r
            )
        )

    return log_gaussian


def make_bimodal_log_posterior():
    """
    Create bimodal Gaussian mixture log-posterior.
    """

    mu1 = np.array([-3.0, -3.0])
    mu2 = np.array([3.0, 3.0])

    cov1 = np.array(
        [
            [1.0, 0.75],
            [0.75, 1.0],
        ]
    )

    cov2 = np.array(
        [
            [1.0, -0.65],
            [-0.65, 1.0],
        ]
    )

    log_gaussian_1 = make_log_gaussian(mu1, cov1)
    log_gaussian_2 = make_log_gaussian(mu2, cov2)

    w1 = 0.5
    w2 = 0.5

    def log_posterior(theta: Array) -> float:
        logp1 = np.log(w1) + log_gaussian_1(theta)
        logp2 = np.log(w2) + log_gaussian_2(theta)
        return logsumexp(np.array([logp1, logp2]))

    return log_posterior


def main() -> None:
    parameter_names = ["theta_0", "theta_1"]

    log_posterior = make_bimodal_log_posterior()

    #x0 = np.array([0.0, 0.0])
    x0 = np.array([-3.0, -3.0])
    #x0 = np.array([0.0, 0.0])

    sampler = AdaptiveMetropolisSampler(
        log_posterior=log_posterior,
        initial_cov=0.1 * np.eye(2),
        parameter_names=parameter_names,
        rng=np.random.default_rng(123),
    )

    result = sampler.run(
        x0=x0,
        n_samples=10000,
        burn_in=1000,
        adapt_until=1000,
        adapt_interval=300,
        start_adapt=500,
        progress=True,
    )

    print("\nAcceptance rate:", result.acceptance_rate)
    print("Post-burn acceptance rate:", result.post_burn_acceptance_rate)

    samples = result.post_burn_samples

    left_mode_fraction = np.mean(samples[:, 0] < 0.0)
    right_mode_fraction = np.mean(samples[:, 0] >= 0.0)

    print("\nApproximate mode occupancy:")
    print(f"left mode  fraction: {left_mode_fraction:.3f}")
    print(f"right mode fraction: {right_mode_fraction:.3f}")

    diag = MCMCDiagnostics(samples, parameter_names=parameter_names)
    diag.print_summary()

    plotter = MCMCPlotter(samples, parameter_names=parameter_names)
    plotter.trace()
    plotter.acf(max_lag=200)
    plotter.marginals(bins=50)
    plotter.corner(bins=50, max_points=10_000)


if __name__ == "__main__":
    main()
