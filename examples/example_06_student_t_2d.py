"""
Example 6: correlated 2D Student-t posterior.

Run:

    examples/example_06_student_t_2d.py

This test case considers a heavy-tailed 2D posterior.

The target distribution is a multivariate Student-t distribution with

    mean = [0, 0]

    scale matrix =
        [[1.0, 0.8],
         [0.8, 2.0]]

and degrees of freedom

    nu = 3

Compared with a Gaussian distribution, the Student-t distribution has
heavier tails. This provides a useful test of whether the Adaptive
Metropolis sampler can adequately explore posterior distributions with
substantial probability away from the central high-density region.

As nu increases, the Student-t distribution approaches a Gaussian.
"""

import numpy as np
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "src"

sys.path.insert(0, str(MODULE_DIR))

from sampler import AdaptiveMetropolisSampler
from diagnostics import MCMCDiagnostics
from plotting import MCMCPlotter


def main():

    parameter_names = ["theta_0", "theta_1"]
    nu = 2.0

    scale_matrix = np.array(
        [
            [1.0, 0.8],
            [0.8, 2.0],
        ]
    )

    inv_scale = np.linalg.inv(scale_matrix)

    d = 2


    def log_posterior(theta):
        """
        Multivariate Student-t posterior:

            p(theta) proportional to

            [1 + (theta^T Sigma^{-1} theta)/nu]
            ^ [-(nu + d)/2]

        Constants independent of theta are omitted.
        """

        theta = np.asarray(theta, dtype=float)

        mahalanobis_sq = theta @ inv_scale @ theta

        logp = (
            -0.5
            * (nu + d)
            * np.log1p(mahalanobis_sq / nu)
        )

        return logp

    sampler = AdaptiveMetropolisSampler(
        log_posterior=log_posterior,
        initial_cov=0.1 * np.eye(2),
        parameter_names=parameter_names,
        rng=np.random.default_rng(123),
    )
    result = sampler.run(
        x0=np.array([3.0, -3.0]),
        n_samples=50_000,
        burn_in=10_000,
        adapt_until=10_000,
        adapt_interval=100,
        start_adapt=500,
        progress=True,
    )
    print("\nAcceptance rate:", result.acceptance_rate)

    print(
        "Post-burn acceptance rate:",
        result.post_burn_acceptance_rate,
    )
    samples = result.post_burn_samples
    diag = MCMCDiagnostics(
        samples,
        parameter_names=parameter_names,
    )

    diag.print_summary()
    plotter = MCMCPlotter(
        samples,
        parameter_names=parameter_names,
    )
    plotter.trace()
    plotter.acf(
        max_lag=200,
    )
    plotter.marginals()
    plotter.corner()

if __name__ == "__main__":
    main()