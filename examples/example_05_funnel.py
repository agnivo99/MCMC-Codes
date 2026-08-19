"""
Example 5: Neal's funnel posterior.

Run:

    examples/example_05_funnel_posterior.py

This test case considers a strongly non-Gaussian posterior with
position-dependent scale.

The target distribution is defined as

    v ~ N(0, 3^2)

    x | v ~ N(0, exp(v))

Equivalently, the conditional standard deviation of x is

    sigma_x = exp(v / 2)

For negative v, the posterior becomes very narrow in x.
For positive v, the posterior becomes very broad.

This produces the characteristic funnel-shaped geometry and provides
a challenging test for a random-walk Metropolis sampler because a
single global proposal covariance cannot simultaneously match both
the narrow and broad regions of the target distribution.
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

    parameter_names = ["v", "x"]
    sigma_v = 10.0


    def log_posterior(theta):
        """
        Neal's funnel:

            v ~ N(0, sigma_v^2)

            x | v ~ N(0, exp(v))

        Therefore

            std(x | v) = exp(v / 2)
        """

        theta = np.asarray(theta, dtype=float)

        v = theta[0]
        x = theta[1]

        # Prior / marginal contribution for v
        logp_v = -0.5 * (v / sigma_v) ** 2

        # Conditional variance of x
        var_x = np.exp(v)

        # Include the Gaussian normalization term because
        # the variance depends on v.
        logp_x_given_v = (
            -0.5 * v
            -0.5 * x**2 / var_x
        )

        return logp_v + logp_x_given_v
    
    sampler = AdaptiveMetropolisSampler(
        log_posterior=log_posterior,
        initial_cov=0.1 * np.eye(2),
        parameter_names=parameter_names,
        rng=np.random.default_rng(123),
    )
    result = sampler.run(
        x0=np.array([0.0, 0.0]),
        n_samples=60_000,
        burn_in=15_000,
        adapt_until=15_000,
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