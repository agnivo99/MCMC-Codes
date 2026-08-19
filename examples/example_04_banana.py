"""
Example 4: banana-shaped 2D posterior.

Run:

    examples/example_04_banana_posterior.py

This test case considers a nonlinear, non-Gaussian posterior with a
banana-shaped geometry.

The target distribution is defined as

    theta_0 ~ N(0, sigma_x^2)

    theta_1 | theta_0 ~
        N(b * (theta_0^2 - sigma_x^2), sigma_y^2)

where b controls the curvature of the banana.

Unlike the correlated Gaussian example, the dependence between the
parameters is nonlinear. Therefore, a single covariance matrix cannot
completely describe the geometry of the target posterior.

This provides a more challenging test for the Adaptive Metropolis
sampler and is useful for checking whether the chain can explore a
curved, non-Gaussian posterior distribution.
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
    sigma_x = 1.0
    sigma_y = 0.5

    # Higher |b| means higher bending and vice versa.
    # The direction of bending depends on the sign of b
    # Positive b means concave upwards
    # Negative b means concave downwards

    b = -0.7 # Curvature of the banana shape controller

    def log_posterior(theta):
        """
        Banana-shaped posterior:

            theta_0 ~ N(0, sigma_x^2)

            theta_1 | theta_0
                ~ N(b * (theta_0^2 - sigma_x^2), sigma_y^2)
        """

        theta = np.asarray(theta, dtype=float)

        x = theta[0]
        y = theta[1]

        # Conditional mean of y
        mean_y = b * (x**2 - sigma_x**2)

        logp_x = -0.5 * (x / sigma_x) ** 2

        logp_y_given_x = (
            -0.5
            * ((y - mean_y) / sigma_y) ** 2
        )

        return logp_x + logp_y_given_x
    
    sampler = AdaptiveMetropolisSampler(
        log_posterior=log_posterior,
        initial_cov=0.1 * np.eye(2),
        parameter_names=parameter_names,
        rng=np.random.default_rng(123),
    )
    result = sampler.run(
        x0=np.array([2.5, -2.0]),
        n_samples=60_000,
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
    plotter.trace(begin=10000, end=20000)
    plotter.acf(
        max_lag=200,
    )
    plotter.marginals()
    plotter.corner()


if __name__ == "__main__":
    main()