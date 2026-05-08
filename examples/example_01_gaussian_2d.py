"""
Example 1: correlated 2D Gaussian posterior.

Run:

    examples/example_01_gaussian_2d.py

This test case is to recover the 2D gaussian with 0 mean 2D gaussian with std 1 and 1.414 and covariance 0.8
See this below in true_cov. 
So, the MH algorithm should recover this. 
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

    true_cov = np.array(
        [
            [1.0, 0.8],
            [0.8, 2.0],
        ]
    )

    inv_cov = np.linalg.inv(true_cov)

    def log_posterior(theta):
        theta = np.asarray(theta, dtype=float)
        return -0.5 * theta @ inv_cov @ theta

    sampler = AdaptiveMetropolisSampler(
        log_posterior=log_posterior,
        initial_cov=0.1 * np.eye(2),
        parameter_names=parameter_names,
        rng=np.random.default_rng(123),
    )

    result = sampler.run(
        x0=np.array([3.0, -3.0]),
        n_samples=20_000,
        burn_in=5_000,
        adapt_until=5_000,
        adapt_interval=100,
        start_adapt=500,
        progress=True,
    )

    print("\nAcceptance rate:", result.acceptance_rate)
    print("Post-burn acceptance rate:", result.post_burn_acceptance_rate)

    samples = result.post_burn_samples

    diag = MCMCDiagnostics(samples, parameter_names=parameter_names)
    diag.print_summary()

    plotter = MCMCPlotter(samples, parameter_names=parameter_names)
    plotter.trace()
    plotter.acf(max_lag=100)
    plotter.marginals()
    plotter.corner()


if __name__ == "__main__":
    main()
