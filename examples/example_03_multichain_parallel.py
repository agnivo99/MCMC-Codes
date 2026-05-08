"""
Example 3: multiple chains in parallel.

Direct implementation of multiple chains in parallel 
Use multiprocessing to load separate chains in separate processes 
Finally combine them and see the results. 
Check the Rhat for each chain to understand how much important initial point is.

Run:

    examples/example_03_multichain_parallel.py
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "src"

sys.path.insert(0, str(MODULE_DIR))

from sampler import AdaptiveMetropolisSampler
from diagnostics import MCMCDiagnostics
from plotting import MCMCPlotter


N_CHAINS = 4
N_SAMPLES = 20000
BURN_IN = 5000
SEED = 123

PARAMETER_NAMES = ["theta_0", "theta_1"]


def log_posterior(theta):
    """
    Standard 2D Gaussian target.
    """

    theta = np.asarray(theta, dtype=float)
    return -0.5 * theta @ theta


def run_chain(args):
    """
    Run one chain in a separate process.
    """

    chain_id, seed, x0 = args

    sampler = AdaptiveMetropolisSampler(
        log_posterior=log_posterior,
        initial_cov=0.5 * np.eye(2),
        parameter_names=PARAMETER_NAMES,
        rng=np.random.default_rng(seed),
    )

    result = sampler.run(
        x0=x0,
        n_samples=N_SAMPLES,
        burn_in=BURN_IN,
        adapt_until=BURN_IN,
        adapt_interval=100,
        start_adapt=500,
        progress=False,
    )

    return {
        "chain_id": chain_id,
        "samples": result.post_burn_samples,
        "acceptance_rate": result.acceptance_rate,
        "post_burn_acceptance_rate": result.post_burn_acceptance_rate,
        "x0": x0,
    }


def main():
    rng = np.random.default_rng(SEED)

    args = []

    for c in range(N_CHAINS):
        chain_id = c + 1
        seed = SEED + 1000 * chain_id

        # Different starting point for each chain.
        x0 = rng.normal(0.0, 5.0, size=2)

        args.append((chain_id, seed, x0))

    results = []

    with ProcessPoolExecutor(max_workers=N_CHAINS) as executor:
        futures = [executor.submit(run_chain, a) for a in args]

        for future in as_completed(futures):
            results.append(future.result())

    results = sorted(results, key=lambda r: r["chain_id"])

    chains = np.array([r["samples"] for r in results])

    print("chains.shape:", chains.shape)

    print("\nAcceptance rates:")
    for r in results:
        print(
            f"Chain {r['chain_id']}: "
            f"overall={r['acceptance_rate']:.3f}, "
            f"post-burn={r['post_burn_acceptance_rate']:.3f}"
        )

    diag = MCMCDiagnostics(chains, parameter_names=PARAMETER_NAMES)
    diag.print_summary()

    plotter = MCMCPlotter(chains, parameter_names=PARAMETER_NAMES, truths=np.zeros(2))

    chain_dict = {
        f"Chain {i + 1}": chains[i]
        for i in range(chains.shape[0])
    }

    plotter.trace(max_points := None)  # see note below
    plotter.compare_marginals(chain_dict, kde=True)
    plotter.corner()


if __name__ == "__main__":
    main()
