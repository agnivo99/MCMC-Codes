:# Adaptive Metropolis-Hastings MCMC

This repository contains a modular implementation of an Adaptive Random-Walk Metropolis-Hastings sampler with diagnostics and plotting utilities.

## Features

- Adaptive proposal covariance during burn-in
- Random-Walk Metropolis-Hastings sampler
- Multiple-chain diagnostics
- Effective sample size
- Split-chain R-hat
- Trace plots
- Autocorrelation plots
- Marginal posterior plots
- Corner plots
- Example test cases including Gaussian, bimodal Gaussian, and multi-chain parallel sampling

## Repository Structure

```text
Adaptive_MH/
├── src/
│   ├── sampler.py
│   ├── diagnostics.py
│   └── plotting.py
│
├── examples/
│   ├── example_01_gaussian_2d.py
│   ├── example_02_bimodal_gaussian.py
│   └── example_03_multichain_parallel.py
│
├── figures/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository and switch to the Adaptive-MH branch:

```bash
git clone git@github.com:agnivo99/MCMC-Codes.git
cd MCMC-Codes
git checkout Adaptive-MH
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The examples add the `src/` folder to the Python path and then import the modules directly:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sampler import AdaptiveMetropolisSampler
from diagnostics import MCMCDiagnostics
from plotting import MCMCPlotter
```

This means users can clone the repository and run the examples without installing the project as a Python package.

## Example 1: Correlated Gaussian

Run:

```bash
python examples/example_01_gaussian_2d.py
```

This samples from a 2D Gaussian target:

```math
\theta \sim \mathcal{N}(0, \Sigma)
```

with covariance

```math
\Sigma =
\begin{bmatrix}
1.0 & 0.8 \\
0.8 & 2.0
\end{bmatrix}.
```

The true posterior mean is:

```math
\mu =
\begin{bmatrix}
0 \\
0
\end{bmatrix}.
```

This example is useful for checking whether the sampler correctly captures a correlated unimodal posterior.

## Example 2: Bimodal Gaussian Mixture

Run:

```bash
python examples/example_02_bimodal_gaussian.py
```

This example samples from a bimodal target distribution:

```math
p(\theta)
=
0.5\mathcal{N}(\theta \mid \mu_1, C_1)
+
0.5\mathcal{N}(\theta \mid \mu_2, C_2).
```

This example demonstrates an important limitation of local random-walk Metropolis samplers: if the modes are well separated, the chain may get trapped in one mode.

A well-mixed chain should visit both modes. A poorly mixed chain may show mode occupancy like:

```text
left mode  fraction: 0.99
right mode fraction: 0.01
```

This is not a code failure; it is a known limitation of local random-walk proposals on multimodal targets.

## Example 3: Multiple Chains in Parallel

Run:

```bash
python examples/example_03_multichain_parallel.py
```

This example launches multiple independent chains in parallel using Python multiprocessing.

It demonstrates:

- different starting points for each chain
- different random seeds for each chain
- chain-wise marginal posterior comparison
- multi-chain R-hat
- effective sample size
- combined corner plots

This is the recommended workflow for checking whether independent chains are sampling the same posterior distribution.

## Basic Sampler Interface

Define a log-posterior function:

```python
def log_posterior(theta):
    return log_prior(theta) + log_likelihood(theta)
```

Then run:

```python
import numpy as np

sampler = AdaptiveMetropolisSampler(
    log_posterior=log_posterior,
    initial_cov=0.1 * np.eye(2),
    parameter_names=["theta_0", "theta_1"],
    rng=np.random.default_rng(123),
)

result = sampler.run(
    x0=np.array([2.0, -2.0]),
    n_samples=20_000,
    burn_in=5_000,
    adapt_until=5_000,
)
```

Use post-burn samples:

```python
samples = result.post_burn_samples
```

## Diagnostics

```python
diag = MCMCDiagnostics(
    samples,
    parameter_names=["theta_0", "theta_1"],
)

diag.print_summary()
```

The summary includes:

- posterior mean
- posterior standard deviation
- credible interval
- effective sample size
- R-hat

For R-hat, use multiple chains with shape:

```text
(n_chains, n_samples, n_parameters)
```

## Plotting

```python
plotter = MCMCPlotter(
    samples,
    parameter_names=["theta_0", "theta_1"],
    truths=np.array([0.0, 0.0]),
)

plotter.trace()
plotter.acf(max_lag=100)
plotter.marginals()
plotter.corner()
```

For chain-wise marginal comparison:

```python
chain_dict = {
    "Chain 1": chains[0],
    "Chain 2": chains[1],
    "Chain 3": chains[2],
    "Chain 4": chains[3],
}

plotter.compare_marginals(
    other_samples_dict=chain_dict,
    kde=True,
)
```

## Adaptive Metropolis Method

The proposal is:

```math
\theta^\star = \theta^{(n)} + \epsilon
```

where

```math
\epsilon \sim \mathcal{N}(0, s^2 C).
```

The default scaling is:

```math
s = \frac{2.38}{\sqrt{d}},
```

where `d` is the number of parameters.

During burn-in, the proposal covariance `C` is adapted using the empirical covariance of the chain. After burn-in, the covariance is frozen.

For most applications, use:

```python
adapt_until = burn_in
```

## Notes

- Low acceptance rate usually means the proposal covariance is too large.
- Very high acceptance rate usually means the proposal covariance is too small.
- A rough target for moderate/high-dimensional random-walk Metropolis is often around 0.15–0.35.
- Adaptive covariance improves local sampling but does not solve strong multimodality.
- For strongly multimodal targets, consider multiple chains, global proposals, or tempering methods.


