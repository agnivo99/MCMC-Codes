# Adaptive Metropolis-Hastings MCMC

This repository contains a modular implementation of an Adaptive Random-Walk Metropolis-Hastings sampler together with convergence diagnostics, plotting utilities, and several example posterior distributions for testing sampler behavior.

The implementation is intended to provide a simple and reusable framework for Bayesian inference problems in which the posterior distribution is available up to an unknown normalization constant.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Test Cases](#example-test-cases)
  - [Example 1: Correlated 2D Gaussian](#example-1-correlated-2d-gaussian)
  - [Example 2: Bimodal Gaussian Mixture](#example-2-bimodal-gaussian-mixture)
  - [Example 3: Multiple Chains in Parallel](#example-3-multiple-chains-in-parallel)
  - [Example 4: Banana-Shaped Posterior](#example-4-banana-shaped-posterior)
  - [Example 5: Neal's Funnel Posterior](#example-5-neals-funnel-posterior)
  - [Example 6: Correlated 2D Student-t Posterior](#example-6-correlated-2d-student-t-posterior)
- [Basic Sampler Interface](#basic-sampler-interface)
- [Adaptive Metropolis Method](#adaptive-metropolis-method)
  - [Metropolis-Hastings Acceptance Step](#metropolis-hastings-acceptance-step)
  - [Adaptive Proposal Covariance](#adaptive-proposal-covariance)
  - [Adaptation Controls](#adaptation-controls)
  - [Initial Proposal Covariance](#initial-proposal-covariance)
- [Diagnostics](#diagnostics)
  - [Effective Sample Size](#effective-sample-size)
  - [Split-Chain R-hat](#split-chain-r-hat)
- [Plotting](#plotting)
  - [Trace Plots](#trace-plots)
  - [Autocorrelation Plots](#autocorrelation-plots)
  - [Marginal Posterior Plots](#marginal-posterior-plots)
  - [Corner Plots](#corner-plots)
  - [Chain-Wise Marginal Comparison](#chain-wise-marginal-comparison)
- [Recommended Workflow](#recommended-workflow)
- [Notes](#notes)
- [Related Publication](#related-publication)

## Features

* Adaptive proposal covariance during burn-in
* Random-Walk Metropolis-Hastings sampling
* User-defined log-posterior functions
* Multiple-chain sampling
* Parallel execution of independent chains
* Effective sample size
* Split-chain (\hat{R})
* Trace plots
* Autocorrelation plots
* Marginal posterior plots
* Corner plots
* Comparison of marginal posteriors across chains
* Support for synthetic truth values in posterior plots
* Example test cases including:

  * correlated Gaussian posterior
  * bimodal Gaussian mixture
  * multiple-chain parallel sampling
  * banana-shaped posterior
  * Neal's funnel posterior
  * heavy-tailed Student-(t) posterior

---

## Repository Structure

```text
MCMC-Codes/
├── src/
│   ├── sampler.py
│   ├── diagnostics.py
│   └── plotting.py
│
├── examples/
│   ├── example_01_gaussian_2d.py
│   ├── example_02_bimodal_gaussian.py
│   ├── example_03_multichain_parallel.py
│   ├── example_04_banana_posterior.py
│   ├── example_05_funnel_posterior.py
│   └── example_06_student_t_2d.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository and switch branch:

```bash
git clone https://github.com/Stochastic-Hypersonics-Research-Group/MCMC-Codes.git
cd MCMC-Codes
git checkout Adaptive-MH
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

The examples add the `src/` directory directly to the Python path:

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

This allows the repository to be cloned and the example scripts to be executed directly without installing the project as a Python package.

---

# Example Test Cases

## Example 1: Correlated 2D Gaussian

Run:

```bash
python examples/example_01_gaussian_2d.py
```

This example samples from a correlated two-dimensional Gaussian target,

```math
\boldsymbol{\theta}
\sim
\mathcal{N}
\left(
\boldsymbol{0},
\Sigma
\right),
```

with covariance

```math
\Sigma =
\begin{bmatrix}
1.0 & 0.8 \\
0.8 & 2.0
\end{bmatrix}.
```
---

## Example 2: Bimodal Gaussian Mixture

Run:

```bash
python examples/example_02_bimodal_gaussian.py
```

This example samples from a bimodal target distribution,

```math
p(\boldsymbol{\theta})
=
0.5
\mathcal{N}
\left(
\boldsymbol{\theta}
\mid
\boldsymbol{\mu}_1,
C_1
\right)
+
0.5
\mathcal{N}
\left(
\boldsymbol{\theta}
\mid
\boldsymbol{\mu}_2,
C_2
\right).
```

This example illustrates an important limitation of local random-walk Metropolis samplers.

When the modes are well separated, a chain initialized near one mode may have difficulty reaching the other mode. For example, a poorly mixed chain may show behavior such as

```text
left mode  fraction: 0.99
right mode fraction: 0.01
```

even when the target distribution assigns equal probability to the two modes.

This is not necessarily a failure of the implementation. It is a known limitation of local random-walk proposals for strongly multimodal target distributions.

Multiple chains initialized in different regions are therefore especially useful for this class of problem.

---

## Example 3: Multiple Chains in Parallel

Run:

```bash
python examples/example_03_multichain_parallel.py
```

This example launches multiple independent MCMC chains using Python multiprocessing.

It demonstrates:

* multiple initial conditions
* independent random seeds
* parallel chain execution
* chain-wise marginal posterior comparison
* combined posterior analysis
* effective sample size
* split-chain (\hat{R})
* combined corner plots

This is the recommended workflow when convergence assessment across independent chains is required.

For multiple chains, the expected sample-array shape is

```text
(n_chains, n_samples, n_parameters)
```

rather than

```text
(n_samples, n_parameters)
```

for a single chain.

---

## Example 4: Banana-Shaped Posterior

Run:

```bash
python examples/example_04_banana_posterior.py
```

This example considers a nonlinear and non-Gaussian posterior with a curved banana-shaped geometry.

The distribution is defined through

```math
\theta_0
\sim
\mathcal{N}
\left(
0,
\sigma_x^2
\right),
```

and

```math
\theta_1
\mid
\theta_0
\sim
\mathcal{N}
\left(
b
\left(
\theta_0^2-\sigma_x^2
\right),
\sigma_y^2
\right).
```

The parameter (b) controls the curvature of the target distribution.

Unlike the correlated Gaussian example, the dependence between the two parameters is nonlinear. Consequently, a single covariance matrix cannot completely describe the posterior geometry.

This example tests whether the sampler can explore a curved posterior distribution and demonstrates one of the limitations of covariance-based adaptation: the proposal covariance captures linear correlation but cannot perfectly represent nonlinear parameter dependence.

---

## Example 5: Neal's Funnel Posterior

Run:

```bash
python examples/example_05_funnel_posterior.py
```

This example samples from Neal's funnel distribution,

```math
v
\sim
\mathcal{N}
\left(
0,
3^2
\right),
```

with

```math
x
\mid
v
\sim
\mathcal{N}
\left(
0,
\exp(v)
\right).
```

The conditional standard deviation of (x) is therefore

```math
\sigma_x
=
\exp
\left(
\frac{v}{2}
\right).
```

For negative values of (v), the target becomes very narrow in the (x)-direction. For positive values of (v), the target becomes increasingly broad.

This produces the characteristic funnel geometry.

The funnel distribution is an intentionally difficult sampling problem because the appropriate proposal scale changes significantly throughout the target distribution. A single global covariance matrix cannot simultaneously represent both the narrow and broad regions efficiently.

This example can therefore expose:

* slow mixing
* large autocorrelation
* difficulty entering narrow regions
* sensitivity to the proposal covariance
* limitations of random-walk Metropolis sampling

Poor performance on this example does not necessarily indicate an implementation error. Neal's funnel is commonly used as a difficult benchmark for MCMC methods.

---

## Example 6: Correlated 2D Student-(t) Posterior

Run:

```bash
python examples/example_06_student_t_2d.py
```

This example considers a correlated multivariate Student-(t) posterior with degrees of freedom

```math
\nu = 3
```

and scale matrix

```math
\Sigma =
\begin{bmatrix}
1.0 & 0.8 \\
0.8 & 2.0
\end{bmatrix}.
```

The target density is proportional to

```math
p(\boldsymbol{\theta})
\propto
\left[
1+
\frac{
\boldsymbol{\theta}^{T}
\Sigma^{-1}
\boldsymbol{\theta}
}{
\nu
}
\right]^{
-(\nu+d)/2
},
```
---

# Basic Sampler Interface

The sampler requires a user-defined log-posterior function.

For a Bayesian inference problem,

```math
p(\boldsymbol{\theta}\mid\boldsymbol{y})
\propto
p(\boldsymbol{y}\mid\boldsymbol{\theta})
p(\boldsymbol{\theta}),
```

so the log-posterior can typically be written as

```python
def log_posterior(theta):
    return log_prior(theta) + log_likelihood(theta)
```

The normalization constant of the posterior is not required.

The sampler can then be initialized as

```python
import numpy as np

sampler = AdaptiveMetropolisSampler(
    log_posterior=log_posterior,
    initial_cov=0.1 * np.eye(2),
    parameter_names=["theta_0", "theta_1"],
    rng=np.random.default_rng(123),
)
```

and run using

```python
result = sampler.run(
    x0=np.array([2.0, -2.0]),
    n_samples=20000,
    burn_in=5000,
    adapt_until=5000,
    start_adapt=500,
    adapt_interval=100,
)
```

Post-burn-in samples can be accessed using

```python
samples = result.post_burn_samples
```

The sampling result also contains information such as the acceptance rate and final learned proposal covariance.

For example:

```python
print(result.acceptance_rate)
print(result.post_burn_acceptance_rate)
print(result.proposal_cov)
```

---

# Adaptive Metropolis Method

The sampler implements an Adaptive Random-Walk Metropolis-Hastings algorithm.

In a standard Random-Walk Metropolis-Hastings algorithm, a candidate state is proposed according to

```math
\boldsymbol{\theta}^{\star}
=
\boldsymbol{\theta}^{(n)}
+
\boldsymbol{\epsilon},
```

where

```math
\boldsymbol{\epsilon}
\sim
\mathcal{N}
\left(
\boldsymbol{0},
s^2 C_n
\right).
```

Here,

- $\boldsymbol{\theta}^{(n)}$ is the current Markov-chain state,
- $C_n$ is the proposal covariance matrix,
- $s$ is a scaling factor,
- $d$ is the dimension of the parameter space.

The default scaling used by the implementation is

```math
s
=
\frac{2.38}{\sqrt{d}},
```

or equivalently,

```math
s^2
=
\frac{2.38^2}{d}.
```

This scaling is commonly used for Gaussian random-walk proposals in moderate- and high-dimensional settings.

---

## Metropolis-Hastings Acceptance Step

For a symmetric Gaussian random-walk proposal, the candidate state is accepted with probability

```math
\alpha
=
\min
\left[
1,
\frac{
p(\boldsymbol{\theta}^{\star}\mid\boldsymbol{y})
}{
p(\boldsymbol{\theta}^{(n)}\mid\boldsymbol{y})
}
\right].
```

Because the implementation works with the log-posterior, the comparison is performed in logarithmic form for improved numerical stability.

If the proposed state is rejected, the chain remains at its current position.

---

## Adaptive Proposal Covariance

The primary difference between the Adaptive Metropolis algorithm and a standard fixed-proposal Metropolis algorithm is that the proposal covariance is updated during an initial adaptation stage.

The empirical covariance of the evolving Markov chain is used to progressively estimate the scale and correlation structure of the target posterior.

Conceptually, the adaptive covariance takes the form

```math
C_n
=
\mathrm{Cov}
\left(
\boldsymbol{\theta}^{(0)},
\ldots,
\boldsymbol{\theta}^{(n)}
\right)
+
\epsilon I
```

where ($\epsilon$ I) is a small diagonal regularization term used to help maintain a positive-definite covariance matrix.

The resulting proposal distribution therefore becomes

```math
q
\left(
\boldsymbol{\theta}^{\star}
\mid
\boldsymbol{\theta}^{(n)}
\right)
=
\mathcal{N}
\left(
\boldsymbol{\theta}^{(n)},
s^2C_n
\right).
```

The purpose of the adaptation is to allow the sampler to learn important posterior characteristics automatically.

For example, if two parameters are strongly correlated, the empirical covariance develops corresponding off-diagonal terms. The proposal distribution can then generate moves preferentially along the correlated direction rather than repeatedly proposing inefficient axis-aligned steps.

Similarly, when parameters have different posterior scales, the adaptive covariance can progressively adjust the proposal variance in each direction.

---

## Adaptation Controls

The adaptation behavior is controlled by several arguments in `sampler.run()`:

```python
result = sampler.run(
    x0=x0,
    n_samples=20000,
    burn_in=5000,
    adapt_until=5000,
    start_adapt=500,
    adapt_interval=100,
)
```

The main arguments are:

### `start_adapt`

Specifies the iteration after which covariance adaptation begins.

```python
start_adapt=500
```

The initial portion of the chain is therefore generated using the supplied initial covariance before empirical covariance adaptation begins.

---

### `adapt_interval`

Specifies how frequently the proposal covariance is updated.

```python
adapt_interval=100
```

For example, an interval of 100 means that the proposal covariance is updated every 100 iterations during the adaptation stage.

---

### `adapt_until`

Specifies the final iteration at which covariance adaptation is allowed.

```python
adapt_until=5000
```

After this point, the proposal covariance is frozen.

---

### `burn_in`

Specifies the number of initial MCMC samples discarded before posterior analysis.

```python
burn_in=5000
```

For most applications, a convenient choice is

```python
adapt_until = burn_in
```

so that covariance adaptation occurs during burn-in and the proposal covariance is fixed for the retained posterior samples.

---

### `n_samples`

Specifies the total number of MCMC iterations.

```python
n_samples=20000
```

The number of retained samples is therefore approximately

```text
n_samples - burn_in
```

when the post-burn samples are used.

---

## Initial Proposal Covariance

The initial proposal covariance is supplied when the sampler is created:

```python
sampler = AdaptiveMetropolisSampler(
    log_posterior=log_posterior,
    initial_cov=0.1 * np.eye(d),
    parameter_names=parameter_names,
    rng=np.random.default_rng(123),
)
```

The initial covariance does not need to accurately reproduce the target posterior covariance.

Its main purpose is to provide a reasonable proposal scale during the initial portion of the chain before enough samples are available for empirical covariance estimation.

However, an extremely poor initial covariance can still lead to inefficient early sampling.

---

# Diagnostics

Posterior diagnostics can be calculated using

```python
diag = MCMCDiagnostics(
    samples,
    parameter_names=["theta_0", "theta_1"],
)

diag.print_summary()
```

The diagnostic summary includes quantities such as:

* posterior mean
* posterior standard deviation
* credible intervals
* effective sample size
* split-chain ($\hat{R}$), when multiple chains are provided

For multiple chains, provide samples with shape

```text
(n_chains, n_samples, n_parameters)
```

For a single chain, samples may have shape

```text
(n_samples, n_parameters)
```

---

## Effective Sample Size

Successive MCMC samples are generally correlated.

Consequently, (N) retained MCMC samples do not necessarily contain the same amount of independent information as (N) independent samples.

The effective sample size provides an estimate of the equivalent number of approximately independent samples contained in the correlated Markov chain.

A low effective sample size relative to the total number of retained samples indicates strong autocorrelation and slow mixing.

---

## Split-Chain ($\hat{R}$)

When multiple chains are available, the split-chain (\hat{R}) diagnostic compares within-chain and between-chain variability.

Values close to one indicate that the chains are sampling statistically similar regions of parameter space.

Substantially larger values may indicate incomplete convergence or poor exploration of the target distribution.

---

# Plotting

Create a plotting object using

```python
plotter = MCMCPlotter(
    samples,
    parameter_names=["theta_0", "theta_1"],
    truths=np.array([0.0, 0.0]),
)
```

Available plots include:

```python
plotter.trace()
plotter.acf(max_lag=100)
plotter.marginals()
plotter.corner()
```

---

## Trace Plots

```python
plotter.trace()
```

Trace plots show the sampled parameter values as a function of MCMC iteration.

They are useful for identifying:

* long-term drift
* poor mixing
* trapping in individual modes
* non-stationary behavior
* different behavior across chains

---

## Autocorrelation Plots

```python
plotter.acf(max_lag=100)
```

The autocorrelation function shows the correlation between MCMC samples separated by different lag values.

Rapid decay of the autocorrelation generally indicates more efficient sampling.

Slow decay indicates that successive samples remain strongly correlated.

---

## Marginal Posterior Plots

```python
plotter.marginals()
```

Marginal posterior plots display the one-dimensional distribution of each selected parameter.

If synthetic truth values are provided, they can also be displayed for reference.

---

## Corner Plots

```python
plotter.corner()
```

Corner plots provide a compact visualization of the posterior distribution.

The diagonal panels show one-dimensional marginal posterior distributions, while the off-diagonal panels show pairwise joint posterior structure.

These plots are particularly useful for identifying parameter correlations and nonlinear dependencies.

---

## Chain-Wise Marginal Comparison

For multiple chains:

```python
chain_dict = {
    "Chain 1": chains[0],
    "Chain 2": chains[1],
    "Chain 3": chains[2],
    "Chain 4": chains[3],
}
```

The marginal distributions can be compared using

```python
plotter.compare_marginals(
    other_samples_dict=chain_dict,
    kde=True,
)
```

Agreement between the marginal distributions from independently initialized chains provides an additional qualitative convergence check.

---

# Recommended Workflow

A typical Bayesian inference calculation may follow the sequence:

```python
# 1. Define the log posterior
def log_posterior(theta):
    return log_prior(theta) + log_likelihood(theta)


# 2. Construct the sampler
sampler = AdaptiveMetropolisSampler(
    log_posterior=log_posterior,
    initial_cov=initial_cov,
    parameter_names=parameter_names,
    rng=np.random.default_rng(123),
)


# 3. Run MCMC
result = sampler.run(
    x0=x0,
    n_samples=100000,
    burn_in=20000,
    adapt_until=20000,
    start_adapt=500,
    adapt_interval=100,
)


# 4. Extract retained samples
samples = result.post_burn_samples


# 5. Evaluate diagnostics
diag = MCMCDiagnostics(
    samples,
    parameter_names=parameter_names,
)

diag.print_summary()


# 6. Plot results
plotter = MCMCPlotter(
    samples,
    parameter_names=parameter_names,
)

plotter.trace()
plotter.acf()
plotter.marginals()
plotter.corner()
```

For more reliable convergence assessment, several independent chains with different initial conditions and random seeds are recommended.

---

# Notes

* Low acceptance rates usually indicate that the proposal steps are too large.
* Very high acceptance rates usually indicate that the proposal steps are too small and that the chain may move slowly through the posterior.
* A rough acceptance-rate range for moderate- to high-dimensional random-walk Metropolis sampling is often approximately (0.15)--(0.35).
* Acceptance rate alone is not sufficient to establish convergence.
* Trace plots, autocorrelation, effective sample size, and multiple-chain diagnostics should also be examined.
* Adaptive covariance can substantially improve sampling efficiency for correlated and anisotropic unimodal posteriors.
* Adaptive covariance does not eliminate the limitations of local random-walk proposals.
* Strong multimodality can result in chains becoming trapped in individual modes.
* Nonlinear posterior geometries, such as the banana distribution, cannot be fully represented by a single covariance matrix.
* Funnel-shaped posteriors can be difficult because the appropriate proposal scale varies throughout parameter space.
* Heavy-tailed distributions may require longer chains to characterize the tails accurately.
* Multiple independent chains should be used whenever computationally feasible.
* Posterior diagnostics should be interpreted collectively rather than relying on any single convergence metric.

---

# Related Publication

The Adaptive Metropolis-Hastings methodology and Bayesian inference framework used in this repository are related to the following doctoral work:

- **Anabel del Val**,  
  *Bayesian calibration and assessment of gas-surface interaction models and experiments in atmospheric entry plasmas*,  
  Ph.D. thesis, Institut Polytechnique de Paris, 2021.  
  [Thesis available on HAL](https://inria.hal.science/tel-03504757)

The present modular implementation, including the diagnostics, plotting utilities, and benchmark examples, has been further developed and maintained by [**Stochastic Hypersonics Research Group**](http://www.umn.edu/~adelvalb).