"""
Core adaptive Random-Walk Metropolis sampler.

This module contains:
    - MCMCResult
    - OnlineCovariance
    - AdaptiveMetropolisSampler

The sampler only requires a user-defined log posterior:

    log_posterior(theta)

which returns log p(theta | data) up to an additive constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


Array = np.ndarray
LogPosterior = Callable[[Array], float]


@dataclass
class MCMCResult:
    """
    Container returned by AdaptiveMetropolisSampler.run().

    Attributes
    ----------
    samples:
        Array of shape (n_samples, n_dim).

    logp:
        Log-posterior values at each stored sample.

    accepted:
        Boolean array indicating whether each proposal was accepted.

    proposal_cov:
        Final learned proposal covariance before internal scale factor.

    initial_position:
        Starting point of the chain.

    final_position:
        Final state of the chain.

    burn_in:
        Number of burn-in samples.

    adapt_until:
        Last iteration where adaptation was allowed.

    parameter_names:
        Names of parameters.
    """

    samples: Array
    logp: Array
    accepted: Array
    proposal_cov: Array
    initial_position: Array
    final_position: Array
    burn_in: int
    adapt_until: int
    parameter_names: list[str]

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate."""
        return float(np.mean(self.accepted))

    @property
    def post_burn_samples(self) -> Array:
        """Samples after discarding burn-in."""
        return self.samples[self.burn_in :]

    @property
    def post_burn_logp(self) -> Array:
        """Log-posterior values after discarding burn-in."""
        return self.logp[self.burn_in :]

    @property
    def post_burn_acceptance_rate(self) -> float:
        """Acceptance rate after burn-in."""
        if self.burn_in >= len(self.accepted):
            return np.nan
        return float(np.mean(self.accepted[self.burn_in :]))


class OnlineCovariance:
    """
    Online covariance estimator using Welford updates.

    This avoids recomputing the covariance from scratch during adaptation.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=float)
        self.M2 = np.zeros((self.dim, self.dim), dtype=float)

    def update(self, x: Array) -> None:
        """
        Add one sample to the online covariance estimate.

        Welford update:

            mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n

            M2_n = M2_{n-1}
                   + (x_n - mean_{n-1})(x_n - mean_n)^T
        """

        x = np.asarray(x, dtype=float)

        if x.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},).")

        self.n += 1

        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean

        self.M2 += np.outer(delta, delta2)

    @property
    def covariance(self) -> Array:
        """
        Return current sample covariance.

        If fewer than two samples are available, return identity.
        """

        if self.n < 2:
            return np.eye(self.dim)

        cov = self.M2 / (self.n - 1)

        # Symmetrize to remove tiny floating-point asymmetry.
        return 0.5 * (cov + cov.T)


class AdaptiveMetropolisSampler:
    """
    Adaptive Random-Walk Metropolis sampler.

    Proposal:

        theta_star = theta_current + L z

    where:

        z ~ N(0, I)

    and:

        L L^T = scale^2 * C + jitter * I

    The default scale is:

        scale = 2.38 / sqrt(d)

    During burn-in, C is adapted using empirical chain covariance.
    """

    def __init__(
        self,
        log_posterior: LogPosterior,
        initial_cov: Array,
        parameter_names: Optional[Sequence[str]] = None,
        scale: Optional[float] = None,
        jitter: float = 1.0e-8,
        rng: Optional[np.random.Generator] = None,
    ):
        self.log_posterior = log_posterior
        self.initial_cov = self._validate_covariance(initial_cov)

        self.dim = self.initial_cov.shape[0]
        self.scale = float(2.38 / np.sqrt(self.dim) if scale is None else scale)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng() if rng is None else rng

        if parameter_names is None:
            self.parameter_names = [f"theta_{i}" for i in range(self.dim)]
        else:
            if len(parameter_names) != self.dim:
                raise ValueError("parameter_names must have length equal to dimension.")
            self.parameter_names = list(parameter_names)

        self.proposal_cov = self.initial_cov.copy()
        self.proposal_chol = self._cholesky_scaled(self.proposal_cov)

    @staticmethod
    def _validate_covariance(cov: Array) -> Array:
        """Validate initial covariance matrix."""

        cov = np.asarray(cov, dtype=float)

        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("initial_cov must be a square 2D array.")

        if not np.allclose(cov, cov.T, atol=1.0e-12):
            raise ValueError("initial_cov must be symmetric.")

        return cov

    def _cholesky_scaled(self, cov: Array) -> Array:
        """
        Compute Cholesky factor of:

            scale^2 * cov + jitter * I

        If Cholesky fails, jitter is increased. If it still fails,
        eigenvalue clipping is used as a last-resort repair.
        """

        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)

        eye = np.eye(self.dim)
        jitter = self.jitter

        for _ in range(8):
            try:
                return np.linalg.cholesky((self.scale**2) * cov + jitter * eye)
            except np.linalg.LinAlgError:
                jitter *= 10.0

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, self.jitter)
        repaired = (eigvecs * eigvals) @ eigvecs.T
        repaired = 0.5 * (repaired + repaired.T)

        return np.linalg.cholesky((self.scale**2) * repaired + jitter * eye)

    def _propose(self, x: Array) -> Array:
        """
        Gaussian random-walk proposal.

            theta_star = theta_current + L z
        """

        z = self.rng.standard_normal(self.dim)
        jump = self.proposal_chol @ z
        return x + jump

    def _step(self, x: Array, logp_x: float) -> tuple[Array, float, bool]:
        """
        Perform one Metropolis-Hastings step.

        Since the proposal is symmetric:

            log_alpha = logp(theta_star) - logp(theta_current)

        Accept if:

            log(u) < log_alpha
        """

        x_prop = self._propose(x)
        logp_prop = float(self.log_posterior(x_prop))

        if not np.isfinite(logp_prop):
            return x, logp_x, False

        log_alpha = logp_prop - logp_x
        accept = np.log(self.rng.random()) < log_alpha

        if accept:
            return x_prop, logp_prop, True

        return x, logp_x, False

    def run(
        self,
        x0: Array,
        n_samples: int,
        burn_in: int = 0,
        adapt_until: Optional[int] = None,
        adapt_interval: int = 100,
        start_adapt: int = 100,
        progress: bool = True,
    ) -> MCMCResult:
        """
        Run one adaptive Metropolis chain.

        Parameters
        ----------
        x0:
            Initial parameter vector.

        n_samples:
            Total number of stored samples, including burn-in.

        burn_in:
            Number of initial samples to discard.

        adapt_until:
            Last iteration where adaptation is allowed.
            If None, adaptation is used only during burn-in.

        adapt_interval:
            Update covariance every `adapt_interval` iterations.

        start_adapt:
            Do not adapt before this many samples are collected.

        progress:
            Print progress information.
        """

        x = np.asarray(x0, dtype=float).copy()

        if x.shape != (self.dim,):
            raise ValueError(f"x0 must have shape ({self.dim},).")

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        if burn_in < 0 or burn_in >= n_samples:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_samples.")

        if adapt_until is None:
            adapt_until = burn_in

        adapt_until = int(adapt_until)

        logp_x = float(self.log_posterior(x))

        if not np.isfinite(logp_x):
            raise ValueError("Initial point has non-finite log posterior.")

        samples = np.empty((n_samples, self.dim), dtype=float)
        logp = np.empty(n_samples, dtype=float)
        accepted = np.zeros(n_samples, dtype=bool)

        cov_est = OnlineCovariance(self.dim)
        initial_position = x.copy()

        for it in range(n_samples):
            x, logp_x, acc = self._step(x, logp_x)

            samples[it] = x
            logp[it] = logp_x
            accepted[it] = acc

            cov_est.update(x)

            do_adapt = (
                adapt_until > 0
                and it + 1 <= adapt_until
                and it + 1 >= start_adapt
                and (it + 1) % adapt_interval == 0
            )

            if do_adapt:
                self.proposal_cov = cov_est.covariance
                self.proposal_chol = self._cholesky_scaled(self.proposal_cov)

            if progress and ((it + 1) % max(1, n_samples // 10) == 0 or it == 0):
                ar = np.mean(accepted[: it + 1])
                print(
                    f"iter {it + 1:>7d}/{n_samples} | "
                    f"acceptance={ar:6.3f} | logp={logp_x: .3e}",
                    flush=True,
                )

        return MCMCResult(
            samples=samples,
            logp=logp,
            accepted=accepted,
            proposal_cov=self.proposal_cov.copy(),
            initial_position=initial_position,
            final_position=x.copy(),
            burn_in=burn_in,
            adapt_until=adapt_until,
            parameter_names=self.parameter_names,
        )