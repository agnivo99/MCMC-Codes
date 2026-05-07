"""
Numerical diagnostics for MCMC chains.

This module contains:
    - autocorrelation
    - integrated autocorrelation time
    - effective sample size
    - split-chain R-hat
    - posterior summary table
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


Array = np.ndarray


class MCMCDiagnostics:
    """
    Diagnostics for one or more MCMC chains.

    Input shapes
    ------------
    One chain:
        samples.shape = (n_samples, n_dim)

    Multiple chains:
        samples.shape = (n_chains, n_samples, n_dim)
    """

    def __init__(self, samples: Array, parameter_names: Optional[Sequence[str]] = None):
        samples = np.asarray(samples, dtype=float)

        if samples.ndim == 1:
            samples = samples[:, None]

        if samples.ndim == 2:
            samples = samples[None, :, :]

        if samples.ndim != 3:
            raise ValueError("samples must have shape (n,), (n,d), or (m,n,d).")

        self.samples = samples
        self.n_chains, self.n_samples, self.dim = samples.shape

        if parameter_names is None:
            self.parameter_names = [f"theta_{i}" for i in range(self.dim)]
        else:
            if len(parameter_names) != self.dim:
                raise ValueError("parameter_names must have length equal to dimension.")
            self.parameter_names = list(parameter_names)

    @staticmethod
    def autocorrelation_1d(x: Array, max_lag: Optional[int] = None) -> Array:
        """
        Fast autocorrelation function using FFT.

        Returns autocorrelation from lag 0 to max_lag.
        """

        x = np.asarray(x, dtype=float)
        x = x - np.mean(x)

        n = len(x)

        if n == 0:
            raise ValueError("x must contain at least one sample.")

        if max_lag is None:
            max_lag = min(n - 1, 1000)

        max_lag = min(max_lag, n - 1)

        size = 1 << (2 * n - 1).bit_length()
        fft_x = np.fft.fft(x, size)

        acov = np.fft.ifft(fft_x * np.conjugate(fft_x)).real[:n]
        acov /= np.arange(n, 0, -1)

        if acov[0] <= 0:
            out = np.zeros(max_lag + 1)
            out[0] = 1.0
            return out

        acf = acov / acov[0]
        return acf[: max_lag + 1]

    @staticmethod
    def integrated_autocorr_time_1d(
        x: Array,
        max_lag: Optional[int] = None,
    ) -> float:
        """
        Estimate integrated autocorrelation time:

            tau = 1 + 2 sum_k rho_k

        Uses the initial positive sequence rule.
        """

        acf = MCMCDiagnostics.autocorrelation_1d(x, max_lag=max_lag)

        positive = acf[1:]
        stop = np.where(positive < 0.0)[0]

        if len(stop) > 0:
            positive = positive[: stop[0]]

        tau = 1.0 + 2.0 * np.sum(positive)

        return float(max(tau, 1.0))

    def ess(self, max_lag: Optional[int] = None) -> Array:
        """
        Effective sample size for each parameter.

            ESS = total number of samples / tau
        """

        ess = np.empty(self.dim, dtype=float)

        for j in range(self.dim):
            taus = []

            for c in range(self.n_chains):
                tau = self.integrated_autocorr_time_1d(
                    self.samples[c, :, j],
                    max_lag=max_lag,
                )
                taus.append(tau)

            tau_mean = np.mean(taus)
            ess[j] = self.n_chains * self.n_samples / tau_mean

        return ess

    def rhat(self) -> Array:
        """
        Split-chain Gelman-Rubin R-hat.

        R-hat compares between-chain variance with within-chain variance.

        Requires at least two chains.
        """

        if self.n_chains < 2:
            return np.full(self.dim, np.nan)

        x = self.samples

        half = self.n_samples // 2

        if half < 2:
            return np.full(self.dim, np.nan)

        x = x[:, : 2 * half, :]
        x = x.reshape(self.n_chains * 2, half, self.dim)

        m, n, _ = x.shape

        chain_means = np.mean(x, axis=1)
        chain_vars = np.var(x, axis=1, ddof=1)

        B = n * np.var(chain_means, axis=0, ddof=1)
        W = np.mean(chain_vars, axis=0)

        var_hat = ((n - 1) / n) * W + B / n

        with np.errstate(divide="ignore", invalid="ignore"):
            rhat = np.sqrt(var_hat / W)

        return rhat

    def summary(self, credible_interval: float = 0.95) -> list[dict[str, float | str]]:
        """
        Posterior summary table.

        Returns a list of dictionaries.
        """

        flat = self.samples.reshape(-1, self.dim)

        alpha = 0.5 * (1.0 - credible_interval)
        lo = 100.0 * alpha
        hi = 100.0 * (1.0 - alpha)

        ess = self.ess()
        rhat = self.rhat()

        rows: list[dict[str, float | str]] = []

        for j, name in enumerate(self.parameter_names):
            rows.append(
                {
                    "parameter": name,
                    "mean": float(np.mean(flat[:, j])),
                    "std": float(np.std(flat[:, j], ddof=1)),
                    f"q{lo:.1f}": float(np.percentile(flat[:, j], lo)),
                    "median": float(np.median(flat[:, j])),
                    f"q{hi:.1f}": float(np.percentile(flat[:, j], hi)),
                    "ESS": float(ess[j]),
                    "Rhat": float(rhat[j]),
                }
            )

        return rows

    def print_summary(self, credible_interval: float = 0.95) -> None:
        """Pretty-print summary table."""

        rows = self.summary(credible_interval=credible_interval)

        if not rows:
            print("No parameters to summarize.")
            return

        keys = list(rows[0].keys())

        header = " | ".join(f"{k:>12s}" for k in keys)
        print(header)
        print("-" * len(header))

        for row in rows:
            vals = []

            for k in keys:
                v = row[k]

                if isinstance(v, str):
                    vals.append(f"{v:>12s}")
                else:
                    vals.append(f"{v:12.4g}")

            print(" | ".join(vals))