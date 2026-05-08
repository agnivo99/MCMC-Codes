"""
Plotting utilities for MCMC chains.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from diagnostics import MCMCDiagnostics


Array = np.ndarray


class MCMCPlotter:
    """
    Plotting utilities for one or more MCMC chains.

    Input shapes
    ------------
    One chain:
        samples.shape = (n_samples, n_dim)

    Multiple chains:
        samples.shape = (n_chains, n_samples, n_dim)
    """

    def __init__(
        self,
        samples: Array,
        parameter_names: Optional[Sequence[str]] = None,
        truths: Optional[Sequence[float]] = None,
    ):
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
                raise ValueError("parameter_names must have length equal to sample dimension.")
            self.parameter_names = list(parameter_names)

        if truths is None:
            self.truths = None
        else:
            truths = np.asarray(truths, dtype=float)
            if truths.shape != (self.dim,):
                raise ValueError("truths must have shape (n_dim,).")
            self.truths = truths

    def _get_param_indices(self, params: Optional[Iterable[int | str]]) -> list[int]:
        """
        Convert parameter names or integer indices into integer indices.
        """

        if params is None:
            return list(range(self.dim))

        indices = []

        for p in params:
            if isinstance(p, str):
                if p not in self.parameter_names:
                    raise ValueError(f"Unknown parameter name: {p}")
                indices.append(self.parameter_names.index(p))
            else:
                p = int(p)
                if p < 0 or p >= self.dim:
                    raise ValueError(f"Parameter index out of range: {p}")
                indices.append(p)

        return indices

    def _flat_samples(self) -> Array:
        """Flatten chains into shape (n_chains * n_samples, n_dim)."""
        return self.samples.reshape(-1, self.dim)

    def trace(
        self,
        params: Optional[Iterable[int | str]] = None,
        begin: int = 0,
        end: Optional[int] = 5000,
        show_truth: bool = True,
    ) -> None:
        """
        Trace plot for selected parameters.
        """

        indices = self._get_param_indices(params)

        begin = max(0, int(begin))
        stop = self.n_samples if end is None else min(int(end), self.n_samples)

        if begin >= stop:
            raise ValueError("begin must be smaller than end.")

        t = np.arange(begin, stop)

        for j in indices:
            plt.figure(figsize=(8, 3))

            for c in range(self.n_chains):
                plt.plot(
                    t,
                    self.samples[c, begin:stop, j],
                    lw=0.8,
                    alpha=0.8,
                    label=f"chain {c + 1}" if self.n_chains > 1 else None,
                )

            if show_truth and self.truths is not None:
                plt.axhline(
                    self.truths[j],
                    linestyle="--",
                    linewidth=1.2,
                    color="black",
                    label="truth",
                )

            plt.xlabel("MCMC iteration")
            plt.ylabel(self.parameter_names[j])
            plt.title(f"Trace plot: {self.parameter_names[j]}")

            if self.n_chains > 1 or (show_truth and self.truths is not None):
                plt.legend()

            plt.tight_layout()
            plt.show()

    def acf(
        self,
        params: Optional[Iterable[int | str]] = None,
        max_lag: int = 100,
    ) -> None:
        """
        Autocorrelation function plot.
        """

        indices = self._get_param_indices(params)
        lags = np.arange(max_lag + 1)

        for j in indices:
            plt.figure(figsize=(6, 3))

            for c in range(self.n_chains):
                acf_vals = MCMCDiagnostics.autocorrelation_1d(
                    self.samples[c, :, j],
                    max_lag=max_lag,
                )

                plt.plot(
                    lags[: len(acf_vals)],
                    acf_vals,
                    lw=1.2,
                    alpha=0.8,
                    label=f"chain {c + 1}" if self.n_chains > 1 else None,
                )

            plt.axhline(0.0, lw=0.8, color="black")
            plt.xlabel("Lag")
            plt.ylabel("ACF")
            plt.title(f"Autocorrelation: {self.parameter_names[j]}")

            if self.n_chains > 1:
                plt.legend()

            plt.tight_layout()
            plt.show()

    def marginals(
        self,
        params: Optional[Iterable[int | str]] = None,
        bins: int = 50,
        show_truth: bool = True,
        density: bool = True,
    ) -> None:
        """
        Plot combined marginal posterior histograms.

        All chains are flattened into one posterior sample array.
        """

        indices = self._get_param_indices(params)
        flat = self._flat_samples()

        for j in indices:
            plt.figure(figsize=(5, 3))

            plt.hist(
                flat[:, j],
                bins=bins,
                density=density,
                alpha=0.75,
            )

            if show_truth and self.truths is not None:
                plt.axvline(
                    self.truths[j],
                    linestyle="--",
                    linewidth=1.2,
                    color="black",
                    label="truth",
                )
                plt.legend()

            plt.xlabel(self.parameter_names[j])
            plt.ylabel("Density" if density else "Count")
            plt.title(f"Marginal posterior: {self.parameter_names[j]}")
            plt.tight_layout()
            plt.show()

    def compare_marginals(
        self,
        other_samples_dict: dict[str, Array],
        params: Optional[Iterable[int | str]] = None,
        bins: int = 50,
        kde: bool = True,
        show_truth: bool = True,
    ) -> None:
        """
        Compare marginal posteriors from multiple chains/datasets.

        Example
        -------
        chain_dict = {
            "Chain 1": chains[0],
            "Chain 2": chains[1],
            "Chain 3": chains[2],
            "Chain 4": chains[3],
        }
        """

        indices = self._get_param_indices(params)

        for j in indices:
            plt.figure(figsize=(7, 5))

            for label, samples in other_samples_dict.items():
                samples = np.asarray(samples, dtype=float)

                if samples.ndim == 3:
                    vals = samples.reshape(-1, samples.shape[-1])[:, j]
                elif samples.ndim == 2:
                    vals = samples[:, j]
                else:
                    raise ValueError(
                        f"Samples for {label} must have shape (n,d) or (m,n,d)."
                    )

                if kde:
                    sns.kdeplot(vals, fill=False, label=label)
                else:
                    plt.hist(
                        vals,
                        bins=bins,
                        density=True,
                        histtype="step",
                        linewidth=1.5,
                        label=label,
                    )

            if show_truth and self.truths is not None:
                plt.axvline(
                    self.truths[j],
                    linestyle="--",
                    linewidth=1.5,
                    color="black",
                    label="truth",
                )

            plt.xlabel(self.parameter_names[j])
            plt.ylabel("Density")
            plt.title(f"Posterior comparison: {self.parameter_names[j]}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def corner(
        self,
        params: Optional[Iterable[int | str]] = None,
        bins: int = 40,
        max_points: int = 10_000,
        show_truth: bool = True,
        show_titles: bool = True,
        figsize_scale: float = 2.7,
    ) -> None:
        """
        Corner plot.

        Diagonal:
            1D marginal histograms.

        Lower triangle:
            pairwise scatter plots.

        Upper triangle:
            hidden.
        """

        indices = self._get_param_indices(params)

        if len(indices) == 0:
            raise ValueError("At least one parameter must be selected.")

        flat = self._flat_samples()
        data = flat[:, indices]
        n_plot = len(indices)

        n_total = data.shape[0]

        if n_total > max_points:
            idx = np.linspace(0, n_total - 1, max_points, dtype=int)
            scatter_data = data[idx]
        else:
            scatter_data = data

        fig, axes = plt.subplots(
            n_plot,
            n_plot,
            figsize=(figsize_scale * n_plot, figsize_scale * n_plot),
            squeeze=False,
        )

        for i in range(n_plot):
            for j in range(n_plot):
                ax = axes[i, j]

                if i < j:
                    ax.axis("off")
                    continue

                idx_x = indices[j]
                name_x = self.parameter_names[idx_x]

                x = data[:, j]
                x_scatter = scatter_data[:, j]

                if i == j:
                    ax.hist(
                        x,
                        bins=bins,
                        density=True,
                        alpha=0.75,
                    )

                    if show_truth and self.truths is not None:
                        ax.axvline(
                            self.truths[idx_x],
                            linestyle="--",
                            linewidth=1.1,
                            color="black",
                        )

                    if show_titles:
                        mu = np.mean(x)
                        sd = np.std(x, ddof=1)
                        ax.set_title(
                            f"{name_x}\nmean={mu:.3g}, sd={sd:.3g}",
                            fontsize=10,
                        )
                    else:
                        ax.set_title(name_x, fontsize=10)

                    ax.set_ylabel("Density")

                else:
                    idx_y = indices[i]
                    name_y = self.parameter_names[idx_y]
                    y_scatter = scatter_data[:, i]

                    ax.plot(
                        x_scatter,
                        y_scatter,
                        ".",
                        alpha=0.35,
                        markersize=2,
                    )

                    if show_truth and self.truths is not None:
                        ax.axvline(
                            self.truths[idx_x],
                            linestyle="--",
                            linewidth=1.0,
                            color="black",
                        )
                        ax.axhline(
                            self.truths[idx_y],
                            linestyle="--",
                            linewidth=1.0,
                            color="black",
                        )

                    if j == 0:
                        ax.set_ylabel(name_y)

                if i == n_plot - 1:
                    ax.set_xlabel(name_x)
                else:
                    ax.set_xticklabels([])

                if i != j and j != 0:
                    ax.set_yticklabels([])

        plt.tight_layout()
        plt.show()

    def pair_scatter(
        self,
        x_param: int | str,
        y_param: int | str,
        max_points: int = 20_000,
        show_truth: bool = True,
    ) -> None:
        """
        Simple 2D scatter plot for two parameters.
        """

        ix = self._get_param_indices([x_param])[0]
        iy = self._get_param_indices([y_param])[0]

        flat = self._flat_samples()

        if flat.shape[0] > max_points:
            idx = np.linspace(0, flat.shape[0] - 1, max_points, dtype=int)
            flat = flat[idx]

        plt.figure(figsize=(5, 5))

        plt.plot(
            flat[:, ix],
            flat[:, iy],
            ".",
            alpha=0.35,
            markersize=2,
        )

        if show_truth and self.truths is not None:
            plt.axvline(
                self.truths[ix],
                linestyle="--",
                linewidth=1.2,
                color="black",
            )
            plt.axhline(
                self.truths[iy],
                linestyle="--",
                linewidth=1.2,
                color="black",
            )

        plt.xlabel(self.parameter_names[ix])
        plt.ylabel(self.parameter_names[iy])
        plt.title(f"{self.parameter_names[ix]} vs {self.parameter_names[iy]}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
