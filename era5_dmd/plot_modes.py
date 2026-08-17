"""Plot the real spatial component of the fitted DMD modes and their
temporal dynamics.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pipeline_utils import load_dmd_model_with_true_time
from svdrom.dmd import OptDMD
from svdrom.logger import setup_logger

logger = setup_logger("PlotModes", "plot_modes.log")


def rank_modes_by_amplitude(opt_dmd: OptDMD) -> np.ndarray:
    """Return mode indices ordered by descending amplitude magnitude."""
    return np.argsort(-np.abs(opt_dmd.amplitudes))


def plot_modes_and_dynamics(
    opt_dmd: OptDMD,
    time_values: np.ndarray,
    mode_ranks: list[int] = (0, 2, 4, 6, 8),
    hours_per_day: int = 24,
) -> plt.Figure:
    """Plot the real spatial pattern and temporal dynamics of the modes at
    `mode_ranks` (by descending amplitude), one row per mode.

    Complex DMD modes come in conjugate pairs with identical amplitude, so
    the default `mode_ranks` picks every other rank to show one mode per
    pair rather than plotting each conjugate twice.

    Returns
    -------
    plt.Figure
        Figure with a spatial mode map and a dynamics line plot per row.
    """
    modes = opt_dmd.modes.unstack()
    mode_order = rank_modes_by_amplitude(opt_dmd)

    n_rows = len(mode_ranks)
    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(20, 3.4 * n_rows))

    for row, mode_rank in enumerate(mode_ranks):
        mode_idx = mode_order[mode_rank]

        ax = fig.add_subplot(n_rows, 2, 2 * row + 1, projection=ccrs.PlateCarree())
        modes[mode_idx, :, :].real.plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            add_colorbar=True,
            cbar_kwargs={"shrink": 0.8, "pad": 0.06},
        )
        ax.coastlines(color="black")
        ax.gridlines(draw_labels=True)
        ax.set_title(f"Mode {mode_rank}")

        ax = fig.add_subplot(n_rows, 2, 2 * row + 2)
        period = 2 * np.pi / abs(opt_dmd.eigs[mode_idx].imag) / hours_per_day
        ax.plot(
            time_values,
            opt_dmd.dynamics[mode_idx, :].real,
            label=f"Oscillation period: {period:.2f} days",
        )
        ax.set_title(f"Dynamics of Mode {mode_rank}")
        ax.set_ylabel("Temporal amplitude")
        ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-config",
        default="plotting_config.yaml",
        help="Path to the plotting YAML config file.",
    )
    args = parser.parse_args()

    with open(args.plot_config) as f:
        plot_config = yaml.safe_load(f)

    results_dir = plot_config["results_dir"]
    opt_dmd = load_dmd_model_with_true_time(
        results_dir, plot_config["dmd_filename"], plot_config.get("data")
    )
    time_values = opt_dmd.dynamics.coords["time"].values

    fig = plot_modes_and_dynamics(
        opt_dmd,
        time_values,
        mode_ranks=plot_config.get("mode_ranks", [0, 2, 4, 6, 8]),
        hours_per_day=plot_config.get("hours_per_day", 24),
    )

    figure_path = Path(results_dir) / plot_config.get("modes_figure_filename", "dmd_modes.png")
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    logger.info("Saved modes figure to %s.", figure_path)


if __name__ == "__main__":
    main()
