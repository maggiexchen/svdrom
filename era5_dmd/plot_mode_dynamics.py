"""Plot a single DMD mode's raw temporal dynamics (no spatial
reconstruction) and correlate it against the true SST anomaly's rolling
mean in the Nino 3.4 box.

Usage
-----
    python plot_mode_dynamics.py --config configs/mode_dynamics_config.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from evaluate_reconstruction import evaluate_mode_dynamics
from plot_sst_anomaly import compute_rolling_mean, load_dmd_model_and_scaler

DEFAULT_ANOMALY_STYLES = {
    "True data (raw)": {"color": "tab:blue", "alpha": 0.3, "linewidth": 1},
    "DMD reconstruction (40 modes)": {"color": "tab:orange", "linewidth": 2},
    "DMD forecast (40 modes)": {"color": "tab:green", "linewidth": 2},
}

from svdrom.logger import setup_logger

logger = setup_logger("PlotModeDynamics", "plot_mode_dynamics.log")


def plot_dynamics(dynamics: xr.DataArray, title: str) -> plt.Figure:
    """Plot a mode's raw temporal dynamics as a function of time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dynamics.time.values, dynamics.values, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Dynamics (dimensionless)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_dynamics_with_anomaly(
    anomaly_series: dict[str, xr.DataArray],
    dynamics: xr.DataArray,
    dynamics_label: str,
    title: str,
    anomaly_styles: dict[str, dict] | None = None,
    legend_loc: str = "upper right",
) -> plt.Figure:
    """Plot SST anomaly timeseries (left axis, °C) together with a DMD
    mode's raw dynamics (right axis, dimensionless) on a shared time axis.
    A twin axis is used rather than one shared scale, since the two
    quantities' magnitudes aren't comparable (anomaly ~O(1) degC, raw
    dynamics ~O(10) dimensionless).
    """
    fig, ax1 = plt.subplots(figsize=(14, 5))
    for label, timeseries in anomaly_series.items():
        ax1.plot(
            timeseries.time.values,
            timeseries.values,
            label=label,
            **(anomaly_styles or {}).get(label, {}),
        )
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("SST anomaly (°C)")

    ax2 = ax1.twinx()
    ax2.plot(
        dynamics.time.values,
        dynamics.values,
        color="tab:purple",
        linestyle="--",
        label=dynamics_label,
    )
    ax2.set_ylabel("Mode dynamics (dimensionless)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc)
    ax1.set_title(title)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/mode_dynamics_config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dmd_config = config["dmd"]
    mode_rank = config["mode_rank"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading DMD model from %s.",
        Path(dmd_config["results_dir"]) / dmd_config["dmd_filename"],
    )
    opt_dmd, _ = load_dmd_model_and_scaler(dmd_config)

    logger.info("Loading true SST anomaly timeseries from %s.", config["true_anomaly_path"])
    true_timeseries = xr.open_dataarray(config["true_anomaly_path"])
    rolling_window_days = config.get("rolling_window_days", 30)
    true_rolling = compute_rolling_mean(true_timeseries, rolling_window_days)

    dynamics = evaluate_mode_dynamics(opt_dmd, true_rolling.time.values, mode_rank)

    dynamics_filename = config.get("dynamics_filename", "mode_dynamics.nc")
    dynamics.to_netcdf(output_dir / dynamics_filename)
    logger.info("Saved dynamics timeseries to %s.", output_dir / dynamics_filename)

    fig = plot_dynamics(
        dynamics,
        title=config.get("plot_title", f"DMD mode {mode_rank} dynamics"),
    )
    figure_filename = config.get("figure_filename", "mode_dynamics.png")
    fig.savefig(output_dir / figure_filename, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s.", output_dir / figure_filename)

    correlation = float(np.corrcoef(dynamics.values, true_rolling.values)[0, 1])
    logger.info(
        "Correlation between mode %d dynamics and true data (%d-day rolling mean): %.4f",
        mode_rank,
        rolling_window_days,
        correlation,
    )
    print(
        f"Correlation between mode {mode_rank} dynamics and true data "
        f"({rolling_window_days}-day rolling mean): {correlation:.4f}"
    )

    combine_config = config.get("combine_with")
    if combine_config is not None:
        anomaly_dir = Path(combine_config["anomaly_dir"])
        combined_true = xr.open_dataarray(anomaly_dir / "nino34_anomaly_true.nc")
        combined_reconstruction = xr.open_dataarray(
            anomaly_dir / "nino34_anomaly_reconstruction.nc"
        )
        combined_forecast = xr.open_dataarray(anomaly_dir / "nino34_anomaly_forecast.nc")
        combined_rolling = compute_rolling_mean(combined_true, rolling_window_days)
        rolling_label = f"True data ({rolling_window_days}-day rolling mean)"

        combined_times = xr.concat(
            [combined_reconstruction.time, combined_forecast.time], dim="time"
        ).values
        combined_dynamics = evaluate_mode_dynamics(opt_dmd, combined_times, mode_rank)

        anomaly_series = {
            "True data (raw)": combined_true,
            rolling_label: combined_rolling,
            "DMD reconstruction (40 modes)": combined_reconstruction,
            "DMD forecast (40 modes)": combined_forecast,
        }
        anomaly_styles = dict(DEFAULT_ANOMALY_STYLES)
        anomaly_styles[rolling_label] = {"color": "tab:blue", "linewidth": 2}

        fig = plot_dynamics_with_anomaly(
            anomaly_series,
            combined_dynamics,
            dynamics_label=combine_config.get(
                "dynamics_label", f"DMD mode {mode_rank} dynamics (right axis)"
            ),
            title=combine_config.get(
                "plot_title", f"Nino 3.4 SST anomaly vs DMD mode {mode_rank} dynamics"
            ),
            anomaly_styles=anomaly_styles,
            legend_loc=combine_config.get("legend_loc", "upper left"),
        )
        combined_figure_filename = combine_config.get(
            "figure_filename", "mode_dynamics_vs_anomaly.png"
        )
        fig.savefig(output_dir / combined_figure_filename, dpi=150, bbox_inches="tight")
        logger.info(
            "Saved combined dynamics/anomaly plot to %s.",
            output_dir / combined_figure_filename,
        )


if __name__ == "__main__":
    main()
