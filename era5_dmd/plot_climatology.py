"""Plot the global (lat-weighted) mean climatological SST as a function of
day-of-year/hour, i.e. its annual cycle, expanded onto a single reference
calendar year.

Usage
-----
    python plot_climatology.py --config configs/climatology_config.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import yaml
from climatology import load_climatology
from plot_sst_anomaly import compute_global_mean, plot_timeseries

from svdrom.logger import setup_logger
from svdrom.weather_utils import expand_time_climatology

logger = setup_logger("PlotClimatology", "plot_climatology.log")


def expand_climatology_range(
    climatology: xr.DataArray, start_year: int, end_year: int
) -> xr.DataArray:
    """Expand a (dayofyear, hour) climatology onto a real multi-year
    calendar time axis, repeating its annual cycle once per year across
    [start_year, end_year] - a generalisation of
    `svdrom.weather_utils.expand_time_climatology` (which only expands
    onto a single reference year) to an arbitrary range.
    """
    climatology = climatology.compute()
    hours = sorted(int(h) for h in climatology.hour.values)
    freq_hours = hours[1] - hours[0]
    times = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-31 23:00", freq=f"{freq_hours}h"
    )
    times = times[times.hour.isin(hours)]
    times_da = xr.DataArray(times, dims="time", coords={"time": times})
    expanded = climatology.sel(
        dayofyear=times_da.dt.dayofyear, hour=times_da.dt.hour
    )
    return expanded.drop_vars(["dayofyear", "hour"], errors="ignore")


def plot_climatology_cycle(
    climatology_timeseries: xr.DataArray, title: str, xlabel: str
) -> plt.Figure:
    """Plot the climatological SST as a function of time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(climatology_timeseries.time.values, climatology_timeseries.values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Climatological SST (°C)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def load_climatology_cycle(
    climatology_dir: str, variable: str, reference_year: int
) -> xr.DataArray:
    """Load a climatology folder, reduce to its global (lat-weighted) mean,
    and expand it onto a single reference calendar year, in Celsius.
    """
    climatology_mean = load_climatology(
        climatology_dir, variable=variable, std_variable=None
    )
    climatology_mean = climatology_mean - 273.15  # Kelvin -> Celsius
    global_mean = compute_global_mean(climatology_mean).compute()
    return expand_time_climatology(global_mean, year=reference_year)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/climatology_config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "climatologies" in config:
        reference_year = config.get("reference_year", 2020)
        series = {}
        for entry in config["climatologies"]:
            logger.info("Loading climatology from %s.", entry["climatology_dir"])
            series[entry["label"]] = load_climatology_cycle(
                entry["climatology_dir"], entry.get("variable", "sst"), reference_year
            )

        climatology_filename = config.get(
            "climatology_filename", "sst_climatology_comparison.nc"
        )
        xr.Dataset(series).to_netcdf(output_dir / climatology_filename)
        logger.info(
            "Saved climatology comparison timeseries to %s.",
            output_dir / climatology_filename,
        )

        fig = plot_timeseries(
            series,
            title=config.get("plot_title", "Global-mean SST climatology comparison"),
            ylabel="Climatological SST (°C)",
            zero_line=False,
        )
        figure_filename = config.get(
            "figure_filename", "sst_climatology_comparison.png"
        )
        fig.savefig(output_dir / figure_filename, dpi=150, bbox_inches="tight")
        logger.info(
            "Saved climatology comparison plot to %s.", output_dir / figure_filename
        )
        return

    climatology_config = config["climatology"]
    logger.info("Loading climatology from %s.", climatology_config["climatology_dir"])
    climatology_mean = load_climatology(
        climatology_config["climatology_dir"],
        variable=climatology_config.get("variable", "sst"),
        std_variable=None,
    )
    climatology_mean = climatology_mean - 273.15  # Kelvin -> Celsius

    global_mean = compute_global_mean(climatology_mean).compute()

    if "start_year" in config and "end_year" in config:
        climatology_timeseries = expand_climatology_range(
            global_mean, config["start_year"], config["end_year"]
        )
        xlabel = "Time"
    else:
        climatology_timeseries = expand_time_climatology(
            global_mean, year=config.get("reference_year", 2020)
        )
        xlabel = "Time (one reference calendar year)"

    climatology_filename = config.get(
        "climatology_filename", "sst_climatology_timeseries.nc"
    )
    climatology_timeseries.to_netcdf(output_dir / climatology_filename)
    logger.info(
        "Saved climatology timeseries to %s.", output_dir / climatology_filename
    )

    fig = plot_climatology_cycle(
        climatology_timeseries,
        title=config.get("plot_title", "Global-mean SST climatology"),
        xlabel=xlabel,
    )
    figure_filename = config.get("figure_filename", "sst_climatology_timeseries.png")
    fig.savefig(output_dir / figure_filename, dpi=150, bbox_inches="tight")
    logger.info("Saved climatology plot to %s.", output_dir / figure_filename)


if __name__ == "__main__":
    main()
