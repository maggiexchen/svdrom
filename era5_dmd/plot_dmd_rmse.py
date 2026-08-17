"""Plot the DMD reconstruction RMSE (over its training window) and
forecast RMSE (beyond it), both against true SST, on the same time axis.

Usage
-----
    python plot_dmd_rmse.py --config configs/dmd_rmse_config.yaml
"""

import argparse
from pathlib import Path

import xarray as xr
import yaml
from data_loading import load_era5_variable
from evaluate_reconstruction import evaluate_rmse
from pipeline_utils import load_or_compute_dataarray, resolve_years, select, start_dask_client
from plot_sst_anomaly import plot_timeseries

from svdrom.logger import setup_logger

logger = setup_logger("PlotDmdRmse", "plot_dmd_rmse.log")


def compute_forecast_rmse(forecast: xr.DataArray, data_config: dict) -> xr.DataArray:
    """Load true SST data matching `forecast`'s time span and compute the
    lat-weighted RMSE against it, as a function of time.
    """
    logger.info("Loading true SST data from %s.", data_config["variable_path"])
    truth = load_era5_variable(
        years=resolve_years(data_config),
        **select(data_config, "variable_path", "file_pattern"),
    )
    return evaluate_rmse(forecast, truth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/dmd_rmse_config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    client = start_dask_client(config.get("dask", {}))
    try:
        reconstruction_rmse_path = config["reconstruction_rmse_path"]
        logger.info(
            "Loading already-computed reconstruction RMSE from %s.",
            reconstruction_rmse_path,
        )
        reconstruction_rmse = xr.open_dataarray(reconstruction_rmse_path)

        forecast_field_path = config["forecast_field_path"]
        logger.info("Loading cached forecast field from %s.", forecast_field_path)
        forecast = xr.open_dataarray(forecast_field_path, chunks={})
        forecast = forecast.sel(time=slice(config.get("forecast_start"), None))

        forecast_rmse_filename = config.get("forecast_rmse_filename", "forecast_rmse.nc")
        forecast_rmse = load_or_compute_dataarray(
            output_dir / forecast_rmse_filename,
            lambda: compute_forecast_rmse(forecast, config["forecast_truth_data"]),
        )

        series = {
            "DMD reconstruction RMSE": reconstruction_rmse,
            "DMD forecast RMSE": forecast_rmse,
        }
        styles = {
            "DMD reconstruction RMSE": {"color": "tab:orange"},
            "DMD forecast RMSE": {"color": "tab:green"},
        }

        fig = plot_timeseries(
            series,
            title=config.get("plot_title", "DMD reconstruction/forecast RMSE"),
            styles=styles,
            ylabel="RMSE (K)",
            zero_line=False,
        )
        figure_filename = config.get("figure_filename", "dmd_rmse.png")
        fig.savefig(output_dir / figure_filename, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s.", output_dir / figure_filename)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
