"""Forecast SST beyond the years a DMD model was fitted on, and evaluate the
forecast against the true data via RMSE.

Calls `OptDMD._predict()` directly rather than the public `forecast()`
method: `forecast()` only accepts a time span and step relative to the end
of the training period, whereas `_predict()` accepts arbitrary time points,
letting the forecast be evaluated at exactly the true data's own timestamps.

Usage
-----
    python evaluate_forecast.py --config configs/forecast_config.yaml
"""

import argparse
import pickle
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from data_loading import load_era5_variable
from pipeline_utils import load_dmd_model_with_true_time, resolve_years, select, start_dask_client
from resampling import resample_snapshots
from standardising import unstandardise

from svdrom.dmd import OptDMD
from svdrom.logger import setup_logger
from svdrom.preprocessing import StandardScaler
from svdrom.weather_utils import compute_rmse

logger = setup_logger("EvaluateForecast", "evaluate_forecast.log")


def load_true_sst(data_config: dict) -> xr.DataArray:
    """Load the true SST data over the requested forecast years, at the
    same temporal resolution used to fit the DMD model.
    """
    da = load_era5_variable(
        years=resolve_years(data_config),
        **select(data_config, "variable_path", "file_pattern"),
    )
    resample_hours = data_config.get("resample_hours")
    if resample_hours is not None:
        da = resample_snapshots(da, hours=resample_hours)
    return da


def times_to_model_units(opt_dmd: OptDMD, times: np.ndarray) -> np.ndarray:
    """Convert real datetimes into the elapsed-time-since-first-training-
    snapshot float representation that `_predict()` expects, using the same
    reference point (`opt_dmd.time_fit[0]`) and units (`opt_dmd.time_units`)
    as the DMD fit.
    """
    time_fit = opt_dmd.time_fit
    if time_fit is None:
        msg = "The DMD model has not been fitted."
        raise RuntimeError(msg)
    deltas = (times - time_fit[0]).astype(f"timedelta64[{opt_dmd.time_units}]")
    return deltas / np.timedelta64(1, opt_dmd.time_units)


def predict_to_dataarray(
    opt_dmd: OptDMD, prediction: np.ndarray, times: np.ndarray
) -> xr.DataArray:
    """Wrap the raw array returned by `_predict()` into an xr.DataArray,
    reusing the spatial coordinates of the fitted DMD modes.
    """
    modes = opt_dmd.modes
    if modes is None:
        msg = "The DMD model has not been fitted."
        raise RuntimeError(msg)
    dims = (modes.dims[0], "time")
    coords = {k: v for k, v in modes.coords.items() if k != "components"}
    coords["time"] = times
    return xr.DataArray(prediction, dims=dims, coords=coords, name="dmd_forecast")


def forecast_physical(
    opt_dmd: OptDMD,
    scaler: StandardScaler,
    times: np.ndarray,
    use_dask: bool = False,
) -> xr.DataArray:
    """Forecast SST at `times` (real datetimes beyond the DMD's training
    period) by calling `_predict()` directly, then unstack back to (time,
    latitude, longitude) and undo standardising.
    """
    logger.info("Forecasting %d snapshots beyond the training period.", len(times))
    t = times_to_model_units(opt_dmd, times)
    prediction = opt_dmd._predict(t, use_dask=use_dask)
    forecast = predict_to_dataarray(opt_dmd, prediction, times)
    forecast = forecast.real.unstack()
    return unstandardise(forecast, scaler)


def evaluate_rmse(forecast: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    """Compute the lat-weighted RMSE between the forecast and the true SST
    data, as a function of time.

    The true data is restricted to the forecast's time/latitude/longitude
    grid first, since dropping NaN/land samples before fitting the SVD (see
    `standardising.stack_spatial_dims`) can remove entire latitude bands
    (e.g. Antarctica) from the forecast's spatial grid.
    """
    truth = truth.sel(
        time=forecast.time,
        latitude=forecast.latitude,
        longitude=forecast.longitude,
    )
    return compute_rmse(ground_truth=truth, prediction=forecast)


def save_rmse(
    rmse: xr.DataArray, output_dir: str, rmse_filename: str = "forecast_rmse.nc"
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rmse.to_netcdf(output_path / rmse_filename)
    logger.info("Saved forecast RMSE to %s.", output_path / rmse_filename)


def evaluate_rmse_map(forecast: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    """Compute the RMSE between the forecast and the true SST data, averaged
    over time instead of latitude/longitude, giving a map of where the
    forecast is best/worst.

    `lat_weighting` is turned off here: it exists to correct for grid-cell
    area when averaging spatially, not when averaging over time.
    """
    truth = truth.sel(
        time=forecast.time,
        latitude=forecast.latitude,
        longitude=forecast.longitude,
    )
    return compute_rmse(
        ground_truth=truth, prediction=forecast, lat_weighting=False, dims="time"
    )


def plot_rmse_map(rmse_map: xr.DataArray, title: str) -> plt.Figure:
    """Plot a time-averaged RMSE map on a Plate Carree projection."""
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    rmse_map.plot.contourf(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Reds",  # sequential, since RMSE is non-negative
        add_colorbar=True,
        cbar_kwargs={"shrink": 0.7, "label": "RMSE (K)", "pad": 0.08},
    )
    ax.coastlines(color="black")
    ax.gridlines(draw_labels=True)
    ax.set_title(title)
    return fig


def save_rmse_map(
    fig: plt.Figure, output_dir: str, rmse_map_filename: str = "forecast_rmse_map.png"
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / rmse_map_filename, dpi=150, bbox_inches="tight")
    logger.info("Saved forecast RMSE map to %s.", output_path / rmse_map_filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/forecast_config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    results_dir = config["results_dir"]

    client = start_dask_client(config.get("dask", {}))
    try:
        opt_dmd = load_dmd_model_with_true_time(
            results_dir, config["dmd_filename"], data_config
        )
        with open(Path(results_dir) / "standard_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        logger.info("Loading true SST data from %s.", data_config["variable_path"])
        truth = load_true_sst(data_config)

        forecast = forecast_physical(
            opt_dmd,
            scaler,
            truth.time.values,
            use_dask=config.get("use_dask", False),
        )
        rmse = evaluate_rmse(forecast, truth).compute()

        logger.info(
            "Forecast RMSE (K): mean=%.4f, max=%.4f.",
            float(rmse.mean()),
            float(rmse.max()),
        )
        save_rmse(rmse, results_dir, config.get("rmse_filename", "forecast_rmse.nc"))

        rmse_map = evaluate_rmse_map(forecast, truth).compute()
        fig = plot_rmse_map(rmse_map, title="Time-averaged DMD forecast RMSE")
        save_rmse_map(
            fig, results_dir, config.get("rmse_map_filename", "forecast_rmse_map.png")
        )
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
