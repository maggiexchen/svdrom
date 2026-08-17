"""Reconstruct SST from a fitted DMD model and evaluate it against the true
data via RMSE.

Usage
-----
    python evaluate_reconstruction.py --config configs/evaluate_config.yaml
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

logger = setup_logger("EvaluateReconstruction", "evaluate_reconstruction.log")


def load_true_sst(data_config: dict) -> xr.DataArray:
    """Load the true SST data over the same years and at the same temporal
    resolution used to fit the DMD model, so its time coordinate lines up
    with the reconstruction's.
    """
    da = load_era5_variable(
        years=resolve_years(data_config),
        **select(data_config, "variable_path", "file_pattern"),
    )
    resample_hours = data_config.get("resample_hours")
    if resample_hours is not None:
        da = resample_snapshots(da, hours=resample_hours)
    return da


def reconstruct_physical(
    opt_dmd: OptDMD,
    scaler: StandardScaler,
    t: slice | int | str | None = None,
    memory_limit_bytes: float = 1e9,
) -> xr.DataArray:
    """Reconstruct the training dataset from the fitted DMD model, unstack
    back to (time, latitude, longitude), and undo standardising.

    Unstacking must happen before undoing standardising: the reconstruction
    is still in stacked `samples` form when it comes out of `reconstruct()`,
    whereas `scaler.mean`/`scaler.std` are indexed by `latitude`/`longitude`
    (they were fitted before spatial stacking), so combining them only lines
    up correctly once the reconstruction has been unstacked too.

    Parameters
    ----------
    t : slice | int | str | None, optional
        The time span to reconstruct, forwarded to `opt_dmd.reconstruct()`
        (see its docstring). Defaults to None, reconstructing the whole
        training dataset, which could potentially be very large.
    """
    logger.info("Reconstructing training data from the DMD model.")
    reconstruction = opt_dmd.reconstruct(t=t, memory_limit_bytes=memory_limit_bytes)
    reconstruction = reconstruction.real.unstack()
    return unstandardise(reconstruction, scaler)


def reconstruct_physical_at_times(
    opt_dmd: OptDMD,
    scaler: StandardScaler,
    times: np.ndarray,
    memory_limit_bytes: float = 1e9,
) -> xr.DataArray:
    """Evaluate the fitted DMD model's continuous-time dynamics at
    arbitrary calendar timestamps, in physical units.

    `OptDMD.reconstruct()` only reconstructs at the exact discrete
    timestamps of the training snapshots (`_generate_reconstruct_time_vector`
    selects among the stored `time_fit` labels). But DMD's modes/
    eigenvalues/amplitudes have no inherent time step of their own -
    `OptDMD._predict()` evaluates the continuous formula
    modes @ diag(amplitudes) @ exp(eigs * t) at any elapsed-time vector, so
    this reuses it (and `_prediction_to_dataarray`) directly, at `times`'s
    own resolution instead of only the grid the model happens to have been
    fit on. This deliberately reaches into `OptDMD`'s private methods
    rather than reimplementing their (Dask-aware, bagging-aware) logic.

    Parameters
    ----------
    times : np.ndarray
        The calendar timestamps to evaluate the model at. Should stay
        within (or close to) the model's training window
        (`opt_dmd.time_fit[0]` to `[-1]`) - evaluating far outside it is
        equivalent to `OptDMD.forecast()`'s extrapolation, with the same
        caveats (unbounded growth/decay of the fitted eigenvalues).
    """
    if opt_dmd.hankel_d > 1:
        msg = (
            "reconstruct_physical_at_times() does not support "
            "Hankel-preprocessed (time-delay embedded) models."
        )
        raise NotImplementedError(msg)

    logger.info("Evaluating DMD dynamics at %d requested timestamps.", len(times))
    t = (
        np.asarray(times, dtype="datetime64[ns]") - opt_dmd.time_fit[0]
    ) / np.timedelta64(1, opt_dmd.time_units)

    estimated_size = opt_dmd._estimate_array_size(t)
    use_dask = estimated_size > memory_limit_bytes
    logger.info(
        "Estimated size is %.3f KB. Will%s use Dask.",
        estimated_size / 1e3,
        "" if use_dask else " not",
    )
    prediction = opt_dmd._predict(t, use_dask)
    reconstruction = opt_dmd._prediction_to_dataarray(prediction, np.asarray(times))
    reconstruction = reconstruction.real.unstack()
    return unstandardise(reconstruction, scaler)


def forecast_physical(
    opt_dmd: OptDMD,
    scaler: StandardScaler,
    forecast_span: str | int,
    dt: str | int | None = None,
    memory_limit_bytes: float = 1e9,
) -> xr.DataArray:
    """Forecast SST beyond the DMD model's training window, unstack back to
    (time, latitude, longitude), and undo standardising.

    Unlike `reconstruct_physical`/`reconstruct_physical_at_times`, this
    extrapolates the fitted dynamics past `opt_dmd.time_fit[-1]` rather
    than evaluating them within the training window, so its accuracy
    degrades the further out it's evaluated - any mode with a positive
    real eigenvalue component grows without bound the further ahead you
    forecast.

    Parameters
    ----------
    forecast_span, dt : see `OptDMD.forecast()`.
    """
    logger.info(
        "Forecasting %s beyond the DMD model's training period.", forecast_span
    )
    forecast = opt_dmd.forecast(forecast_span, dt=dt, memory_limit_bytes=memory_limit_bytes)
    forecast = forecast.real.unstack()
    return unstandardise(forecast, scaler)


def reconstruct_mode_contribution(
    opt_dmd: OptDMD,
    scaler: StandardScaler,
    times: np.ndarray,
    mode_rank: int,
    add_mean: bool = False,
) -> xr.DataArray:
    """Evaluate a single DMD mode's own physical-space contribution at
    `times`, in physical units.

    `mode_rank` is a mode's position when all modes are ordered by
    descending amplitude - the same "Mode N" labelling used by
    `plot_modes.py`/`animate_modes.py` (e.g. `dmd_modes.png`,
    `animation_dmd_mode6.gif`), not its raw storage index. Complex DMD
    modes come in conjugate pairs sharing one amplitude, so both members
    of `mode_rank`'s pair are included together, which is what makes the
    result real-valued (a lone unpaired complex mode would leave a
    residual imaginary part with no physical meaning).

    By default (`add_mean=False`), this does *not* add back `scaler.mean`:
    the per-pixel mean was removed once, collectively, before the DMD
    fit - it isn't attributable to any single mode, so a single mode's
    contribution is inherently a deviation around zero (only `scaler.std`
    rescales it back to physical units), directly comparable to an SST
    anomaly without any further climatology subtraction.

    Set `add_mean=True` to add `scaler.mean` back anyway, treating this
    mode's contribution as if it were a full physical SST field in its
    own right (e.g. to then subtract a climatology from it the same way
    as the full reconstruction, rather than using the result as-is).
    """
    mode_order = np.argsort(-np.abs(opt_dmd.amplitudes))
    idx = mode_order[mode_rank]
    conj_idx = int(np.argmin(np.abs(opt_dmd.eigs - np.conj(opt_dmd.eigs[idx]))))
    indices = sorted({idx, conj_idx})
    logger.info(
        "Evaluating mode rank %d (raw indices %s, period %.1f days) at %d timestamps.",
        mode_rank,
        indices,
        2 * np.pi / abs(opt_dmd.eigs[idx].imag) / 24,
        len(times),
    )

    modes = opt_dmd.modes
    t = (
        np.asarray(times, dtype="datetime64[ns]") - opt_dmd.time_fit[0]
    ) / np.timedelta64(1, opt_dmd.time_units)
    exp_term = np.exp(np.outer(opt_dmd.eigs[indices], t))
    prediction = modes.values[:, indices] @ (
        opt_dmd.amplitudes[indices][:, None] * exp_term
    )

    coords = {k: v for k, v in modes.coords.items() if k != "components"}
    coords[opt_dmd.time_dimension] = np.asarray(times)
    contribution = xr.DataArray(
        prediction, dims=(modes.dims[0], opt_dmd.time_dimension), coords=coords
    )
    contribution = contribution.real.unstack()
    contribution = contribution * scaler.std
    if add_mean:
        contribution = contribution + scaler.mean
    return contribution


def evaluate_mode_dynamics(
    opt_dmd: OptDMD, times: np.ndarray, mode_rank: int
) -> xr.DataArray:
    """Evaluate a single DMD mode's raw temporal dynamics
    (amplitude * exp(eigenvalue * t)) at `times`, with no spatial
    reconstruction at all - the same dimensionless per-mode quantity
    `plot_modes.py` plots as `opt_dmd.dynamics[mode_idx, :].real`, just
    evaluable at arbitrary times rather than only the model's own
    training-fit snapshots.

    `mode_rank` follows the same "Mode N" convention as
    `reconstruct_mode_contribution` (position when ranked by descending
    amplitude). Unlike that function, no conjugate pairing is applied -
    matching `plot_modes.py`, only the real part of this single raw mode's
    own complex dynamics is returned, with no `scaler.std`/`mean` and no
    spatial pattern involved.
    """
    mode_order = np.argsort(-np.abs(opt_dmd.amplitudes))
    idx = mode_order[mode_rank]
    t = (
        np.asarray(times, dtype="datetime64[ns]") - opt_dmd.time_fit[0]
    ) / np.timedelta64(1, opt_dmd.time_units)
    dynamics = opt_dmd.amplitudes[idx] * np.exp(opt_dmd.eigs[idx] * t)
    return xr.DataArray(
        dynamics.real, dims="time", coords={"time": np.asarray(times)}
    )


def evaluate_rmse(reconstruction: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    """Compute the lat-weighted RMSE between the reconstruction and the true
    data, as a function of time.

    The true data is restricted to the reconstruction's time/latitude/
    longitude grid first, since dropping NaN/land samples before fitting the
    SVD (see `standardising.stack_spatial_dims`) can remove entire
    latitude bands (e.g. Antarctica) from the reconstruction's spatial grid.
    """
    truth = truth.sel(
        time=reconstruction.time,
        latitude=reconstruction.latitude,
        longitude=reconstruction.longitude,
    )
    return compute_rmse(ground_truth=truth, prediction=reconstruction)


def save_rmse(
    rmse: xr.DataArray, output_dir: str, rmse_filename: str = "reconstruction_rmse.nc"
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rmse.to_netcdf(output_path / rmse_filename)
    logger.info("Saved reconstruction RMSE to %s.", output_path / rmse_filename)


def evaluate_rmse_map(reconstruction: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    """Compute the RMSE between the reconstruction and the true SST data,
    averaged over time instead of latitude/longitude, giving a map of where
    the reconstruction is best/worst.

    `lat_weighting` is turned off here: it exists to correct for grid-cell
    area when averaging spatially, not when averaging over time.
    """
    truth = truth.sel(
        time=reconstruction.time,
        latitude=reconstruction.latitude,
        longitude=reconstruction.longitude,
    )
    return compute_rmse(
        ground_truth=truth, prediction=reconstruction, lat_weighting=False, dims="time"
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
    fig: plt.Figure,
    output_dir: str,
    rmse_map_filename: str = "reconstruction_rmse_map.png",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / rmse_map_filename, dpi=150, bbox_inches="tight")
    logger.info("Saved reconstruction RMSE map to %s.", output_path / rmse_map_filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/evaluate_config.yaml",
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

        reconstruction = reconstruct_physical(opt_dmd, scaler)
        rmse = evaluate_rmse(reconstruction, truth).compute()

        logger.info(
            "Reconstruction RMSE (K): mean=%.4f, max=%.4f.",
            float(rmse.mean()),
            float(rmse.max()),
        )
        save_rmse(rmse, results_dir, config.get("rmse_filename", "reconstruction_rmse.nc"))

        rmse_map = evaluate_rmse_map(reconstruction, truth).compute()
        fig = plot_rmse_map(rmse_map, title="Time-averaged DMD reconstruction RMSE")
        save_rmse_map(
            fig,
            results_dir,
            config.get("rmse_map_filename", "reconstruction_rmse_map.png"),
        )
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
