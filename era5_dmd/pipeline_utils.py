"""Shared helpers for the ERA5 DMD pipeline scripts (run.py,
evaluate_reconstruction.py, evaluate_forecast.py, mode_forcing_field.py).
"""

import os
from collections.abc import Callable
from pathlib import Path

import xarray as xr
from dask.distributed import Client

from data_loading import infer_snapshot_times
from dmd_fitting import load_dmd_results, realign_opt_dmd_time

from svdrom.dmd import OptDMD
from svdrom.logger import setup_logger

logger = setup_logger("PipelineUtils", "pipeline_utils.log")


def select(config: dict, *keys: str) -> dict:
    """Restrict `config` to `keys` that are actually present, so it can be
    unpacked into a function call without erroring on unrelated keys or
    overriding that function's own defaults with missing/null values.
    """
    return {k: config[k] for k in keys if k in config}


def resolve_years(data_config: dict) -> list[int] | None:
    """Return the explicit `years` list if given, otherwise the inclusive
    `start_year`-`end_year` range, otherwise None (load every file).
    """
    if data_config.get("years") is not None:
        return list(data_config["years"])
    start_year = data_config.get("start_year")
    end_year = data_config.get("end_year")
    if start_year is not None and end_year is not None:
        return list(range(start_year, end_year + 1))
    return None


def load_dmd_model_with_true_time(
    results_dir: str,
    dmd_filename: str,
    data_config: dict | None = None,
) -> OptDMD:
    """Load a fitted OptDMD model, correcting its time axis in place (see
    `dmd_fitting.realign_opt_dmd_time`) if it was fit on a bare positional
    index rather than real calendar time - which happens when its raw
    snapshot files carry no usable internal time coordinate (e.g.
    top_net_thermal_radiation_24h_acc; see `data_loading._expand_time`).

    `data_config` (needing at least `variable_path`, plus optionally
    `file_pattern`/`years`/`start_year`/`end_year`) is required in that
    case, to recover the true snapshot timestamps from the raw filenames.
    It's unused, and may be omitted, if the model already has real
    calendar time.
    """
    opt_dmd, _ = load_dmd_results(results_dir, dmd_filename)
    if opt_dmd._is_datetime:
        return opt_dmd
    if data_config is None:
        msg = (
            "This model's time_fit is not real calendar time (its raw "
            "snapshot files carry no usable internal time coordinate), and "
            "no 'data' config section was given to recover the true "
            "timestamps from filenames - pass one (see "
            "data_loading.infer_snapshot_times)."
        )
        logger.exception(msg)
        raise ValueError(msg)
    true_times = infer_snapshot_times(
        data_config["variable_path"],
        data_config.get("file_pattern", "*.nc"),
        resolve_years(data_config),
    )
    return realign_opt_dmd_time(opt_dmd, true_times)


def load_or_compute_dataarray(
    path: str | Path, compute_fn: Callable[[], xr.DataArray]
) -> xr.DataArray:
    """Load a cached DataArray from `path` if it already exists, otherwise
    compute it via `compute_fn()`, save it to `path` as NetCDF, and return
    it - so an expensive DMD reconstruction/forecast doesn't need to be
    recomputed every time a script re-runs over the same time period (e.g.
    while iterating on a plot's styling).

    Parameters
    ----------
    path : str | Path
        Where to look for/save the cached NetCDF file.
    compute_fn : Callable[[], xr.DataArray]
        Zero-argument function that computes the (possibly Dask-backed)
        DataArray if no cache is found.

    Notes
    -----
    Both branches keep the result Dask-backed rather than loading it
    eagerly into plain memory, for the same reason: a large (multi-GB)
    field, if gathered onto a single worker at once (e.g. via an eager
    `.compute()`, or by opening the cached file without `chunks=`), can
    exceed that worker's `memory_limit` even though the cluster as a whole
    has plenty of memory - especially once it's combined with another
    Dask-backed array (e.g. climatology) in a further computation. Reading
    and writing chunk-by-chunk instead avoids ever materializing the whole
    array in one place.

    `chunks={}` (auto-detect the on-disk chunking) isn't enough on its
    own: NetCDF files written by `to_netcdf()` without an explicit
    `encoding` are typically stored contiguously (no internal chunk
    structure), so there's nothing for `{}` to detect and it falls back to
    a single whole-array chunk anyway - hence the explicit `time` chunk
    size below, applied on every load regardless of how the file happens
    to be stored on disk.
    """
    path = Path(path)
    if path.exists():
        logger.info("Loading cached result from %s.", path)
        return xr.open_dataarray(path, chunks={"time": 200})

    result = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(path)
    logger.info("Saved result to %s.", path)
    return result


def start_dask_client(dask_config: dict) -> Client | None:
    """Start a local Dask distributed client for the pipeline's Dask-backed
    computations (loading, standardising, fitting, saving), or return None
    if no `dask` config section is given.
    """
    if not dask_config:
        return None
    local_directory = dask_config.get("local_directory")
    if local_directory:
        os.makedirs(local_directory, exist_ok=True)
    client = Client(**dask_config)
    logger.info("Started Dask client. Dashboard: %s", client.dashboard_link)
    return client
