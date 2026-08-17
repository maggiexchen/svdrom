"""Load a variable's ERA5 .nc files into a single xarray DataArray."""

import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from svdrom.logger import setup_logger

logger = setup_logger("DataLoading", "data_loading.log")


def _infer_time_from_filename(source: str | None) -> np.datetime64 | None:
    """Parse a 'YYYY-MM-DD-HH' timestamp from a file's name, for raw files
    whose internal `time` dimension carries no usable coordinate values.
    Returns None if `source` is missing or doesn't match that format.
    """
    if not source:
        return None
    try:
        return np.datetime64(datetime.strptime(Path(source).stem, "%Y-%m-%d-%H"))
    except ValueError:
        return None


def _expand_time(ds: xr.Dataset) -> xr.Dataset:
    """Ensure `time` is a proper, real-datetime coordinate before individual
    snapshot files are concatenated.

    Most variables already carry `time` as a scalar coordinate, which is
    promoted here to a length-1 dimension. Some variables (e.g.
    top_net_thermal_radiation_24h_acc) instead have a bare, coordinate-less
    `time` dimension with no timestamp recorded anywhere inside the file -
    for these, the true timestamp is recovered from the file's name instead
    (files are named "YYYY-MM-DD-HH.nc").
    """
    if "time" in ds.coords and ds.time.ndim == 0:
        time_val = np.datetime64(ds.time.values)
        return ds.expand_dims(time=[time_val])
    if "time" in ds.dims and "time" not in ds.coords:
        time_val = _infer_time_from_filename(ds.encoding.get("source"))
        if time_val is None:
            msg = (
                "Dataset has an unlabelled 'time' dimension and no "
                "'YYYY-MM-DD-HH'-formatted source filename to recover a "
                f"timestamp from (source: {ds.encoding.get('source')!r})."
            )
            logger.exception(msg)
            raise ValueError(msg)
        return ds.assign_coords(time=[time_val])
    return ds


def infer_snapshot_times(
    variable_path: str,
    file_pattern: str = "*.nc",
    years: list[int] | None = None,
) -> np.ndarray:
    """Recover the true calendar timestamp of each of a variable's snapshot
    files by parsing filenames ("YYYY-MM-DD-HH.nc"), without opening any of
    them.

    Useful to recover the true elapsed time of an already-fitted DMD model
    whose raw files don't carry a usable internal time coordinate (see
    `_expand_time`), so its time axis can be corrected post-hoc without
    re-fitting (see `dmd_fitting.realign_opt_dmd_time`).
    """
    files = _list_era5_files(variable_path, file_pattern, years)
    times = [_infer_time_from_filename(f) for f in files]
    if any(t is None for t in times):
        msg = (
            "Could not parse a 'YYYY-MM-DD-HH' timestamp from every file "
            f"in {variable_path!r}."
        )
        logger.exception(msg)
        raise ValueError(msg)
    return np.array(times, dtype="datetime64[ns]")


def _list_era5_files(
    variable_path: str,
    file_pattern: str,
    years: list[int] | None,
) -> list[str]:
    files = sorted(glob.glob(os.path.join(variable_path, file_pattern)))
    if not files:
        msg = f"No files matching {file_pattern!r} found in {variable_path!r}."
        raise FileNotFoundError(msg)

    if years is not None:
        year_prefixes = tuple(str(year) for year in years)
        files = [f for f in files if os.path.basename(f).startswith(year_prefixes)]
        if not files:
            msg = f"No files for years {years} found in {variable_path!r}."
            raise FileNotFoundError(msg)

    logger.info("Found %d files in %s.", len(files), variable_path)
    return files


def _open_batch(files: list[str], time_chunk: int) -> xr.Dataset:
    return xr.open_mfdataset(
        files,
        preprocess=_expand_time,
        combine="nested",
        concat_dim="time",
        parallel=True,
        engine="h5netcdf",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks={"time": time_chunk},
    )


def load_era5_variable(
    variable_path: str,
    file_pattern: str = "*.nc",
    years: list[int] | None = None,
    time_chunk: int = 1,
    n_batches: int = 10,
) -> xr.DataArray:
    """Open all matching .nc files for one ERA5 variable as a single
    time-concatenated, Dask-backed DataArray.

    Files are opened in `n_batches` groups, each in parallel via h5netcdf,
    then concatenated. This keeps the underlying Dask graph small enough to
    build quickly even for tens of thousands of files.

    Parameters
    ----------
    variable_path : str
        Directory containing the per-timestep .nc files for the variable,
        e.g. ".../sea_surface_temperature/sea_surface_temperature/data".
    file_pattern : str, optional
        Glob pattern (relative to `variable_path`) selecting which files to
        load, independent of year. Defaults to "*.nc".
    years : list[int] | None, optional
        If given, only load files whose name starts with one of these
        years (filenames are assumed to start with "YYYY..."). This is the
        only year filter — `file_pattern` should not itself encode years.
        Defaults to None, which loads every file matching `file_pattern`.
    time_chunk : int, optional
        Dask chunk size along the `time` dimension. Defaults to 1.
    n_batches : int, optional
        Number of batches to split the file list into before opening each
        batch in parallel. Defaults to 10.

    Returns
    -------
    xr.DataArray
        The variable's data, concatenated along `time`.
    """
    files = _list_era5_files(variable_path, file_pattern, years)

    batches = np.array_split(files, min(n_batches, len(files)))
    datasets = []
    for i, batch in enumerate(batches, start=1):
        logger.info("Reading batch %d out of %d...", i, len(batches))
        datasets.append(_open_batch(list(batch), time_chunk))

    dataset = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    variable_name = next(iter(dataset.data_vars))
    logger.info("Loaded variable %r.", variable_name)
    return dataset[variable_name]
