"""Load a precomputed climatology (as saved under
/data/data_maggie/era5_regridded_1deg/.../climatology_*/data, one file per
day-of-year/hour-of-day combination) and compute the anomaly of a field
(e.g. a DMD reconstruction) relative to it.
"""

import re
from pathlib import Path

import xarray as xr

from svdrom.logger import setup_logger

logger = setup_logger("Climatology", "climatology.log")

_FILENAME_RE = re.compile(r"(\d{3})-(\d{2})\.nc$")


def _assign_dayofyear_hour(ds: xr.Dataset) -> xr.Dataset:
    """Parse the "<dayofyear>-<hour>.nc" filename into `dayofyear`/`hour`
    coordinates, so per-file datasets can be combined by `xr.open_mfdataset`
    into a single (dayofyear, hour, ...) climatology DataArray.
    """
    source = ds.encoding.get("source")
    match = _FILENAME_RE.search(source) if source else None
    if match is None:
        msg = (
            'Expected a "<dayofyear>-<hour>.nc"-named climatology file '
            f"(got source: {source!r})."
        )
        raise ValueError(msg)
    dayofyear, hour = int(match.group(1)), int(match.group(2))
    return ds.expand_dims(dayofyear=[dayofyear], hour=[hour])


def load_climatology(
    climatology_dir: str,
    variable: str = "sst",
    std_variable: str | None = "sst_std",
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """Load a directory of per-(dayofyear, hour) climatology files into a
    single DataArray, in the same (dayofyear, hour, latitude, longitude)
    layout produced in-memory by `svdrom.weather_utils.compute_climatology`.

    Parameters
    ----------
    climatology_dir : str
        Directory containing the per-day-of-year/hour-of-day .nc files, e.g.
        ".../sea_surface_temperature/climatology_1990_2019/data".
    variable : str, optional
        Name of the climatological mean variable inside each file. Defaults
        to "sst".
    std_variable : str | None, optional
        Name of the climatological standard deviation variable inside each
        file. Set to None if not needed. Defaults to "sst_std".

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        The climatological mean, or (mean, std) if `std_variable` is given.
        Both are Dask-backed, indexed by (dayofyear, hour, latitude,
        longitude).
    """
    files = sorted(str(f) for f in Path(climatology_dir).glob("*.nc"))
    files = [f for f in files if _FILENAME_RE.search(f)]
    if not files:
        msg = f'No "<dayofyear>-<hour>.nc" files found in {climatology_dir!r}.'
        raise FileNotFoundError(msg)

    logger.info("Loading %d climatology files from %s.", len(files), climatology_dir)
    ds = xr.open_mfdataset(
        files,
        preprocess=_assign_dayofyear_hour,
        combine="by_coords",
        engine="h5netcdf",
    )
    if std_variable is None:
        return ds[variable]
    return ds[variable], ds[std_variable]


def compute_anomaly(field: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
    """Subtract `climatology` from `field`, matching each of `field`'s
    timestamps to its climatological value by day-of-year and hour-of-day.

    Parameters
    ----------
    field : xr.DataArray
        The field to compute the anomaly of (e.g. a DMD reconstruction),
        with a `time` dimension.
    climatology : xr.DataArray
        The climatology as a function of `dayofyear` and `hour`, such as
        returned by `load_climatology` (or
        `svdrom.weather_utils.compute_climatology`).

    Returns
    -------
    xr.DataArray
        `field`'s anomaly relative to the climatology, on `field`'s original
        time/latitude/longitude grid.
    """
    matched_climatology = climatology.sel(
        dayofyear=field.time.dt.dayofyear, hour=field.time.dt.hour
    )
    return field - matched_climatology
