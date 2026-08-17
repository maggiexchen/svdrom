"""Resample a DataArray to a coarser, fixed-interval time grid."""

from datetime import timedelta

import xarray as xr

from svdrom.logger import setup_logger

logger = setup_logger("Resampling", "resampling.log")


def resample_snapshots(da: xr.DataArray, hours: int) -> xr.DataArray:
    """Resample `da` along `time` to a fixed interval, in hours, keeping the
    nearest original snapshot for each new timestamp.

    Parameters
    ----------
    da : xr.DataArray
        Data indexed along a `time` dimension.
    hours : int
        Desired spacing between snapshots, in hours.

    Returns
    -------
    xr.DataArray
        The resampled DataArray.
    """
    logger.info("Resampling to %d-hourly snapshots.", hours)
    return da.resample(time=timedelta(hours=hours)).nearest()
