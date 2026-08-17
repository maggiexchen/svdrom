"""Standardise a DataArray and flatten its spatial dimensions."""

from pathlib import Path

import numpy as np
import xarray as xr

from svdrom.logger import setup_logger
from svdrom.preprocessing import StandardScaler, variable_spatial_stack

logger = setup_logger("Standardising", "standardising.log")


def standardise(da: xr.DataArray) -> tuple[xr.DataArray, StandardScaler]:
    """Standardise `da` to zero mean (and unit standard deviation) along
    `time`, fitting a new StandardScaler.

    Returns
    -------
    tuple[xr.DataArray, StandardScaler]
        The standardised DataArray and the fitted scaler, so the latter can
        be reused (e.g. to un-standardise later results).
    """
    logger.info("Standardising data along the time dimension.")
    scaler = StandardScaler()
    normed_da = scaler(da, with_std=False)
    return normed_da, scaler


def unstandardise(da: xr.DataArray, scaler: StandardScaler) -> xr.DataArray:
    """Undo `standardise()`, multiplying by the fitted standard deviation
    (if used) and adding back the fitted mean, so data standardised with
    `scaler` can be compared against the original, physical-space data.
    """
    if scaler.with_std:
        da = da * scaler.std
    return da + scaler.mean


def stack_spatial_dims(
    da: xr.DataArray, dims: tuple[str, ...] = ("latitude", "longitude")
) -> xr.DataArray:
    """Stack `dims` into a single sample dimension, transpose so time is the
    trailing dimension, and drop any sample containing NaNs or infs (e.g.
    land points for an ocean-only variable such as SST).

    Returns
    -------
    xr.DataArray
        A 2D (samples x time) DataArray with no missing or infinite values.
    """
    logger.info("Stacking spatial dimensions %s.", dims)
    stacked = variable_spatial_stack(da, dims=dims)
    stacked_t = stacked.T
    stacked_t = stacked_t.where(np.isfinite(stacked_t))
    stacked_t = stacked_t.dropna(dim="samples", how="any")
    stacked_t = stacked_t.drop_vars("valid_time", errors="ignore")
    return stacked_t


def compute_nan_mask(da: xr.DataArray) -> xr.DataArray:
    """Compute the boolean mask of the samples `stack_spatial_dims` would
    drop: True wherever `da` is NaN or infinite at any time, on the full,
    unstacked spatial grid.
    """
    mask = (~np.isfinite(da)).any(dim="time")
    mask.name = "nan_mask"
    return mask


def save_nan_mask(
    da: xr.DataArray, output_dir: str, filename: str = "nan_mask.nc"
) -> Path:
    """Compute and save the NaN/land mask to a
    NetCDF file inside `output_dir`, so it can later be used to reinsert
    NaNs at those grid points into DMD output (e.g. `mode_forcing_field`)
    that was reconstructed only from the dropped-and-fitted samples.
    """
    mask = compute_nan_mask(da)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    mask.to_netcdf(file_path)
    logger.info("Saved NaN mask to %s.", file_path)
    return file_path


def apply_nan_mask(field: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Reinsert NaNs at the masked (e.g. land) grid points that
    `stack_spatial_dims` dropped before fitting, restoring `field` (as
    produced by unstacking DMD modes or a field built from them) onto
    `mask`'s full original spatial grid.

    `field` may be missing some of `mask`'s latitude/longitude values
    entirely rather than carrying them as per-point NaNs: an all-masked
    band (e.g. Antarctica for SST) disappears from the stacked index in
    `stack_spatial_dims` rather than surviving the drop with NaN values.
    `field` is therefore first reindexed onto `mask`'s full grid, then the
    mask is applied on top.
    """
    field = field.reindex_like(mask, copy=False)
    return field.where(~mask)
