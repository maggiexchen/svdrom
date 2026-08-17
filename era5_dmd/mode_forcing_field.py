"""Generate the physical-space forcing field of a single DMD mode, for use
as an auxiliary input channel to a downstream forecasting model (e.g.
Aurora).

Usage
-----
    python mode_forcing_field.py --config configs/forcing_field_config.yaml
"""

import argparse
import pickle
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from pipeline_utils import load_dmd_model_with_true_time
from plot_modes import rank_modes_by_amplitude
from standardising import apply_nan_mask

from svdrom.dmd import OptDMD
from svdrom.logger import setup_logger
from svdrom.preprocessing import StandardScaler

logger = setup_logger("ModeForcingField", "mode_forcing_field.log")


def times_to_model_units(opt_dmd: OptDMD, times: np.ndarray) -> np.ndarray:
    """Convert real datetimes into the elapsed-time-since-first-training-
    snapshot float representation that the mode's `exp(eig * t)` dynamics
    expect, using the same reference point (`opt_dmd.time_fit[0]`) and units
    (`opt_dmd.time_units`) as the DMD fit. Mirrors
    `evaluate_forecast.times_to_model_units`.
    """
    time_fit = opt_dmd.time_fit
    if time_fit is None:
        msg = "The DMD model has not been fitted."
        raise RuntimeError(msg)
    deltas = (times - time_fit[0]).astype(f"timedelta64[{opt_dmd.time_units}]")
    return deltas / np.timedelta64(1, opt_dmd.time_units)


def build_target_times(
    opt_dmd: OptDMD,
    freq: str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> np.ndarray:
    """Build the timestamps at which to evaluate a mode's dynamics.

    If `start_date` and/or `end_date` are given, generate a fixed-step grid
    over `[start_date, end_date]` (inclusive) at `freq` instead of the fit's
    own period - the mode's analytic `amplitude * exp(eig * t)` dynamics are
    defined at any `t`, so this also covers dates outside the fitted period
    (extrapolation). `freq` is required in this case, since there's no
    "native" cadence to fall back to for an arbitrary range.

    Otherwise: if `freq` is None, reuse the model's native (daily) fit
    timestamps. If `freq` is a pandas frequency string (e.g. "6h"), generate
    a fixed-step grid spanning the same period as the fit, so the forcing
    field lines up with other input fields sampled at that cadence (e.g. the
    raw, natively 6-hourly SST files used elsewhere as Aurora inputs).
    """
    if start_date is not None or end_date is not None:
        if freq is None:
            msg = "'target_freq' must be given when 'start_date'/'end_date' are given."
            raise ValueError(msg)
        start = pd.Timestamp(start_date) if start_date is not None else pd.Timestamp(opt_dmd.time_fit[0])
        end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(opt_dmd.time_fit[-1])
        return pd.date_range(start=start, end=end, freq=freq).values
    if freq is None:
        return opt_dmd.time_fit
    start = pd.Timestamp(opt_dmd.time_fit[0])
    end = pd.Timestamp(opt_dmd.time_fit[-1]) + pd.Timedelta(days=1) - pd.Timedelta(freq)
    return pd.date_range(start=start, end=end, freq=freq).values


def mode_forcing_field(
    opt_dmd: OptDMD,
    scaler: StandardScaler,
    mode_rank: int,
    target_freq: str | None = None,
    time_chunk_size: int = 2000,
    mask: xr.DataArray | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> xr.DataArray:
    """Reconstruct the physical-space, time-varying contribution of a single
    DMD mode, ranked `mode_rank` by descending amplitude (0 = strongest).

    Only the fitted standard deviation is used to rescale the mode's
    contribution to physical units; the climatological mean is *not* added
    back, since it is shared by every mode and isn't part of any single
    mode's contribution. The result is a physical-units (e.g. Kelvin)
    anomaly field driven by this mode alone, suitable as a forcing field
    rather than a standalone absolute field.

    Complex DMD modes fitted on real-valued data come in conjugate pairs
    with identical amplitude (and hence adjacent ranks), and taking the real
    part of either mode in a pair gives the same field, since Re(z) ==
    Re(conj(z)). A warning is logged if `mode_rank` is the conjugate partner
    of a stronger (lower-ranked) mode.

    `target_freq` controls the output cadence. Leaving it None reuses
    `opt_dmd.dynamics`'s native fit timestamps (e.g. daily for the SST
    model). A mode's time evolution is the analytic formula
    `amplitude * exp(eig * t)`, defined at any `t`, so passing a pandas
    frequency string (e.g. "6h") re-evaluates that formula directly at a
    finer, fixed-step grid instead of interpolating the native field -
    matching the cadence of other input fields it needs to sit alongside
    (e.g. raw 6-hourly SST or ttr_24hr) rather than only providing values at
    the fit's own native timestamps.

    The full (time x latitude x longitude) field can be tens of GB at
    sub-daily cadence, so the time coefficient is built as a Dask array
    chunked into blocks of `time_chunk_size` steps: the outer product with
    the (small) spatial pattern, and the final write to disk, are then
    computed one chunk at a time instead of all at once.

    `mask`, if given (as saved by `standardising.save_nan_mask`), is used to
    reinsert NaNs at the masked (e.g. land) grid points that were dropped
    before the SVD/DMD fit (see `standardising.stack_spatial_dims`) -
    including whole bands (e.g. Antarctica) that dropped out of the fitted
    grid entirely rather than surviving as per-point NaNs. Leave it None to
    return the field on whatever (possibly reduced) grid the fit's modes
    cover.
    """
    mode_order = rank_modes_by_amplitude(opt_dmd)
    if not 0 <= mode_rank < len(mode_order):
        msg = f"'mode_rank' must be between 0 and {len(mode_order) - 1}."
        raise ValueError(msg)
    mode_idx = mode_order[mode_rank]

    for other_rank in range(mode_rank):
        other_idx = mode_order[other_rank]
        if np.isclose(opt_dmd.eigs[mode_idx], np.conj(opt_dmd.eigs[other_idx])):
            logger.warning(
                "Mode rank %d (index %d) is the complex-conjugate partner of "
                "rank %d (index %d); its real-valued forcing field will be "
                "identical to rank %d's.",
                mode_rank,
                mode_idx,
                other_rank,
                other_idx,
                other_rank,
            )

    times = build_target_times(opt_dmd, target_freq, start_date, end_date)
    t = da.from_array(times_to_model_units(opt_dmd, times), chunks=time_chunk_size)
    coefficient = opt_dmd.amplitudes[mode_idx] * da.exp(opt_dmd.eigs[mode_idx] * t)
    coefficient_da = xr.DataArray(coefficient, dims="time", coords={"time": times})

    modes = opt_dmd.modes.unstack()
    field = (modes.isel(components=mode_idx) * coefficient_da).real
    field = field.transpose("time", "latitude", "longitude")

    if scaler.with_std:
        field = field * scaler.std

    if mask is not None:
        field = apply_nan_mask(field, mask)

    field.name = "dmd_mode_forcing_field"
    field.attrs.update(
        mode_rank=mode_rank,
        mode_index=int(mode_idx),
        eigenvalue_real=float(opt_dmd.eigs[mode_idx].real),
        eigenvalue_imag=float(opt_dmd.eigs[mode_idx].imag),
        amplitude=float(np.abs(opt_dmd.amplitudes[mode_idx])),
        description=(
            "Single DMD mode's spatio-temporal contribution, rescaled by the "
            "fitted standard deviation (climatological mean not added back)."
        ),
        target_freq=target_freq or "native (daily fit timestamps)",
    )
    return field


def save_forcing_field(
    field: xr.DataArray,
    output_dir: str,
    filename_template: str,
    mode_rank: int,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename_template.format(mode_rank=mode_rank)
    field.to_netcdf(file_path)
    logger.info("Saved mode %d forcing field to %s.", mode_rank, file_path)
    return file_path


def save_forcing_field_per_timestep(
    opt_dmd: OptDMD,
    scaler: StandardScaler,
    mode_rank: int,
    output_dir: str,
    target_freq: str | None = None,
    filename_format: str = "%Y-%m-%d-%H",
    dtype: str = "float32",
    log_every: int = 2000,
    mask: xr.DataArray | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    """Write a single DMD mode's forcing field as one file per timestep,
    named and structured like the raw ERA5 input files (e.g.
    ".../sea_surface_temperature/data/2025-06-30-00.nc"): dimensions
    (latitude, longitude) only, with `time`/`valid_time`/`number`/`step`/
    `surface` as scalar coordinates, rather than a single file with a `time`
    dimension. This keeps the forcing field structurally coherent with the
    other per-timestep input fields it will sit alongside (e.g. as an
    auxiliary Aurora input).

    Only one (latitude, longitude) slice is ever held in memory at a time,
    so this scales to any number of timesteps regardless of available RAM -
    unlike `mode_forcing_field()`, which builds the full (time x latitude x
    longitude) array.

    `mask`, if given (as saved by `standardising.save_nan_mask`), is used to
    reinsert NaNs at the masked (e.g. land) grid points dropped before the
    fit, as in `mode_forcing_field()` - see its docstring for details.
    """
    mode_order = rank_modes_by_amplitude(opt_dmd)
    if not 0 <= mode_rank < len(mode_order):
        msg = f"'mode_rank' must be between 0 and {len(mode_order) - 1}."
        raise ValueError(msg)
    mode_idx = mode_order[mode_rank]

    for other_rank in range(mode_rank):
        other_idx = mode_order[other_rank]
        if np.isclose(opt_dmd.eigs[mode_idx], np.conj(opt_dmd.eigs[other_idx])):
            logger.warning(
                "Mode rank %d (index %d) is the complex-conjugate partner of "
                "rank %d (index %d); its real-valued forcing field will be "
                "identical to rank %d's.",
                mode_rank,
                mode_idx,
                other_rank,
                other_idx,
                other_rank,
            )

    times = build_target_times(opt_dmd, target_freq, start_date, end_date)
    t = times_to_model_units(opt_dmd, times)
    coefficients = opt_dmd.amplitudes[mode_idx] * np.exp(opt_dmd.eigs[mode_idx] * t)

    modes = opt_dmd.modes.unstack()
    spatial_pattern = modes.isel(components=mode_idx)
    if mask is not None:
        spatial_pattern = apply_nan_mask(spatial_pattern, mask)
    spatial_values = spatial_pattern.values  # (latitude, longitude), complex, static

    std_grid = None
    if scaler.with_std:
        std_grid = scaler.std.sel(
            latitude=spatial_pattern["latitude"], longitude=spatial_pattern["longitude"]
        ).values

    lat = spatial_pattern["latitude"].values
    lon = spatial_pattern["longitude"].values
    # Pick up whatever scalar (0-d) coords the variable's modes happen to
    # carry (e.g. "number", "step", "surface" for SST; "number", "surface"
    # only for ttr_24hr, which has no "step") rather than assuming a fixed
    # set - not every variable's raw files carry the same scalar metadata.
    scalar_coords = {
        c: v.values for c, v in spatial_pattern.coords.items() if v.ndim == 0
    }

    attrs = dict(
        mode_rank=mode_rank,
        mode_index=int(mode_idx),
        eigenvalue_real=float(opt_dmd.eigs[mode_idx].real),
        eigenvalue_imag=float(opt_dmd.eigs[mode_idx].imag),
        amplitude=float(np.abs(opt_dmd.amplitudes[mode_idx])),
        description=(
            "Single DMD mode's spatio-temporal contribution, rescaled by the "
            "fitted standard deviation (climatological mean not added back)."
        ),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, (time_val, coeff) in enumerate(zip(times, coefficients)):
        field_2d = (spatial_values * coeff).real
        if std_grid is not None:
            field_2d = field_2d * std_grid
        field_2d = field_2d.astype(dtype)

        ds_out = xr.Dataset(
            {"dmd_mode_forcing_field": (("latitude", "longitude"), field_2d)},
            coords={
                "latitude": lat,
                "longitude": lon,
                "time": time_val,
                "valid_time": time_val,
                **scalar_coords,
            },
            attrs=attrs,
        )
        file_path = output_path / f"{pd.Timestamp(time_val).strftime(filename_format)}.nc"
        ds_out.to_netcdf(file_path, engine="h5netcdf")

        if (i + 1) % log_every == 0 or i == len(times) - 1:
            logger.info(
                "Wrote %d/%d per-timestep forcing field files to %s.",
                i + 1,
                len(times),
                output_path,
            )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/forcing_field_config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = config["results_dir"]
    mode_rank = config.get("mode_rank", 1)
    target_freq = config.get("target_freq")
    start_date = config.get("start_date")
    end_date = config.get("end_date")

    opt_dmd = load_dmd_model_with_true_time(
        results_dir, config["dmd_filename"], config.get("data")
    )
    with open(Path(results_dir) / "standard_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    mask_path = Path(results_dir) / config.get("mask_filename", "nan_mask.nc")
    mask = None
    if mask_path.exists():
        mask = xr.open_dataarray(mask_path)
        logger.info("Loaded NaN mask from %s.", mask_path)
    else:
        logger.info(
            "No NaN mask found at %s; land points will not be re-inserted.",
            mask_path,
        )

    per_timestep_dir = config.get("per_timestep_dir")
    if per_timestep_dir:
        save_forcing_field_per_timestep(
            opt_dmd,
            scaler,
            mode_rank,
            per_timestep_dir,
            target_freq=target_freq,
            filename_format=config.get("filename_format", "%Y-%m-%d-%H"),
            mask=mask,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        field = mode_forcing_field(
            opt_dmd,
            scaler,
            mode_rank,
            target_freq=target_freq,
            mask=mask,
            start_date=start_date,
            end_date=end_date,
        )
        save_forcing_field(
            field,
            results_dir,
            config.get("filename_template", "forcing_field_mode{mode_rank}.nc"),
            mode_rank,
        )


if __name__ == "__main__":
    main()
