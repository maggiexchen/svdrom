"""Compute each variable's DMD mode forcing field mean and standard
deviation, for use as the downstream model's (e.g. Aurora's) hardcoded
per-variable normalisation constants for this auxiliary input channel.

Usage
-----
    python forcing_field_stats.py --config configs/forcing_field_stats_config.yaml
"""

import argparse
import json
from pathlib import Path

import xarray as xr
import yaml

from svdrom.logger import setup_logger

logger = setup_logger("ForcingFieldStats", "forcing_field_stats.log")


def load_forcing_field(
    per_timestep_dir: str | None = None,
    input_file: str | None = None,
    file_pattern: str = "*.nc",
    time_chunk: int = 500,
) -> xr.DataArray:
    """Load a forcing field's values as a single DataArray, either from a
    directory of per-timestep files (as written by
    `mode_forcing_field.save_forcing_field_per_timestep`) or a single
    combined NetCDF (as written by `mode_forcing_field.save_forcing_field`).
    """
    if per_timestep_dir is not None:
        ds = xr.open_mfdataset(
            str(Path(per_timestep_dir) / file_pattern),
            combine="nested",
            concat_dim="time",
            parallel=True,
            engine="h5netcdf",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            chunks={"time": time_chunk},
        )
        return ds["dmd_mode_forcing_field"]
    if input_file is not None:
        return xr.open_dataarray(input_file, chunks={"time": time_chunk})
    msg = "Either 'per_timestep_dir' or 'input_file' must be given."
    raise ValueError(msg)


def compute_stats(field: xr.DataArray) -> tuple[float, float]:
    """Compute `field`'s (mean, std) across every dimension, ignoring NaNs
    (e.g. masked land points reinserted by `standardising.apply_nan_mask`).
    """
    mean = float(field.mean(skipna=True).compute())
    std = float(field.std(skipna=True).compute())
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/forcing_field_stats_config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    stats = {}
    for entry in config["variables"]:
        name = entry["name"]
        logger.info("Computing forcing field stats for %r.", name)
        field = load_forcing_field(
            per_timestep_dir=entry.get("per_timestep_dir"),
            input_file=entry.get("input_file"),
            file_pattern=entry.get("file_pattern", "*.nc"),
        )
        mean, std = compute_stats(field)
        stats[name] = {"mean": mean, "std": std}
        logger.info("%s: mean=%.8g, std=%.8g", name, mean, std)

    output_path = Path(config["output_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved forcing field stats to %s.", output_path)


if __name__ == "__main__":
    main()
