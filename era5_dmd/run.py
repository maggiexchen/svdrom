"""Run the full ERA5 DMD pipeline: load, resample, standardise, stack,
fit SVD, then fit DMD on the SVD factors.

Usage
-----
    python run.py --config configs/config.yaml
"""

import argparse

import yaml
from data_loading import load_era5_variable
from dmd_fitting import fit_opt_dmd, save_dmd_results
from evaluate_reconstruction import evaluate_rmse, reconstruct_physical, save_rmse
from pipeline_utils import resolve_years, select, start_dask_client
from resampling import resample_snapshots
from standardising import save_nan_mask, stack_spatial_dims, standardise
from svd_fitting import fit_truncated_svd, save_svd_results

from svdrom.logger import setup_logger
from svdrom.preprocessing import StandardScaler

logger = setup_logger("Run", "run.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    svd_config = config.get("svd", {})
    dmd_config = config.get("dmd", {})
    output_config = config.get("output", {})

    client = start_dask_client(config.get("dask", {}))
    try:
        logger.info("Loading .nc files from %s.", data_config["variable_path"])
        da = load_era5_variable(
            years=resolve_years(data_config),
            **select(data_config, "variable_path", "file_pattern"),
        )

        resample_hours = data_config.get("resample_hours")
        if resample_hours is not None:
            da = resample_snapshots(da, hours=resample_hours)

        lat_range = data_config.get("lat_range")
        lon_range = data_config.get("lon_range")
        if lat_range is not None or lon_range is not None:
            logger.info(
                "Restricting to latitude=%s, longitude=%s.", lat_range, lon_range
            )
            da = da.sel(
                latitude=slice(*lat_range) if lat_range else slice(None),
                longitude=slice(*lon_range) if lon_range else slice(None),
            )

        normed_da, scaler = standardise(da)
        # Standardising disabled for this run: fit on the raw data
        # directly. `scaler` stays an identity (mean=0, no std scaling),
        # computing nothing, so save_svd_results/reconstruct_physical
        # below still run unchanged, in physical units.
        # normed_da = da
        # scaler = StandardScaler()
        # scaler._mean = 0

        if output_config.get("save_mask"):
            save_nan_mask(
                normed_da,
                output_config.get("output_dir", "data/results/variable"),
                output_config.get("mask_filename", "nan_mask.nc"),
            )

        stacked_da = stack_spatial_dims(normed_da)

        rand_svd = fit_truncated_svd(stacked_da, **svd_config)
        save_svd_results(rand_svd, scaler, **select(output_config, "output_dir"))

        opt_dmd = fit_opt_dmd(rand_svd, **dmd_config)
        save_dmd_results(opt_dmd, **select(output_config, "output_dir", "dmd_filename"))

        logger.info("Computing reconstruction RMSE against the true data.")
        reconstruction = reconstruct_physical(opt_dmd, scaler)
        rmse = evaluate_rmse(reconstruction, da).compute()
        logger.info(
            "Reconstruction RMSE: mean=%.4f, max=%.4f.",
            float(rmse.mean()),
            float(rmse.max()),
        )
        save_rmse(
            rmse,
            output_config.get("output_dir", "data/results/variable"),
            output_config.get("rmse_filename", "reconstruction_rmse.nc"),
        )

        logger.info(
            "Pipeline finished. Results written to %s.",
            output_config.get("output_dir", "data/results/variable"),
        )
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
