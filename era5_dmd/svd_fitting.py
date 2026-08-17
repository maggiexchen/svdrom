"""Fit a truncated SVD on the standardised, stacked data."""

import pickle
from pathlib import Path

import xarray as xr

from svdrom.logger import setup_logger
from svdrom.svd import TruncatedSVD

logger = setup_logger("SVDFitting", "svd_fitting.log")


def fit_truncated_svd(
    stacked_da: xr.DataArray,
    n_components: int = 40,
    n_power_iter: int = 2,
    n_oversamples: int = 15,
) -> TruncatedSVD:
    """Fit a randomized, truncated SVD on a (samples x time) DataArray.

    Returns
    -------
    TruncatedSVD
        The fitted decomposition, exposing `.u`, `.s`, `.v`.
    """
    logger.info("Fitting truncated SVD with %d components.", n_components)
    rand_svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        rechunk=True,
        compute_var_ratio=True,
    )
    rand_svd.fit(stacked_da, n_power_iter=n_power_iter, n_oversamples=n_oversamples)
    return rand_svd


def save_svd_results(
    rand_svd: TruncatedSVD, scaler, output_dir: str = "data/results/sst"
) -> None:
    """Pickle the fitted SVD and the StandardScaler used upstream, so both
    can be reloaded without repeating the (expensive) fitting steps.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "rand_svd_results.pkl", "wb") as f:
        pickle.dump(rand_svd, f)
    with open(output_path / "standard_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved SVD results and scaler to %s.", output_path)
