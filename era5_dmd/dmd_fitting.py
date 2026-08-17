"""Fit Optimized DMD on the SVD factors."""

import pickle
from pathlib import Path

import numpy as np

from svdrom.dmd import OptDMD
from svdrom.logger import setup_logger
from svdrom.svd import TruncatedSVD

logger = setup_logger("DMDFitting", "dmd_fitting.log")


def fit_opt_dmd(
    rand_svd: TruncatedSVD,
    n_modes: int = 10,
    time_units: str = "h",
) -> OptDMD:
    """Fit Optimized DMD on the `u`, `s`, `v` factors of a fitted SVD.

    Returns
    -------
    OptDMD
        The fitted DMD model, exposing `.modes`, `.amplitudes`, `.eigs`,
        `.dynamics`.
    """
    logger.info("Fitting OptDMD with %d modes.", n_modes)
    opt_dmd = OptDMD(n_modes=n_modes, time_units=time_units)
    opt_dmd.fit(rand_svd.u, rand_svd.s, rand_svd.v)
    return opt_dmd


def save_dmd_results(
    opt_dmd: OptDMD,
    output_dir: str = "data/results/variable",
    dmd_filename: str | None = None,
) -> None:
    """Pickle the fitted DMD model alongside its time values.

    Parameters
    ----------
    dmd_filename : str | None, optional
        Name of the output file. Defaults to None, which names the file
        after the number of fitted modes, e.g. "optdmd_10_modes.pkl".
    """
    if dmd_filename is None:
        dmd_filename = f"optdmd_{opt_dmd.n_modes}_modes.pkl"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / dmd_filename, "wb") as f:
        pickle.dump(
            {
                "model": opt_dmd,
                "time_values": opt_dmd.dynamics.coords["time"].values,
            },
            f,
        )
    logger.info("Saved DMD results to %s.", output_path / dmd_filename)


def load_dmd_results(output_dir: str, dmd_filename: str) -> tuple[OptDMD, np.ndarray]:
    """Load a DMD model and its saved time values, as pickled by
    `save_dmd_results`.

    Returns
    -------
    tuple[OptDMD, np.ndarray]
        The fitted DMD model and the time values matching its `dynamics`.
    """
    path = Path(output_dir) / dmd_filename
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded DMD results from %s.", path)
    return data["model"], data["time_values"]


def realign_opt_dmd_time(opt_dmd: OptDMD, true_times: np.ndarray) -> OptDMD:
    """Correct, in place, an OptDMD model that was fit on a bare positional
    index (`time_fit` = 0, 1, 2, ...) instead of real calendar time - e.g.
    because its raw snapshot files carried no usable internal time
    coordinate (see `data_loading._expand_time`) - by relabelling it with
    the true timestamps of its training snapshots, without re-fitting.

    DMD's (mode, amplitude, eigenvalue) triple only depends on the
    *relative* spacing between fit snapshots, not on how that spacing
    happens to be labelled. If every snapshot is uniformly relabelled from
    "1 unit apart" to `true_step` apart, dividing the eigenvalues (and, if
    bagging was used, their standard deviation) by the ratio between the
    true and the originally-assumed spacing reproduces the exact same
    fitted dynamics when evaluated against true elapsed time instead of the
    positional index - the expensive SVD/DMD fit itself doesn't need to be
    redone. Modes and amplitudes are spatial/reference-time quantities and
    are unaffected by this relabelling.

    This assumes the training snapshots are (at least very close to)
    uniformly spaced in real time - like the original fit itself, which
    also implicitly assumed uniform spacing. A handful of missing snapshots
    in the raw archive (which would make the true spacing slightly
    irregular) were already invisible to the original fit and remain a
    (typically negligible) source of approximation here too; fully
    accounting for them would require re-fitting on the true, possibly
    irregular time vector.

    Parameters
    ----------
    opt_dmd : OptDMD
        A fitted model whose `time_fit` is not already real calendar time
        (`opt_dmd._is_datetime` is False). If it already is, this is a
        no-op.
    true_times : np.ndarray
        The true calendar timestamp of each training snapshot, in the same
        order as `opt_dmd.time_fit` (e.g. from
        `data_loading.infer_snapshot_times`). Must have the same length.

    Returns
    -------
    OptDMD
        The same `opt_dmd` instance, corrected in place.
    """
    if opt_dmd.time_fit is None or opt_dmd.eigs is None:
        msg = "The DMD model has not been fitted."
        raise RuntimeError(msg)
    if opt_dmd._is_datetime:
        logger.info(
            "Model's time_fit is already real calendar time; nothing to correct."
        )
        return opt_dmd
    if len(true_times) != len(opt_dmd.time_fit):
        msg = (
            "'true_times' must have the same length as the model's "
            f"time_fit ({len(opt_dmd.time_fit)}), got {len(true_times)}."
        )
        raise ValueError(msg)

    true_times = np.asarray(true_times, dtype="datetime64[ns]")
    true_deltas_h = np.diff(true_times) / np.timedelta64(1, opt_dmd.time_units)
    true_t_fit = np.concatenate(([0.0], np.cumsum(true_deltas_h)))

    # Ratio between the true snapshot spacing and the positional-index
    # spacing (1 unit) originally assumed during the fit.
    step_ratio = np.median(true_deltas_h) / np.median(np.diff(opt_dmd._t_fit))
    logger.info(
        "Rescaling eigenvalues by 1/%.3f to correct for the true median "
        "snapshot spacing (%.3f %s) versus the positional-index spacing "
        "assumed during the fit.",
        step_ratio,
        np.median(true_deltas_h),
        opt_dmd.time_units,
    )

    opt_dmd._eigs = opt_dmd._eigs / step_ratio
    if opt_dmd._eigs_std is not None:
        opt_dmd._eigs_std = opt_dmd._eigs_std / step_ratio
    opt_dmd._time_fit = true_times
    opt_dmd._t_fit = true_t_fit
    opt_dmd._is_datetime = True
    if opt_dmd._dynamics is not None:
        opt_dmd._dynamics = opt_dmd._dynamics.assign_coords(time=true_times)

    return opt_dmd
