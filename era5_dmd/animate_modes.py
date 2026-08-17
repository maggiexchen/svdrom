"""Animate the spatial reconstruction of individual DMD modes over time."""

import argparse
import os
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pipeline_utils import load_dmd_model_with_true_time
from plot_modes import rank_modes_by_amplitude
from svdrom.dmd import OptDMD
from svdrom.logger import setup_logger

logger = setup_logger("AnimateModes", "animate_modes.log")


def _format_time(time_value) -> str:
    if np.issubdtype(np.asarray(time_value).dtype, np.datetime64):
        return np.datetime_as_string(time_value, unit="h")
    return str(time_value)


def animate_mode(
    opt_dmd: OptDMD,
    time_values: np.ndarray,
    mode_rank: int,
    n_frames: int = 200,
    samples_per_cycle: int = 20,
    window_start: str | None = None,
    window_end: str | None = None,
    max_frames: int = 2000,
) -> animation.FuncAnimation:
    """Animate the spatial pattern of the mode at `mode_rank` (by descending
    amplitude) scaled by its own temporal dynamics,
    X_mode(t) = Phi_mode * dynamics_mode(t), real part only.

    `window_start`/`window_end` (e.g. "2012-01-01"/"2016-01-01") fix the
    calendar span shown, the same for every mode - independent of each
    mode's own period. Within that fixed span, the spacing between
    animation frames is chosen from the mode's own oscillation period,
    targeting `samples_per_cycle` frames per cycle, rather than a stride
    derived from the window's length. A stride tuned for a slow mode (e.g.
    annual) would badly undersample a fast mode (e.g. sub-monthly) -
    aliasing that makes it appear to evolve far more slowly than its true
    period. This means different modes end up with different total frame
    counts for the same calendar window (a fast mode simply has more cycles,
    and hence more frames, to show); `max_frames` caps this (widening the
    stride, at the cost of coarser per-cycle detail) to bound memory/file
    size in case a very fast mode would otherwise need an excessive number
    of frames. If neither `window_start` nor `window_end` is given, the full
    training record is used, and `n_frames` bounds the total instead
    (subsampled evenly, which may then be Nyquist-limited for fast modes
    over a long record - narrow the window instead for those). Non-
    oscillatory (purely growing/decaying) modes always just span the
    (possibly windowed) record evenly across `n_frames`.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    modes = opt_dmd.modes.unstack()
    mode_order = rank_modes_by_amplitude(opt_dmd)
    mode_idx = mode_order[mode_rank]

    time_values = np.asarray(time_values)
    n_time = opt_dmd.dynamics.shape[1]
    if window_start is not None or window_end is not None:
        mask = np.ones(n_time, dtype=bool)
        if window_start is not None:
            mask &= time_values >= np.datetime64(window_start)
        if window_end is not None:
            mask &= time_values <= np.datetime64(window_end)
        window_idx = np.nonzero(mask)[0]
        if len(window_idx) == 0:
            msg = (
                f"No training snapshots fall within the requested window "
                f"[{window_start}, {window_end}]."
            )
            raise ValueError(msg)
        start_i, end_i = int(window_idx[0]), int(window_idx[-1])
    else:
        start_i, end_i = 0, n_time - 1
    n_window = end_i - start_i + 1

    eig = opt_dmd.eigs[mode_idx]
    if abs(eig.imag) > 1e-12:
        dt = np.median(np.diff(opt_dmd._t_fit))  # avg snapshot spacing, opt_dmd.time_units
        period_steps = (2 * np.pi / abs(eig.imag)) / dt  # snapshots per cycle
        frame_stride = max(1, round(period_steps / samples_per_cycle))
    else:
        frame_stride = max(1, n_window // n_frames)

    time_idx = np.arange(start_i, end_i + 1, frame_stride)
    if len(time_idx) > max_frames:
        logger.warning(
            "Mode %d would need %d frames to hit samples_per_cycle=%d over "
            "the requested window; capping at max_frames=%d (coarser "
            "per-cycle detail).",
            mode_rank,
            len(time_idx),
            samples_per_cycle,
            max_frames,
        )
        frame_stride = max(1, round(n_window / max_frames))
        time_idx = np.arange(start_i, end_i + 1, frame_stride)
    frame_time_values = time_values[time_idx]

    lat = modes.coords["latitude"].values
    lon = modes.coords["longitude"].values

    spatial_pattern = modes[mode_idx, :, :].values
    dynamics_vals = opt_dmd.dynamics[mode_idx, time_idx].values
    recon_frames = (spatial_pattern[None, :, :] * dynamics_vals[:, None, None]).real

    vmax = np.nanpercentile(np.abs(recon_frames), 99)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines(color="black")
    ax.gridlines(draw_labels=True)

    mesh = ax.pcolormesh(
        lon,
        lat,
        recon_frames[0],
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label("Standardised anomaly")
    title = ax.set_title(f"Mode {mode_rank} — {_format_time(frame_time_values[0])}")

    def update(frame_idx):
        mesh.set_array(recon_frames[frame_idx].ravel())
        title.set_text(f"Mode {mode_rank} — {_format_time(frame_time_values[frame_idx])}")
        return mesh, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(time_idx), interval=150, blit=False
    )
    plt.close(fig)
    return anim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-config",
        default="plotting_config.yaml",
        help="Path to the plotting YAML config file.",
    )
    args = parser.parse_args()

    with open(args.plot_config) as f:
        plot_config = yaml.safe_load(f)
    animation_config = plot_config.get("animation", {})

    tmp_dir = animation_config.get("tempdir")
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        tempfile.tempdir = tmp_dir

    results_dir = plot_config["results_dir"]
    opt_dmd = load_dmd_model_with_true_time(
        results_dir, plot_config["dmd_filename"], plot_config.get("data")
    )
    time_values = opt_dmd.dynamics.coords["time"].values

    mode_ranks = plot_config.get("mode_ranks", [0, 2, 4, 6, 8])
    n_frames = animation_config.get("n_frames", 200)
    samples_per_cycle = animation_config.get("samples_per_cycle", 20)
    window_start = animation_config.get("window_start")
    window_end = animation_config.get("window_end")
    max_frames = animation_config.get("max_frames", 2000)
    fps = animation_config.get("fps", 8)
    dpi = animation_config.get("dpi", 150)
    filename_template = animation_config.get(
        "filename_template", "animation_dmd_mode{mode_rank}.gif"
    )

    for mode_rank in mode_ranks:
        anim = animate_mode(
            opt_dmd,
            time_values,
            mode_rank,
            n_frames=n_frames,
            samples_per_cycle=samples_per_cycle,
            window_start=window_start,
            window_end=window_end,
            max_frames=max_frames,
        )
        gif_path = Path(results_dir) / filename_template.format(mode_rank=mode_rank)
        anim.save(gif_path, writer="pillow", fps=fps, dpi=dpi)
        logger.info("Saved animation for mode %d to %s.", mode_rank, gif_path)


if __name__ == "__main__":
    main()
