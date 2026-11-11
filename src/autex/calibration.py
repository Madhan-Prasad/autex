"""Detector calibration, SUTR bias relaxation modeling, and slope fitting for autex."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from autex.logging import get_logger

logger = get_logger("calibration")


def remove_pedestal(cube: np.ndarray) -> np.ndarray:
    """Subtract the initial reset frame (D0) from all SUTR readout frames.

    Parameters
    ----------
    cube : np.ndarray
        3D array of shape (Nframes, Ny, Nx).

    Returns
    -------
    np.ndarray
        Pedestal-subtracted cube with frame 0 zeroed out.
    """
    logger.debug("Subtracting reset pedestal (D_i - D_0) across %d frames.", cube.shape[0])
    return cube - cube[0:1, :, :]


def split_quadrants(frame_or_cube: np.ndarray) -> list[np.ndarray]:
    """Slice a 1024x1024 detector array into its four 512x512 amplifier quadrants.

    Layout:
      Q1 (Bottom-Left) : [0:512, 0:512]
      Q2 (Top-Left)    : [512:1024, 0:512]
      Q3 (Bottom-Right): [0:512, 512:1024]
      Q4 (Top-Right)   : [512:1024, 512:1024]

    Parameters
    ----------
    frame_or_cube : np.ndarray
        2D image (1024, 1024) or 3D cube (Nframes, 1024, 1024).

    Returns
    -------
    list[np.ndarray]
        List containing [Q1, Q2, Q3, Q4].
    """
    if frame_or_cube.ndim == 2:
        q1 = frame_or_cube[0:512, 0:512]
        q2 = frame_or_cube[512:1024, 0:512]
        q3 = frame_or_cube[0:512, 512:1024]
        q4 = frame_or_cube[512:1024, 512:1024]
    elif frame_or_cube.ndim == 3:
        q1 = frame_or_cube[:, 0:512, 0:512]
        q2 = frame_or_cube[:, 512:1024, 0:512]
        q3 = frame_or_cube[:, 0:512, 512:1024]
        q4 = frame_or_cube[:, 512:1024, 512:1024]
    else:
        raise ValueError(f"Expected 2D or 3D array, got ndim={frame_or_cube.ndim}")

    return [q1, q2, q3, q4]


def assemble_quadrants(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray) -> np.ndarray:
    """Recombine four 512x512 quadrants back into a 1024x1024 array or cube.

    Parameters
    ----------
    q1, q2, q3, q4 : np.ndarray
        Individual amplifier quadrants.

    Returns
    -------
    np.ndarray
        Reconstructed 1024x1024 array or 3D cube.
    """
    axis_y = 0 if q1.ndim == 2 else 1
    axis_x = 1 if q1.ndim == 2 else 2

    left_half = np.concatenate([q1, q2], axis=axis_y)
    right_half = np.concatenate([q3, q4], axis=axis_y)
    return np.concatenate([left_half, right_half], axis=axis_x)


def interpolate_row_clock_delays(
    master_1d_curve: np.ndarray,
    num_rows: int = 512,
    extrapolate: bool = True,
) -> np.ndarray:
    """Resample master decay curve across sequential row-clock delay offsets.

    Parameters
    ----------
    master_1d_curve : np.ndarray
        1D temporal array representing master decay profile across frames (length N).
    num_rows : int, optional
        Number of physical detector rows per amplifier channel, by default 512.
    extrapolate : bool, optional
        Whether to extrapolate boundary frames, by default True.

    Returns
    -------
    np.ndarray
        2D array of shape (Nframes, num_rows) with row-specific time delays.
    """
    n_frames = len(master_1d_curve)
    t_nominal = np.arange(n_frames, dtype=np.float64)

    fill_val = "extrapolate" if extrapolate else (master_1d_curve[0], master_1d_curve[-1])
    spline = interp1d(t_nominal, master_1d_curve, kind="cubic", fill_value=fill_val)

    row_indices = np.arange(num_rows, dtype=np.float64)
    # Clock delay: Row 0 is read last (delay ~1.0), Row 511 first (delay ~0.0)
    delays = (num_rows - row_indices) / float(num_rows)

    # Broadcasted time grid: shape (Nframes, num_rows)
    t_grid = t_nominal[:, np.newaxis] + delays[np.newaxis, :]
    interpolated_model = spline(t_grid)

    logger.debug("Interpolated row clock delays: grid shape %s", interpolated_model.shape)
    return interpolated_model


def fit_sutr_slope(
    cube: np.ndarray,
    fit_slice: slice | None = None,
    time_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform vectorized linear least-squares regression along the temporal axis.

    Calculates flux count rate (beta, ADU/frame) and intercept (alpha) for every pixel.

    Parameters
    ----------
    cube : np.ndarray
        3D SUTR cube of shape (Nframes, Ny, Nx).
    fit_slice : slice | None, optional
        Frame index slice to fit across. Defaults to omitting the reset transient.
    time_array : np.ndarray | None, optional
        Optional explicit time timestamps. Defaults to frame indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (slope_map, intercept_map) each of shape (Ny, Nx).
    """
    n_frames = cube.shape[0]

    if time_array is None:
        time_array = np.arange(n_frames, dtype=np.float64)

    if fit_slice is None:
        if n_frames >= 50:
            fit_slice = slice(10, n_frames - 2)
        elif n_frames >= 8:
            fit_slice = slice(3, n_frames - 1)
        else:
            fit_slice = slice(0, n_frames)

    y = cube[fit_slice, :, :]
    x = time_array[fit_slice]
    n = len(x)

    if n < 2:
        logger.warning("Fewer than 2 frames available for slope fitting. Returning zeros.")
        return np.zeros(cube.shape[1:], dtype=np.float32), np.zeros(cube.shape[1:], dtype=np.float32)

    # Vectorized linear least-squares
    sx = np.sum(x)
    sxx = np.sum(x**2)
    sy = np.sum(y, axis=0)
    sxy = np.sum(x[:, np.newaxis, np.newaxis] * y, axis=0)

    denom = n * sxx - sx**2
    if denom == 0:
        return np.zeros(cube.shape[1:], dtype=np.float32), np.zeros(cube.shape[1:], dtype=np.float32)

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    return slope.astype(np.float32), intercept.astype(np.float32)
