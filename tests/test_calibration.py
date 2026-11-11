"""Automated unit tests for autex.calibration module."""

import numpy as np
import pytest

from autex.calibration import (
    assemble_quadrants,
    fit_sutr_slope,
    interpolate_row_clock_delays,
    remove_pedestal,
    split_quadrants,
)


def test_pedestal_subtraction():
    # Synthetic cube with static pedestal of 5000 ADU
    cube = np.ones((10, 100, 100), dtype=np.float32) * 5000.0
    for i in range(10):
        cube[i] += i * 10.0  # Linear ramp

    ped_removed = remove_pedestal(cube)
    assert np.allclose(ped_removed[0], 0.0)
    assert np.allclose(ped_removed[5], 50.0)


def test_quadrant_splitting_and_assembly():
    full_frame = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    quadrants = split_quadrants(full_frame)

    assert len(quadrants) == 4
    for q in quadrants:
        assert q.shape == (512, 512)

    reconstructed = assemble_quadrants(*quadrants)
    assert np.array_equal(full_frame, reconstructed)


def test_row_clock_delay_interpolation():
    # 20 frames decay curve
    t = np.arange(20, dtype=np.float64)
    curve = 1000.0 * np.exp(-t / 5.0)

    interpolated = interpolate_row_clock_delays(curve, num_rows=512)
    assert interpolated.shape == (20, 512)

    # Row 0 should lag behind Row 511
    assert interpolated[5, 0] != interpolated[5, 511]


def test_sutr_slope_fitting():
    n_frames, ny, nx = 20, 50, 50
    cube = np.zeros((n_frames, ny, nx), dtype=np.float32)

    true_slope = 7.5
    true_intercept = 120.0

    for i in range(n_frames):
        cube[i, :, :] = true_intercept + true_slope * i

    slope_map, intercept_map = fit_sutr_slope(cube)
    assert slope_map.shape == (ny, nx)
    assert np.allclose(slope_map, true_slope, atol=1e-4)
    assert np.allclose(intercept_map, true_intercept, atol=1e-4)
