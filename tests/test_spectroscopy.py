"""Automated unit tests for autex.spectroscopy module."""

import numpy as np
import pytest

from autex.spectroscopy import (
    calibrate_wavelength,
    correct_telluric_absorption,
    extract_optimal_spectrum,
    fit_spectral_trace,
    subtract_nod_pair,
)


def test_subtract_nod_pair():
    img_a = np.full((100, 100), 50.0, dtype=np.float32)
    img_b = np.full((100, 100), 20.0, dtype=np.float32)
    diff = subtract_nod_pair(img_a, img_b)
    assert np.allclose(diff, 30.0)


def test_fit_spectral_trace():
    ny, nx = 200, 256
    diff = np.zeros((ny, nx), dtype=np.float32)

    # Injected curved trace: y = 100 + 0.05 * x
    x_axis = np.arange(nx)
    y_true = 100.0 + 0.05 * x_axis

    for x in x_axis:
        cy = int(round(y_true[x]))
        diff[cy - 2 : cy + 3, x] = 100.0

    y_fitted = fit_spectral_trace(diff, search_window_y=(80, 130), poly_order=1)
    assert np.allclose(y_fitted, y_true, atol=0.8)


def test_extract_optimal_spectrum():
    ny, nx = 100, 200
    image = np.zeros((ny, nx), dtype=np.float32)
    y_trace = np.full(nx, fill_value=50.0)

    # Injected starlight continuum
    for x in range(nx):
        image[48:53, x] = 20.0  # Total flux per column = 100.0

    flux_1d, flux_err_1d = extract_optimal_spectrum(image, y_trace, aperture_radius=5)
    assert len(flux_1d) == nx
    assert np.isclose(np.median(flux_1d), 100.0, atol=5.0)
    assert np.all(flux_err_1d > 0.0)


def test_wavelength_and_telluric_correction():
    pixels = np.arange(100)
    # Linear dispersion: 1.0 um at x=0, 2.5 um at x=100
    coeffs = [0.015, 1.0]
    wavelengths = calibrate_wavelength(pixels, coeffs)
    assert np.isclose(wavelengths[0], 1.0)
    assert np.isclose(wavelengths[-1], 2.485)

    # Telluric correction test
    sci = np.full(100, 50.0)
    std = np.full(100, 10.0)
    corr = correct_telluric_absorption(sci, std, standard_bb_temperature=9500.0, wavelengths=wavelengths)
    assert len(corr) == 100
    assert not np.isnan(corr[50])
