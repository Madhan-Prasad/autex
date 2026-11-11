"""Automated unit tests for autex.photometry module."""

import numpy as np
import pytest

from autex.photometry import (
    detect_sources,
    estimate_background_2d,
    extract_aperture_photometry,
    measure_empirical_fwhm,
)


def test_estimate_background_2d():
    ny, nx = 256, 256
    true_sky = 25.0
    image = np.full((ny, nx), fill_value=true_sky, dtype=np.float32)
    # Add random noise
    rng = np.random.default_rng(42)
    image += rng.normal(0.0, 1.0, size=(ny, nx)).astype(np.float32)

    bkg_map, rms_map, sky_median = estimate_background_2d(image, box_size=(32, 32))
    assert np.isclose(sky_median, true_sky, atol=0.2)
    assert bkg_map.shape == (ny, nx)
    assert np.isclose(np.median(rms_map), 1.0, atol=0.2)


def test_source_detection_and_fwhm():
    ny, nx = 256, 256
    image = np.zeros((ny, nx), dtype=np.float32)

    # Inject synthetic Gaussian stars
    y, x = np.mgrid[0:ny, 0:nx]
    # Star 1 at (100, 100), Star 2 at (180, 150)
    sigma = 3.0 / 2.355  # FWHM = 3.0
    star1 = 500.0 * np.exp(-((x - 100) ** 2 + (y - 100) ** 2) / (2 * sigma ** 2))
    star2 = 300.0 * np.exp(-((x - 180) ** 2 + (y - 150) ** 2) / (2 * sigma ** 2))
    image += star1 + star2

    sources = detect_sources(image, bkg_rms=1.0, expected_fwhm=3.0, threshold_sigma=4.0)
    assert len(sources) >= 2

    # Verify positions
    xs = [s.x for s in sources]
    ys = [s.y for s in sources]
    assert any(abs(x_pos - 100) < 1.0 and abs(y_pos - 100) < 1.0 for x_pos, y_pos in zip(xs, ys))
    assert any(abs(x_pos - 180) < 1.0 and abs(y_pos - 150) < 1.0 for x_pos, y_pos in zip(xs, ys))

    # Test FWHM measurement
    measured_fwhm = measure_empirical_fwhm(image, sources)
    assert 2.0 <= measured_fwhm <= 4.5


def test_aperture_photometry_accuracy():
    ny, nx = 200, 200
    sky_level = 10.0
    image = np.full((ny, nx), fill_value=sky_level, dtype=np.float32)

    # Place an isolated circular top-hat source of radius 3 pixels with total flux 1000 ADU
    cx, cy = 100, 100
    y, x = np.mgrid[0:ny, 0:nx]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    source_mask = dist <= 3.0
    n_pix = np.sum(source_mask)
    image[source_mask] += 1000.0 / n_pix

    records = extract_aperture_photometry(
        image,
        positions=[(cx, cy)],
        fwhm=3.5,
        gain=4.5,
        read_noise=15.0,
        aperture_scale=1.1,
        annulus_in_scale=2.5,
        annulus_out_scale=4.0,
    )

    assert len(records) == 1
    rec = records[0]
    assert np.isclose(rec.sky_background_per_pixel, sky_level, atol=0.1)
    # Recovered net flux should match injected ~1000 ADU
    assert np.isclose(rec.net_flux, 1000.0, atol=50.0)
    assert not np.isnan(rec.instrumental_mag)
