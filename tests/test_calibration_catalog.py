"""Automated unit tests for autex.calibration_catalog module."""

import numpy as np
from astropy.wcs import WCS
import pytest

from autex.calibration_catalog import calculate_photometric_zero_point
from autex.photometry import PhotometryRecord


def test_photometric_zero_point_solving():
    # Construct a simple synthetic tangent plane WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [512.0, 512.0]
    w.wcs.cdelt = [-0.0001, 0.0001]
    w.wcs.crval = [180.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Injected true zero-point
    true_zp = 21.5

    # Create 3 synthetic detected stars
    positions = [(500.0, 500.0), (450.0, 550.0), (600.0, 480.0)]
    phot_records = []
    catalog_stars = []

    for idx, (x, y) in enumerate(positions):
        ra, dec = w.all_pix2world(x, y, 0)
        inst_mag = -10.0 + idx * 0.5
        std_mag = inst_mag + true_zp  # Exact zero point

        rec = PhotometryRecord(
            source_id=idx + 1,
            x=x,
            y=y,
            aperture_sum=1000.0,
            sky_background_per_pixel=10.0,
            net_flux=1000.0,
            flux_err=10.0,
            instrumental_mag=inst_mag,
            mag_err=0.01,
        )
        phot_records.append(rec)
        catalog_stars.append({
            "ra": float(ra),
            "dec": float(dec),
            "Kmag": float(std_mag),
        })

    # Solve for ZP
    sol = calculate_photometric_zero_point(
        phot_records,
        catalog_stars,
        wcs=w,
        filter_band="Ks",
        match_radius_arcsec=1.5,
    )

    assert sol.num_matched_stars == 3
    assert np.isclose(sol.zero_point, true_zp, atol=1e-3)
    assert np.isclose(sol.zero_point_err, 0.0, atol=1e-3)
