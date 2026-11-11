"""Automated unit tests for autex.astrometry module."""

import numpy as np
import pytest

from autex.astrometry import build_wcs_header, compute_frame_features
from autex.photometry import SourceDetection


def test_frame_object_features():
    # Construct 4 stars symmetrically placed inside inscribed circle (radius <= 512)
    sources = [
        SourceDetection(source_id=1, x=512.0, y=600.0, flux=1000.0, peak=100.0, fwhm=3.0),
        SourceDetection(source_id=2, x=512.0, y=424.0, flux=1000.0, peak=100.0, fwhm=3.0),
        SourceDetection(source_id=3, x=600.0, y=512.0, flux=1000.0, peak=100.0, fwhm=3.0),
        SourceDetection(source_id=4, x=424.0, y=512.0, flux=1000.0, peak=100.0, fwhm=3.0),
    ]

    features = compute_frame_features(sources, detector_dim=(1024, 1024))
    assert features is not None
    assert features.star_count == 4
    assert np.isclose(features.mean_brightness, 100.0, atol=1e-2)
    # Symmetric layout centered on (512, 512) yields center of mass near 0
    assert np.isclose(features.center_of_mass_distance, 0.0, atol=1e-1)


def test_wcs_header_generation():
    ra_target = 83.822
    dec_target = -5.391
    wcs = build_wcs_header(ra_target, dec_target, pixel_scale_arcsec=0.307, rotation_deg=0.0)

    assert wcs.is_celestial
    # Center pixel (512.5, 512.5) must project back to target coordinates
    ra_proj, dec_proj = wcs.all_pix2world(512.5, 512.5, 1)
    assert np.isclose(ra_proj, ra_target, atol=1e-4)
    assert np.isclose(dec_proj, dec_target, atol=1e-4)
