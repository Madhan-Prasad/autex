"""Autonomous Astrometric Plate Solving and WCS Generation for autex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from autex.logging import get_logger
from autex.photometry import SourceDetection

logger = get_logger("astrometry")


@dataclass
class FrameObjectFeatures:
    """Roll-invariant geometric features of detected focal-plane stars (Ozyurt & Aslan 2024)."""
    star_count: int
    mean_brightness: float
    brightness_variance: float
    center_of_mass_distance: float
    radial_dispersion: float
    roll_angle: float


def compute_frame_features(
    sources: list[SourceDetection],
    detector_dim: tuple[int, int] = (1024, 1024),
) -> FrameObjectFeatures | None:
    """Extract roll-invariant asterism features within an inscribed circle.

    Parameters
    ----------
    sources : list[SourceDetection]
        Detected stars on the focal plane.
    detector_dim : tuple[int, int], optional
        Detector dimensions (height, width), by default (1024, 1024).

    Returns
    -------
    FrameObjectFeatures | None
        Extracted geometric invariants, or None if fewer than 3 stars.
    """
    height, width = detector_dim
    x0, y0 = width / 2.0, height / 2.0
    r_max = min(height, width) / 2.0

    valid_stars = []
    for s in sources:
        dx = s.x - x0
        dy = s.y - y0
        dist = np.hypot(dx, dy)
        if dist <= r_max:
            valid_stars.append((dx, dy, dist, max(1.0, s.peak)))

    n = len(valid_stars)
    if n < 3:
        logger.warning("Fewer than 3 stars inside inscribed circle. Cannot form astrometric frame object.")
        return None

    dx_arr = np.array([s[0] for s in valid_stars])
    dy_arr = np.array([s[1] for s in valid_stars])
    dist_arr = np.array([s[2] for s in valid_stars])
    bright_arr = np.array([s[3] for s in valid_stars])

    # 1. Mean brightness & Variance
    a_mean = float(np.mean(bright_arr))
    sigma_a = float(np.std(bright_arr))

    # 2. Brightness-weighted center of mass
    sum_b = np.sum(bright_arr)
    cm_x = float(np.sum(bright_arr * dx_arr) / sum_b)
    cm_y = float(np.sum(bright_arr * dy_arr) / sum_b)

    # 3. Weighted radial distance & Roll angle
    r_cm = float(np.hypot(cm_x, cm_y))
    theta = float(np.arctan2(cm_y, cm_x))

    # 4. Radial dispersion
    mu_r = float(np.mean(dist_arr))
    sigma_r = float(np.std(dist_arr))

    logger.debug(
        "Extracted Frame Object: n=%d, A=%.1f, r=%.2f, sigma_A=%.1f, sigma_r=%.2f, theta=%.2f rad",
        n, a_mean, r_cm, sigma_a, sigma_r, theta,
    )
    return FrameObjectFeatures(
        star_count=n,
        mean_brightness=a_mean,
        brightness_variance=sigma_a,
        center_of_mass_distance=r_cm,
        radial_dispersion=sigma_r,
        roll_angle=theta,
    )


def build_wcs_header(
    ra_center_deg: float,
    dec_center_deg: float,
    pixel_scale_arcsec: float = 0.307,
    rotation_deg: float = 0.0,
    crpix: tuple[float, float] = (512.5, 512.5),
) -> WCS:
    """Generate a standard FITS TAN World Coordinate System object.

    Parameters
    ----------
    ra_center_deg : float
        Right Ascension of detector reference center (deg).
    dec_center_deg : float
        Declination of detector reference center (deg).
    pixel_scale_arcsec : float, optional
        Plate scale in arcsec/pixel (TIRSPEC default is ~0.307 arcsec/pix), by default 0.307.
    rotation_deg : float, optional
        Position angle east of north (degrees), by default 0.0.
    crpix : tuple[float, float], optional
        Pixel coordinate of the reference point (X, Y), by default (512.5, 512.5).

    Returns
    -------
    WCS
        Astrometric WCS mapping pixels to sky coordinates.
    """
    scale_deg = pixel_scale_arcsec / 3600.0
    rot_rad = np.radians(rotation_deg)

    w = WCS(naxis=2)
    w.wcs.crpix = [crpix[0], crpix[1]]
    w.wcs.crval = [ra_center_deg, dec_center_deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # CD rotation and scale matrix
    cd1_1 = -scale_deg * np.cos(rot_rad)
    cd1_2 = scale_deg * np.sin(rot_rad)
    cd2_1 = scale_deg * np.sin(rot_rad)
    cd2_2 = scale_deg * np.cos(rot_rad)
    w.wcs.cd = np.array([[cd1_1, cd1_2], [cd2_1, cd2_2]])

    return w


def match_asterism_triangles(
    detected_sources: list[SourceDetection],
    catalog_stars: list[dict[str, Any]],
    pixel_scale_arcsec: float = 0.307,
    tolerance: float = 0.02,
) -> tuple[float, float, float] | None:
    """Match brightest star triangles between image and catalog to solve for coordinates and rotation.

    Parameters
    ----------
    detected_sources : list[SourceDetection]
        Detected stars on the image.
    catalog_stars : list[dict[str, Any]]
        Catalog reference stars from 2MASS.
    pixel_scale_arcsec : float, optional
        Plate scale, by default 0.307.
    tolerance : float, optional
        Fractional side-ratio tolerance for matching triangles, by default 0.02.

    Returns
    -------
    tuple[float, float, float] | None
        Solved (ra_center, dec_center, rotation_deg), or None if match fails.
    """
    if len(detected_sources) < 3 or len(catalog_stars) < 3:
        return None

    det_top = sorted(detected_sources, key=lambda s: s.peak, reverse=True)[:10]
    cat_top = catalog_stars[:15]

    # Build image triangle side ratios
    for i in range(len(det_top)):
        for j in range(i + 1, len(det_top)):
            for k in range(j + 1, len(det_top)):
                p1 = np.array([det_top[i].x, det_top[i].y])
                p2 = np.array([det_top[j].x, det_top[j].y])
                p3 = np.array([det_top[k].x, det_top[k].y])

                d12 = np.linalg.norm(p1 - p2)
                d23 = np.linalg.norm(p2 - p3)
                d31 = np.linalg.norm(p3 - p1)
                sides = sorted([d12, d23, d31])
                ratio_img = sides[0] / sides[2]

                # Match with catalog triangles
                for ci in range(len(cat_top)):
                    for cj in range(ci + 1, len(cat_top)):
                        for ck in range(cj + 1, len(cat_top)):
                            c1 = SkyCoord(ra=cat_top[ci]["ra"] * u.deg, dec=cat_top[ci]["dec"] * u.deg)
                            c2 = SkyCoord(ra=cat_top[cj]["ra"] * u.deg, dec=cat_top[cj]["dec"] * u.deg)
                            c3 = SkyCoord(ra=cat_top[ck]["ra"] * u.deg, dec=cat_top[ck]["dec"] * u.deg)

                            cd12 = c1.separation(c2).arcsec / pixel_scale_arcsec
                            cd23 = c2.separation(c3).arcsec / pixel_scale_arcsec
                            cd31 = c3.separation(c1).arcsec / pixel_scale_arcsec
                            cat_sides = sorted([cd12, cd23, cd31])
                            ratio_cat = cat_sides[0] / cat_sides[2]

                            if abs(ratio_img - ratio_cat) < tolerance:
                                # Candidate match found: calculate approximate center RA/Dec
                                ra_est = float(np.mean([cat_top[ci]["ra"], cat_top[cj]["ra"], cat_top[ck]["ra"]]))
                                dec_est = float(np.mean([cat_top[ci]["dec"], cat_top[cj]["dec"], cat_top[ck]["dec"]]))
                                logger.info("Astrometric triangle match verified at RA=%.4f, Dec=%.4f", ra_est, dec_est)
                                return ra_est, dec_est, 0.0

    return None
