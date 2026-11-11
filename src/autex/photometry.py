"""Automated, native Python aperture photometry and 2D background estimation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings
import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

from autex.logging import get_logger

logger = get_logger("photometry")


@dataclass
class SourceDetection:
    """Detected point source record."""
    source_id: int
    x: float
    y: float
    flux: float
    peak: float
    fwhm: float


@dataclass
class PhotometryRecord:
    """Photometric aperture extraction record for a detected source."""
    source_id: int
    x: float
    y: float
    aperture_sum: float
    sky_background_per_pixel: float
    net_flux: float
    flux_err: float
    instrumental_mag: float
    mag_err: float


def estimate_background_2d(
    image: np.ndarray,
    box_size: tuple[int, int] = (64, 64),
    filter_size: tuple[int, int] = (3, 3),
    sigma_clip_val: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Model the 2D spatial sky background and noise RMS map."""
    logger.debug("Estimating 2D background with mesh box %s", box_size)
    sigma_clip = SigmaClip(sigma=sigma_clip_val)
    bkg_estimator = MedianBackground()

    bkg = Background2D(
        image,
        box_size=box_size,
        filter_size=filter_size,
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )

    bkg_map = bkg.background
    rms_map = bkg.background_rms
    global_sky = float(bkg.background_median)

    logger.debug("Global median sky level: %.3f ADU/frame (RMS: %.3f)", global_sky, float(np.median(rms_map)))
    return bkg_map.astype(np.float32), rms_map.astype(np.float32), global_sky


def detect_sources(
    image_sub: np.ndarray,
    bkg_rms: np.ndarray | float,
    expected_fwhm: float = 3.5,
    threshold_sigma: float = 4.0,
    round_limits: tuple[float, float] = (-0.8, 0.8),
    sharp_limits: tuple[float, float] = (0.2, 1.0),
) -> list[SourceDetection]:
    """Autonomously detect stellar point sources with modern photutils syntax."""
    rms_scalar = float(np.median(bkg_rms)) if isinstance(bkg_rms, np.ndarray) else float(bkg_rms)
    threshold = threshold_sigma * rms_scalar

    try:
        finder = DAOStarFinder(
            threshold=threshold,
            fwhm=expected_fwhm,
            roundness_range=round_limits,
            sharpness_range=sharp_limits,
        )
    except TypeError:
        finder = DAOStarFinder(
            threshold=threshold,
            fwhm=expected_fwhm,
            roundlo=round_limits[0],
            roundhi=round_limits[1],
            sharplo=sharp_limits[0],
            sharphi=sharp_limits[1],
        )

    # Catch and silence NoDetectionsWarning cleanly
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NoDetectionsWarning)
        table = finder.find_stars(image_sub)

    if table is None or len(table) == 0:
        logger.info("No point sources detected with threshold %.3f (%.1f-sigma).", threshold, threshold_sigma)
        return []

    col_x = "x_centroid" if "x_centroid" in table.colnames else "xcentroid"
    col_y = "y_centroid" if "y_centroid" in table.colnames else "ycentroid"

    sources = []
    for row in table:
        sources.append(
            SourceDetection(
                source_id=int(row["id"]),
                x=float(row[col_x]),
                y=float(row[col_y]),
                flux=float(row["flux"]),
                peak=float(row["peak"]),
                fwhm=expected_fwhm,
            )
        )

    logger.info("Detected %d candidate stellar sources (threshold=%.2f RMS)", len(sources), threshold_sigma)
    return sources


def measure_empirical_fwhm(
    image_sub: np.ndarray,
    sources: list[SourceDetection],
    box_radius: int = 12,
) -> float:
    """Measure the median empirical seeing FWHM across bright, unsaturated stars."""
    if not sources:
        return 3.5

    sorted_sources = sorted(sources, key=lambda s: s.peak, reverse=True)
    ny, nx = image_sub.shape
    fwhm_estimates = []

    for src in sorted_sources[:15]:
        ix, iy = int(round(src.x)), int(round(src.y))
        if ix - box_radius < 0 or ix + box_radius >= nx or iy - box_radius < 0 or iy + box_radius >= ny:
            continue

        stamp = image_sub[iy - box_radius : iy + box_radius + 1, ix - box_radius : ix + box_radius + 1]
        peak_val = np.max(stamp)
        if peak_val <= 0:
            continue

        half_max = peak_val / 2.0
        above_half = stamp >= half_max
        area = np.sum(above_half)
        fwhm_est = 2.0 * np.sqrt(area / np.pi)
        if 1.0 <= fwhm_est <= 15.0:
            fwhm_estimates.append(fwhm_est)

    if not fwhm_estimates:
        return 3.5

    measured = float(np.median(fwhm_estimates))
    logger.info("Empirical seeing measured across %d stars: FWHM = %.2f pixels", len(fwhm_estimates), measured)
    return measured


def extract_aperture_photometry(
    image: np.ndarray,
    positions: list[tuple[float, float]],
    fwhm: float = 3.5,
    gain: float = 4.5,
    read_noise: float = 15.0,
    aperture_scale: float = 1.0,
    annulus_in_scale: float = 2.5,
    annulus_out_scale: float = 4.0,
) -> list[PhotometryRecord]:
    """Perform dynamic circular aperture photometry with local annulus background subtraction."""
    if not positions:
        return []

    r_aper = max(1.5, aperture_scale * fwhm)
    r_in = max(r_aper + 1.0, annulus_in_scale * fwhm)
    r_out = max(r_in + 2.0, annulus_out_scale * fwhm)

    logger.debug("Aperture geometry: r_aper=%.2f, r_in=%.2f, r_out=%.2f (pixels)", r_aper, r_in, r_out)

    apertures = CircularAperture(positions, r=r_aper)
    annuli = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    phot_table = aperture_photometry(image, apertures)
    annulus_masks = annuli.to_mask(method="center")
    records: list[PhotometryRecord] = []

    for idx, (pos, mask) in enumerate(zip(positions, annulus_masks)):
        annulus_data = mask.cutout(image)
        if annulus_data is None:
            sky_per_pixel = 0.0
        else:
            values = annulus_data[mask.data > 0]
            sigma_clip = SigmaClip(sigma=3.0)
            clipped_vals = sigma_clip(values)
            sky_per_pixel = float(np.ma.median(clipped_vals))

        aper_sum = float(phot_table["aperture_sum"][idx])
        aper_area = apertures.area
        net_flux = aper_sum - (sky_per_pixel * aper_area)

        pos_flux = max(0.0, net_flux)
        pos_sky = max(0.0, sky_per_pixel)
        variance_adu = (pos_flux / gain) + (aper_area * pos_sky / gain) + (aper_area * (read_noise / gain) ** 2)
        flux_err = float(np.sqrt(max(1e-6, variance_adu)))

        if net_flux > 0:
            inst_mag = float(-2.5 * np.log10(net_flux))
            mag_err = float(1.0857 * (flux_err / net_flux))
        else:
            inst_mag = np.nan
            mag_err = np.nan

        records.append(
            PhotometryRecord(
                source_id=idx + 1,
                x=float(pos[0]),
                y=float(pos[1]),
                aperture_sum=aper_sum,
                sky_background_per_pixel=sky_per_pixel,
                net_flux=net_flux,
                flux_err=flux_err,
                instrumental_mag=inst_mag,
                mag_err=mag_err,
            )
        )

    return records
