"""Automated Near-Infrared Spectroscopic Reduction Engine for autex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.optimize import curve_fit

from autex.logging import get_logger

logger = get_logger("spectroscopy")


@dataclass
class ExtractedSpectrum1D:
    """Extracted 1D astronomical spectrum."""
    pixels: np.ndarray
    wavelength: np.ndarray | None
    flux: np.ndarray
    flux_err: np.ndarray
    header_meta: dict[str, Any]


def subtract_nod_pair(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Subtract dithered/nodded spectral pair (A - B) to eliminate sky emission.

    Parameters
    ----------
    image_a : np.ndarray
        Calibrated 2D slope image at Slit Position A.
    image_b : np.ndarray
        Calibrated 2D slope image at Slit Position B.

    Returns
    -------
    np.ndarray
        Differential spectral frame with positive trace A and negative trace B.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"Array shape mismatch: {image_a.shape} vs {image_b.shape}")
    logger.info("Computing nod pair difference (A - B).")
    return (image_a - image_b).astype(np.float32)


def fit_spectral_trace(
    diff_image: np.ndarray,
    search_window_y: tuple[int, int] | None = None,
    poly_order: int = 2,
    bin_width: int = 32,
) -> np.ndarray:
    """Locate and trace the stellar continuum centroid along the dispersion axis.

    Parameters
    ----------
    diff_image : np.ndarray
        2D difference frame (A - B). Dispersion axis is horizontal (X), spatial is vertical (Y).
    search_window_y : tuple[int, int] | None, optional
        (ymin, ymax) pixel boundaries to isolate positive trace A, by default None.
    poly_order : int, optional
        Polynomial degree for trace curvature model, by default 2.
    bin_width : int, optional
        Column binning width along dispersion axis to increase SNR, by default 32.

    Returns
    -------
    np.ndarray
        Trace center Y-pixel coordinate for every dispersion column X. Shape: (Nx,).
    """
    ny, nx = diff_image.shape
    ymin, ymax = search_window_y if search_window_y is not None else (0, ny)
    logger.info("Tracing spectrum across spatial window [%d, %d]...", ymin, ymax)

    x_bins = []
    y_centroids = []

    # Iterate across dispersion axis in bins
    for x_start in range(0, nx, bin_width):
        x_end = min(nx, x_start + bin_width)
        x_mid = 0.5 * (x_start + x_end)

        # Spatial 1D slice across the bin
        spatial_profile = np.median(diff_image[ymin:ymax, x_start:x_end], axis=1)

        # Baseline shift to positive domain
        baseline = np.median(spatial_profile)
        profile_sub = spatial_profile - baseline
        profile_sub = np.clip(profile_sub, a_min=0, a_max=None)

        total_weight = np.sum(profile_sub)
        if total_weight > 0:
            y_indices = np.arange(ymin, ymax, dtype=np.float64)
            centroid_y = float(np.sum(profile_sub * y_indices) / total_weight)
            x_bins.append(x_mid)
            y_centroids.append(centroid_y)

    if len(x_bins) < (poly_order + 1):
        raise RuntimeError("Insufficient signal to map trace along dispersion axis.")

    # Fit smooth polynomial along dispersion axis: y_trace(x)
    coeffs = np.polyfit(x_bins, y_centroids, deg=poly_order)
    x_full = np.arange(nx, dtype=np.float64)
    y_trace = np.polyval(coeffs, x_full)

    logger.info("Trace modeled successfully. Mean centroid Y = %.2f pixels.", float(np.mean(y_trace)))
    return y_trace


def extract_optimal_spectrum(
    image: np.ndarray,
    y_trace: np.ndarray,
    variance_map: np.ndarray | None = None,
    aperture_radius: int = 6,
    gain: float = 4.5,
    read_noise: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1D spectrum using spatial profile weighting (Horne's Algorithm).

    Parameters
    ----------
    image : np.ndarray
        Calibrated 2D frame.
    y_trace : np.ndarray
        Trace spatial center coordinate Y for each column X (length Nx).
    variance_map : np.ndarray | None, optional
        Precomputed variance map. If None, derived from gain and read noise.
    aperture_radius : int, optional
        Spatial half-width for extraction profile, by default 6 pixels.
    gain : float, optional
        Detector gain (e-/ADU), by default 4.5.
    read_noise : float, optional
        Read noise (e-), by default 15.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (flux_1d, flux_err_1d) across dispersion columns.
    """
    ny, nx = image.shape
    if variance_map is None:
        variance_map = (np.abs(image) / gain) + ((read_noise / gain) ** 2)

    flux_1d = np.zeros(nx, dtype=np.float64)
    var_1d = np.zeros(nx, dtype=np.float64)

    # 1. Construct empirical spatial profile P(y|x)
    profile_box = np.zeros((2 * aperture_radius + 1, nx), dtype=np.float64)
    for x in range(nx):
        y_center = int(round(y_trace[x]))
        y_lo = max(0, y_center - aperture_radius)
        y_hi = min(ny, y_center + aperture_radius + 1)
        slice_len = y_hi - y_lo
        if slice_len > 0:
            profile_box[:slice_len, x] = image[y_lo:y_hi, x]

    # Profile normalization
    profile_mean = np.median(profile_box, axis=1)
    profile_mean = np.clip(profile_mean, a_min=0, a_max=None)
    profile_sum = np.sum(profile_mean)
    p_norm = profile_mean / profile_sum if profile_sum > 0 else np.ones_like(profile_mean) / len(profile_mean)

    # 2. Variance-weighted extraction across each column
    for x in range(nx):
        y_center = int(round(y_trace[x]))
        y_lo = max(0, y_center - aperture_radius)
        y_hi = min(ny, y_center + aperture_radius + 1)

        data_col = image[y_lo:y_hi, x]
        var_col = variance_map[y_lo:y_hi, x]
        p_col = p_norm[:len(data_col)]

        p_norm_col = p_col / np.sum(p_col) if np.sum(p_col) > 0 else p_col

        # Horne's optimal sum: sum(P * D / V) / sum(P^2 / V)
        weights = p_norm_col / var_col
        denom = np.sum(p_norm_col * weights)

        if denom > 0:
            flux_1d[x] = np.sum(weights * data_col) / denom
            var_1d[x] = 1.0 / denom
        else:
            flux_1d[x] = np.sum(data_col)
            var_1d[x] = np.sum(var_col)

    flux_err_1d = np.sqrt(np.clip(var_1d, a_min=1e-8, a_max=None))
    logger.info("Extracted optimal 1D spectrum across %d dispersion columns.", nx)
    return flux_1d, flux_err_1d


def calibrate_wavelength(
    pixel_indices: np.ndarray,
    dispersion_poly_coeffs: list[float] | np.ndarray,
) -> np.ndarray:
    """Map dispersion pixel indices to physical wavelengths in micrometers.

    Parameters
    ----------
    pixel_indices : np.ndarray
        Array of column pixel indices (e.g. 0 to 1023).
    dispersion_poly_coeffs : list[float] | np.ndarray
        Polynomial coefficients in descending degree: [c_n, ..., c_1, c_0].

    Returns
    -------
    np.ndarray
        Wavelengths in micrometers.
    """
    return np.polyval(dispersion_poly_coeffs, pixel_indices)


def correct_telluric_absorption(
    science_spectrum: np.ndarray,
    standard_spectrum: np.ndarray,
    standard_bb_temperature: float = 9500.0,
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    """Correct for telluric absorption features by dividing by a telluric standard star.

    Parameters
    ----------
    science_spectrum : np.ndarray
        Extracted 1D target spectrum.
    standard_spectrum : np.ndarray
        Extracted 1D telluric standard star spectrum (observed at similar airmass).
    standard_bb_temperature : float, optional
        Standard star effective blackbody temperature (A0V standard is ~9500 K), by default 9500.0.
    wavelengths : np.ndarray | None, optional
        Wavelength array in micrometers for continuum shaping, by default None.

    Returns
    -------
    np.ndarray
        Telluric-corrected, continuum-normalized science spectrum.
    """
    # Guard against division by near-zero counts in deep atmospheric absorption bands
    std_clean = np.where(np.abs(standard_spectrum) < 1e-4, np.nan, standard_spectrum)
    ratio = science_spectrum / std_clean

    if wavelengths is not None:
        # Planck blackbody function approximation for continuum slope restoration: B_lambda(T)
        c = 3.0e8
        h = 6.626e-34
        k = 1.38e-23
        lam_m = wavelengths * 1e-6
        planck = (2.0 * h * c**2) / (lam_m**5 * (np.exp((h * c) / (lam_m * k * standard_bb_temperature)) - 1.0))
        planck_norm = planck / np.nanmedian(planck)
        ratio *= planck_norm

    return ratio
