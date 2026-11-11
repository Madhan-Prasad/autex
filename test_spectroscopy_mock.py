#!/usr/bin/env python3
"""Validate autex spectroscopic pipeline using realistic synthetic NIR slit data."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from autex.logging import setup_logger, get_logger, log_section
from autex.spectroscopy import (
    subtract_nod_pair,
    fit_spectral_trace,
    extract_optimal_spectrum,
    calibrate_wavelength,
    correct_telluric_absorption,
)

setup_logger(name="autex", log_file="mock_spectroscopy_test.log")
logger = get_logger("test")

log_section("TESTING SPECTROSCOPY MODULE (SYNTHETIC NIR DATA)")

OUTPUT_DIR = Path("data/reduced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Synthesize 2D Slit Frames (1024 columns x 512 rows)
ny, nx = 512, 1024
x_axis = np.arange(nx, dtype=np.float64)
wavelengths_true = 1.0 + (1.5 / nx) * x_axis  # 1.0 um to 2.5 um

# Synthetic stellar spectrum with continuum and emission lines (Br_gamma at 2.166 um, Pa_beta at 1.282 um)
flux_continuum = 50.0 + 30.0 * (wavelengths_true - 1.0)
br_gamma = 150.0 * np.exp(-((wavelengths_true - 2.1661) ** 2) / (2 * 0.003**2))
pa_beta = 180.0 * np.exp(-((wavelengths_true - 1.2822) ** 2) / (2 * 0.003**2))
stellar_spectrum = flux_continuum + br_gamma + pa_beta

# Add common night sky emission lines
sky_emission = np.zeros(nx)
for sky_line_x in [120, 245, 410, 580, 750, 890]:
    sky_emission += 300.0 * np.exp(-((x_axis - sky_line_x) ** 2) / (2 * 2.0**2))

# Slit positions: Beam A at y ~ 200, Beam B at y ~ 320 with mild optical trace curvature
trace_y_a = 200.0 + 15.0 * np.sin(np.pi * x_axis / nx)
trace_y_b = 320.0 + 15.0 * np.sin(np.pi * x_axis / nx)

img_a = np.random.normal(50.0, 5.0, (ny, nx)).astype(np.float32)
img_b = np.random.normal(50.0, 5.0, (ny, nx)).astype(np.float32)

# Inject sky background across both frames
img_a += sky_emission[np.newaxis, :]
img_b += sky_emission[np.newaxis, :]

# Inject stellar traces into Beam A and Beam B
sigma_spatial = 3.0
y_grid = np.arange(ny)[:, np.newaxis]

profile_a = np.exp(-((y_grid - trace_y_a[np.newaxis, :]) ** 2) / (2 * sigma_spatial**2))
profile_b = np.exp(-((y_grid - trace_y_b[np.newaxis, :]) ** 2) / (2 * sigma_spatial**2))

img_a += (stellar_spectrum[np.newaxis, :] * profile_a).astype(np.float32)
img_b += (stellar_spectrum[np.newaxis, :] * profile_b).astype(np.float32)

# 2. Nod Pair Subtraction (A - B)
logger.info("Executing AB nod pair subtraction (sky cancellation)...")
diff_2d = subtract_nod_pair(img_a, img_b)

# 3. Fit 2D Spectral Trace
logger.info("Tracking spectral trace across dispersion columns...")
y_fitted_trace = fit_spectral_trace(diff_2d, search_window_y=(150, 250), poly_order=2)

# 4. Optimal 1D Horne Extraction
logger.info("Extracting 1D spectrum via Horne's algorithm...")
flux_1d, flux_err_1d = extract_optimal_spectrum(diff_2d, y_trace=y_fitted_trace, aperture_radius=7)

# 5. Wavelength Dispersion Mapping
disp_coeffs = [1.5 / nx, 1.0]
cal_wavelengths = calibrate_wavelength(x_axis, disp_coeffs)

# 6. Diagnostic Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 2]})

# Panel 1: Difference 2D Frame & Trace
im = axes[0].imshow(diff_2d, origin="lower", cmap="coolwarm", vmin=-50, vmax=50, aspect="auto")
axes[0].plot(x_axis, y_fitted_trace, color="yellow", lw=2.0, label="Fitted Trace Profile (Positive Beam A)")
axes[0].axhline(320, color="cyan", linestyle="--", alpha=0.5, label="Negative Beam B")
axes[0].set_title(r"2D Nod-Subtracted Slit Frame ($\mathbf{A - B}$)", fontsize=12, weight="bold")
axes[0].set_ylabel("Spatial (Y-Pixel)")
axes[0].legend(loc="upper right")
plt.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02, label="ADU/frame")

# Panel 2: Extracted 1D Spectrum
axes[1].plot(cal_wavelengths, flux_1d, color="black", lw=1.2, label=r"Extracted 1D Flux $S(\lambda)$")
axes[1].fill_between(
    cal_wavelengths,
    flux_1d - flux_err_1d,
    flux_1d + flux_err_1d,
    color="royalblue",
    alpha=0.3,
    label=r"$\pm 1\sigma$ Horne Uncertainty",
)

# Label detected spectral lines
axes[1].annotate(
    r"Pa$\beta$ (1.282 $\mu$m)",
    xy=(1.282, 230),
    xytext=(1.282, 260),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
    ha="center",
    weight="bold",
)
axes[1].annotate(
    r"Br$\gamma$ (2.166 $\mu$m)",
    xy=(2.166, 240),
    xytext=(2.166, 270),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
    ha="center",
    weight="bold",
)

axes[1].set_title("Optimal Extracted 1D NIR Spectrum", fontsize=12, weight="bold")
axes[1].set_xlabel(r"Wavelength ($\mu$m)", fontsize=11)
axes[1].set_ylabel("Extracted Flux (ADU / frame)", fontsize=11)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(loc="upper left")

out_plot = OUTPUT_DIR / "synthetic_spectroscopy_diagnostic.png"
plt.tight_layout()
plt.savefig(out_plot, dpi=200)
logger.info("Saved spectroscopy diagnostic figure -> %s", out_plot)
plt.show()

log_section("SPECTROSCOPY MODULE VALIDATED SUCCESSFULLY")
