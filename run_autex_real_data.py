#!/usr/bin/env python3
"""Run autex on real telescope observations with 10-year PCA bias reconstruction."""

from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from astropy.io import fits

import autex
from autex.calibration import fit_sutr_slope, remove_pedestal
from autex.config import Config
from autex.io import load_fits, validate_sutr_cube
from autex.logging import get_logger, log_section, setup_logger

# -----------------------------------------------------------------------------
# 1. Pipeline Setup & Path Configuration
# -----------------------------------------------------------------------------
setup_logger(name="autex", log_file="real_data_run.log")
logger = get_logger("runner")

log_section(f"AUTEX REAL DATA REDUCTION (v{autex.__version__})")

DATA_BASE = Path("/Users/maddy/Documents/NIUS/Data")
PCA_DIR = DATA_BASE / "tirspec_curves/pca_fits"
AVGCUR_PATH = DATA_BASE / "tirspec_curves/avgcur.npy"
DARK_DIR = DATA_BASE / "tirspec_curves/20161124"
FITS_DIR = Path("/Users/maddy/Documents/NIUS/Data/Dark-376")
OUTPUT_DIR = Path("/Users/maddy/Documents/NIUS/AUTEX/data/reduced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Ingest Pre-Trained 10-Year PCA Models
# -----------------------------------------------------------------------------
logger.info("Loading 10-year archival calibration priors...")
scaler = joblib.load(PCA_DIR / "scalar_10year_fit.pkl")
pca = joblib.load(PCA_DIR / "pca_10year_fit.pkl")
avgcur = np.load(AVGCUR_PATH)
logger.info("Loaded Scaler, PCA (10-components), and Master Average Curve.")

# -----------------------------------------------------------------------------
# 3. Construct Night Master Bias Model from Real Dark Curves
# -----------------------------------------------------------------------------
dark_files = sorted(list(DARK_DIR.glob("**/curves_Dark*.npy")))
if not dark_files:
    dark_files = sorted(list(DARK_DIR.glob("curves_Dark*.npy")))

logger.info("Discovered %d calibration dark curve file(s) in %s", len(dark_files), DARK_DIR.name)

night_dark_curves = []
for f in dark_files:
    curves = np.load(f)
    n_frames, _ = curves.shape
    if n_frames != 112:
        continue
    # Row baseline normalization (subtract temporal median per row)
    curves_norm = curves - np.median(curves, axis=0)
    # Collapse rows to 1D characteristic curve
    frame_avg = np.median(curves_norm, axis=1)
    night_dark_curves.append(frame_avg)

if not night_dark_curves:
    raise RuntimeError("No valid 112-frame dark curves found to build the bias model!")

master_night_curve = np.median(np.array(night_dark_curves), axis=0)

# Project onto 10-year PCA subspace
curve_residual = master_night_curve - avgcur
X_scaled = scaler.transform(curve_residual.reshape(1, -1))
X_pca = pca.transform(X_scaled)
X_reconstructed_scaled = pca.inverse_transform(X_pca)
curve_pca_fitted = scaler.inverse_transform(X_reconstructed_scaled).flatten()

# Restore full physical decay amplitude
final_curve = curve_pca_fitted + avgcur
logger.info("Constructed PCA-denoised master dark curve across 112 frames.")

# -----------------------------------------------------------------------------
# 4. Sequential Row Clock Delay Interpolation (CubicSpline)
# -----------------------------------------------------------------------------
t_original = np.arange(112, dtype=np.float64) + 0.5
cs = CubicSpline(t_original, final_curve, extrapolate=True)

# Generate 2D time grid accounting for row delay: t_r = frame + (512 - r) / 512
t_frames = np.arange(112, dtype=np.float64)[:, np.newaxis]
row_delays = ((512.0 - np.arange(512, dtype=np.float64)) / 512.0)[np.newaxis, :]
t_grid = t_frames + row_delays

rows_interp = cs(t_grid)  # Shape: (112, 512)

# Tile 4 quadrants into full (112, 1024, 1024) 3D bias array
quadrant = np.repeat(rows_interp[:, :, np.newaxis], 512, axis=2)
halfframe = np.concatenate([quadrant, quadrant], axis=2)
full_bias_cube = np.concatenate([halfframe, halfframe], axis=1).astype(np.float32)
logger.info("Assembled full 3D detector bias model: %s", full_bias_cube.shape)

# -----------------------------------------------------------------------------
# 5. Process Real Science FITS Data
# -----------------------------------------------------------------------------
raw_fits = sorted(list(FITS_DIR.glob("*.fits*")))
logger.info("Discovered %d raw FITS exposures in %s", len(raw_fits), FITS_DIR)

if not raw_fits:
    logger.warning("No FITS files found in %s! Halting before slope fitting.", FITS_DIR)
else:
    target_fits = raw_fits[0]  # Take first exposure as test target
    logger.info("Reducing target science cube: %s", target_fits.name)

    raw_cube, header = load_fits(target_fits)
    raw_cube = validate_sutr_cube(raw_cube, expected_frames=raw_cube.shape[0])
    
    # 1. Subtract reset pedestal (D_i - D_0)
    cube_ped_sub = remove_pedestal(raw_cube)

    # 2. Subtract clock-interpolated PCA bias model
    n_reads = min(cube_ped_sub.shape[0], full_bias_cube.shape[0])
    cube_calibrated = cube_ped_sub[:n_reads] - full_bias_cube[:n_reads]

    # 3. Fit linear SUTR slopes
    time_axis = np.arange(n_reads, dtype=np.float64)
    slope_raw, _ = fit_sutr_slope(cube_ped_sub[:n_reads], time_array=time_axis)
    slope_calibrated, _ = fit_sutr_slope(cube_calibrated, time_array=time_axis)

    # Save calibrated output FITS
    out_path = OUTPUT_DIR / f"autex_calibrated_{target_fits.stem}.fits"
    hdr_out = header.copy()
    hdr_out["PIPELINE"] = ("autex", "Automated TIFR Extractor")
    hdr_out["BUNIT"] = ("ADU/frame", "Photon count rate")
    fits.writeto(out_path, slope_calibrated, header=hdr_out, overwrite=True)
    logger.info("Saved scientific slope map -> %s", out_path)

    # -------------------------------------------------------------------------
    # 6. Side-by-Side Visual Diagnostic Inspection
    # -------------------------------------------------------------------------
    raw_vals = slope_raw.compressed() if hasattr(slope_raw, "compressed") else slope_raw.flatten()
    cal_vals = slope_calibrated.compressed() if hasattr(slope_calibrated, "compressed") else slope_calibrated.flatten()

    vmin = min(np.percentile(raw_vals, 5), np.percentile(cal_vals, 5))
    vmax = max(np.percentile(raw_vals, 95), np.percentile(cal_vals, 95))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(slope_raw, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(r"Raw Slope ($\mathbf{Slope_{raw}}$)", fontsize=12, weight="bold")
    axes[0].set_xlabel("Detector Column (X)")
    axes[0].set_ylabel("Detector Row (Y)")
    axes[0].axvline(512, color="red", linestyle="--", alpha=0.5)
    axes[0].axhline(512, color="red", linestyle="--", alpha=0.5)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Count Rate (ADU / frame)")

    im1 = axes[1].imshow(slope_calibrated, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(r"autex Calibrated Slope ($\mathbf{Slope_{darksub}}$)", fontsize=12, weight="bold")
    axes[1].set_xlabel("Detector Column (X)")
    axes[1].axvline(512, color="red", linestyle="--", alpha=0.5)
    axes[1].axhline(512, color="red", linestyle="--", alpha=0.5)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Count Rate (ADU / frame)")

    plt.suptitle("autex Pipeline: Real Telescope Data Reduction", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slope_comparison_diagnostic.png", dpi=200)
    logger.info("Diagnostic comparison plot saved to %s", OUTPUT_DIR / "slope_comparison_diagnostic.png")
    plt.show()

log_section("RUN COMPLETED SUCCESSFULLY")