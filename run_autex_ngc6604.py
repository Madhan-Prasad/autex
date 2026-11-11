#!/usr/bin/env python3
"""Run autex on real NGC6604 sequential NDR observation slices in Dark-31."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from astropy.io import fits

import autex
from autex.calibration import fit_sutr_slope, remove_pedestal
from autex.io import stack_ndr_sequence
from autex.logging import setup_logger, get_logger, log_section

setup_logger(name="autex", log_file="ngc6604_run.log")
logger = get_logger("runner")

log_section(f"AUTEX REAL DATA: NGC6604-92 REDUCTION (v{autex.__version__})")

DATA_BASE = Path("/Users/maddy/Documents/NIUS/Data")
PCA_DIR = DATA_BASE / "tirspec_curves/pca_fits"
AVGCUR_PATH = DATA_BASE / "tirspec_curves/avgcur.npy"
TARGET_DIR = DATA_BASE / "Dark-31"
OUTPUT_DIR = Path("/Users/maddy/Documents/NIUS/AUTEX/data/reduced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Pre-trained Calibration Priors
logger.info("Loading 10-year archival calibration priors...")
scaler = joblib.load(PCA_DIR / "scalar_10year_fit.pkl")
pca = joblib.load(PCA_DIR / "pca_10year_fit.pkl")
avgcur = np.load(AVGCUR_PATH)

# 2. Gather All Sequential 2D NDR Slices for Sequence 92
logger.info("Scanning directory: %s", TARGET_DIR)
ngc_slices = sorted(list(TARGET_DIR.glob("*NGC6604-92-debug*.fits*")))
if not ngc_slices:
    ngc_slices = sorted(list(TARGET_DIR.glob("*NGC6604*.fits*")))

logger.info("Found %d NDR slice files in %s", len(ngc_slices), TARGET_DIR.name)
if not ngc_slices:
    raise FileNotFoundError(f"No NGC6604 slice files found in {TARGET_DIR}")

# 3. Stack Slices into 3D SUTR Ramp Cube: (N_reads, 1024, 1024)
raw_cube, base_header = stack_ndr_sequence(ngc_slices)
n_reads, ny, nx = raw_cube.shape
logger.info("Assembled raw 3D data cube: %s (N_reads=%d)", raw_cube.shape, n_reads)

# 4. Construct Matching Clock-Interpolated Bias Model
dark_decay_slice = avgcur[:n_reads]
t_original = np.arange(n_reads, dtype=np.float64) + 0.5
cs = CubicSpline(t_original, dark_decay_slice, extrapolate=True)

# Generate row-delay grid: t_r = frame + (512 - r) / 512
t_frames = np.arange(n_reads, dtype=np.float64)[:, np.newaxis]
row_delays = ((512.0 - np.arange(512, dtype=np.float64)) / 512.0)[np.newaxis, :]
t_grid = t_frames + row_delays
rows_interp = cs(t_grid)  # Shape: (n_reads, 512)

# Tile 4 quadrants into full (n_reads, 1024, 1024)
quadrant = np.repeat(rows_interp[:, :, np.newaxis], 512, axis=2)
halfframe = np.concatenate([quadrant, quadrant], axis=2)
full_bias_cube = np.concatenate([halfframe, halfframe], axis=1).astype(np.float32)
logger.info("Reconstructed 3D clock-interpolated bias model: %s", full_bias_cube.shape)

# 5. Reset Pedestal Subtraction & Calibration
raw_ped_sub = remove_pedestal(raw_cube)
calibrated_cube = raw_ped_sub - full_bias_cube

# 6. Fit SUTR Slopes Along Ramp
time_axis = np.arange(n_reads, dtype=np.float64)
fit_window = slice(3, n_reads - 1) if n_reads >= 8 else slice(0, n_reads)

logger.info("Fitting linear SUTR slopes across frames %d to %d...", fit_window.start, fit_window.stop)
slope_raw, _ = fit_sutr_slope(raw_ped_sub, fit_slice=fit_window, time_array=time_axis)
slope_calibrated, _ = fit_sutr_slope(calibrated_cube, fit_slice=fit_window, time_array=time_axis)

# 7. Save Calibrated Output FITS
out_fits_path = OUTPUT_DIR / "autex_calibrated_NGC6604-92.fits"
out_hdr = base_header.copy()
out_hdr["BUNIT"] = ("ADU/frame", "Photon count rate per readout frame")
out_hdr["PIPELINE"] = ("autex", "Automated TIFR Extractor")
out_hdr["NREADS"] = n_reads
fits.writeto(out_fits_path, slope_calibrated, header=out_hdr, overwrite=True)
logger.info("Saved scientific slope map to: %s", out_fits_path)

# 8. Render Seamless Side-by-Side Diagnostic Inspection Plot (No Quadrant Lines)
raw_flat = slope_raw.flatten()
cal_flat = slope_calibrated.flatten()

vmin = min(np.percentile(raw_flat, 5), np.percentile(cal_flat, 5))
vmax = max(np.percentile(raw_flat, 95), np.percentile(cal_flat, 95))

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

im0 = axes[0].imshow(slope_raw, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title(r"Raw Slope Image ($\mathbf{Slope_{raw}}$)", fontsize=13, weight="bold")
axes[0].set_xlabel("Detector Column (X)", fontsize=11)
axes[0].set_ylabel("Detector Row (Y)", fontsize=11)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Count Rate (ADU / frame)")

im1 = axes[1].imshow(slope_calibrated, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
axes[1].set_title(r"autex Calibrated Slope ($\mathbf{Slope_{darksub}}$)", fontsize=13, weight="bold")
axes[1].set_xlabel("Detector Column (X)", fontsize=11)
axes[1].set_ylabel("Detector Row (Y)", fontsize=11)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Count Rate (ADU / frame)")

plt.suptitle(f"autex Reduction: NGC6604-92 in Dark-31 ({n_reads} SUTR Frames Stacked)", fontsize=14, weight="bold")
plt.tight_layout()

plot_out = OUTPUT_DIR / "NGC6604-92_slope_diagnostic.png"
plt.savefig(plot_out, dpi=200)
logger.info("Saved seamless diagnostic comparison plot to: %s", plot_out)
plt.show()

log_section("NGC6604 REDUCTION COMPLETED")
