#!/usr/bin/env python3
"""Run automated photometric extraction and diagnostic inspection on autex calibrated FITS."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from autex.io import load_fits
from autex.logging import setup_logger, get_logger, log_section
from autex.photometry import (
    estimate_background_2d,
    detect_sources,
    measure_empirical_fwhm,
    extract_aperture_photometry,
)

setup_logger(name="autex", log_file="photometry_run.log")
logger = get_logger("runner")

log_section("AUTEX AUTOMATED PHOTOMETRY EXTRACTION")

FITS_PATH = Path("data/reduced/autex_calibrated_NGC6604-92.fits")
if not FITS_PATH.is_file():
    raise FileNotFoundError(f"Calibrated slope image not found at: {FITS_PATH}")

# 1. Ingest Calibrated 2D Slope Image
logger.info("Loading calibrated slope image: %s", FITS_PATH.name)
image, header = load_fits(FITS_PATH)

# 2. Dynamic 2D Mesh Background Modeling
logger.info("Modeling 2D background and noise RMS...")
bkg_map, rms_map, global_sky = estimate_background_2d(image, box_size=(64, 64))
image_sub = image - bkg_map

# 3. Autonomous Source Detection
logger.info("Detecting candidate stellar point sources...")
sources = detect_sources(image_sub, rms_map, expected_fwhm=3.5, threshold_sigma=4.5)

# 4. Empirical Seeing (FWHM) Determination
fwhm = measure_empirical_fwhm(image_sub, sources)
logger.info("Measured frame seeing: FWHM = %.2f pixels", fwhm)

# 5. Adaptive Aperture Photometry
positions = [(s.x, s.y) for s in sources]
records = extract_aperture_photometry(
    image,
    positions=positions,
    fwhm=fwhm,
    gain=4.5,
    read_noise=15.0,
    aperture_scale=1.1,
    annulus_in_scale=2.5,
    annulus_out_scale=4.0,
)

# 6. Summary of Brightest Sources
log_section("EXTRACTED SOURCES (TOP 10)")
valid_records = [r for r in records if not np.isnan(r.instrumental_mag)]
sorted_records = sorted(valid_records, key=lambda r: r.net_flux, reverse=True)

for r in sorted_records[:10]:
    logger.info(
        "ID %3d: X=%6.1f, Y=%6.1f | Net Flux = %9.2f ADU/frame | Inst Mag = %6.3f +/- %5.3f",
        r.source_id, r.x, r.y, r.net_flux, r.instrumental_mag, r.mag_err
    )

# 7. Diagnostic Photometric Overlay Plot
fig, ax = plt.subplots(figsize=(10, 10))
vals = image.flatten()
vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
im = ax.imshow(image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count Rate (ADU / frame)")

r_aper = 1.1 * fwhm
for r in sorted_records:
    circ = Circle((r.x, r.y), r_aper, color="red", fill=False, lw=1.2, alpha=0.8)
    ax.add_patch(circ)
    ax.text(r.x + 6, r.y + 6, f"#{r.source_id}", color="white", fontsize=8, weight="bold")

ax.set_title(f"autex Photometry: {len(sorted_records)} Sources Detected (FWHM = {fwhm:.2f} pix)", fontsize=13, weight="bold")
ax.set_xlabel("Detector Column (X)")
ax.set_ylabel("Detector Row (Y)")

out_plot = Path("data/reduced/photometry_inspection.png")
plt.tight_layout()
plt.savefig(out_plot, dpi=200)
logger.info("Saved photometric diagnostic overlay to: %s", out_plot)
plt.show()

log_section("PHOTOMETRY COMPLETE")
