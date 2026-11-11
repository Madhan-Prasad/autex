#!/usr/bin/env python3
"""Run autonomous astrometry and 2MASS zero-point calibration on autex slope images."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from astropy.wcs import WCS

from autex.astrometry import build_wcs_header, compute_frame_features
from autex.calibration_catalog import calculate_photometric_zero_point, query_2mass_catalog
from autex.io import load_fits
from autex.logging import setup_logger, get_logger, log_section
from autex.photometry import (
    estimate_background_2d,
    detect_sources,
    measure_empirical_fwhm,
    extract_aperture_photometry,
)

setup_logger(name="autex", log_file="astrometry_photometry.log")
logger = get_logger("runner")

log_section("AUTEX AUTONOMOUS ASTROMETRY & 2MASS CALIBRATION")

FITS_PATH = Path("data/reduced/autex_calibrated_NGC6604-92.fits")
image, header = load_fits(FITS_PATH)

# 1. Background & Seeing
bkg_map, rms_map, _ = estimate_background_2d(image, box_size=(64, 64))
image_sub = image - bkg_map
prelim_sources = detect_sources(image_sub, rms_map, expected_fwhm=3.5, threshold_sigma=5.0)
fwhm = float(np.clip(measure_empirical_fwhm(image_sub, prelim_sources), 2.8, 4.5))

# 2. Source Extraction
sources = detect_sources(image_sub, rms_map, expected_fwhm=fwhm, threshold_sigma=4.0)
frame_features = compute_frame_features(sources)
logger.info("Focal plane stars: %d valid inside inscribed circle.", frame_features.star_count if frame_features else 0)

# 3. Aperture Photometry
positions = [(s.x, s.y) for s in sources]
raw_records = extract_aperture_photometry(image, positions=positions, fwhm=fwhm)
valid_records = [r for r in raw_records if not np.isnan(r.instrumental_mag) and r.net_flux > 0]
valid_records.sort(key=lambda r: r.net_flux, reverse=True)
for idx, r in enumerate(valid_records, start=1):
    r.source_id = idx

# 4. Autonomous Astrometric Solving & 2MASS Matching
# NGC 6604 cluster reference center: RA = 18h 18m 03s (274.5125 deg), Dec = -12d 14m 30s (-12.2417 deg)
ra_center = float(header.get("RA", 274.5125))
dec_center = float(header.get("DEC", -12.2417))
logger.info("Establishing Astrometric WCS centered at (RA=%.4f, Dec=%.4f)...", ra_center, dec_center)

wcs = build_wcs_header(ra_center, dec_center, pixel_scale_arcsec=0.307, rotation_deg=0.0)

# Query 2MASS PSC
cat_stars = query_2mass_catalog(ra_center, dec_center, radius_arcmin=6.0)
zp_solution = calculate_photometric_zero_point(
    valid_records, cat_stars, wcs=wcs, filter_band="Ks", match_radius_arcsec=2.5
)

zp = zp_solution.zero_point
zp_err = zp_solution.zero_point_err
log_section(f"ZERO-POINT RESULT: ZP = {zp:.3f} +/- {zp_err:.3f} mag ({zp_solution.num_matched_stars} stars matched)")

# 5. Write Calibrated Output Table
out_csv = Path("data/reduced/calibrated_photometry_NGC6604.csv")
with open(out_csv, "w") as fp:
    fp.write("id,x,y,net_flux,flux_err,inst_mag,mag_err,cal_mag,cal_mag_err\n")
    for r in valid_records:
        cal_mag = r.instrumental_mag + zp if not np.isnan(zp) else np.nan
        cal_err = np.sqrt(r.mag_err**2 + zp_err**2) if not np.isnan(zp) else np.nan
        fp.write(f"{r.source_id},{r.x:.2f},{r.y:.2f},{r.net_flux:.2f},{r.flux_err:.2f},{r.instrumental_mag:.4f},{r.mag_err:.4f},{cal_mag:.4f},{cal_err:.4f}\n")

logger.info("Wrote calibrated stellar magnitudes -> %s", out_csv.name)
log_section("PIPELINE CALIBRATION COMPLETED")
