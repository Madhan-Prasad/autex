"""End-to-end reduction pipeline orchestrator for autex."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from autex.calibration import (
    assemble_quadrants,
    fit_sutr_slope,
    interpolate_row_clock_delays,
    remove_pedestal,
    split_quadrants,
)
from autex.calibration_catalog import (
    calculate_photometric_zero_point,
    query_2mass_catalog,
)
from autex.catalog import ObservationCatalog
from autex.config import Config
from autex.io import load_fits, validate_sutr_cube
from autex.logging import get_logger, log_section
from autex.photometry import (
    detect_sources,
    estimate_background_2d,
    extract_aperture_photometry,
    measure_empirical_fwhm,
)
from autex.spectroscopy import (
    calibrate_wavelength,
    extract_optimal_spectrum,
    fit_spectral_trace,
    subtract_nod_pair,
)

logger = get_logger("pipeline")


class ReductionPipeline:
    """Automated orchestrator executing dark modeling, slope extraction, photometry, and spectroscopy."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.catalog = ObservationCatalog()
        self.output_dir = Path(self.config.get("paths.output_dir", "data/reduced")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_master_dark_model(self, dark_records: list[dict[str, Any]]) -> np.ndarray | None:
        """Construct a 3D full-frame bias model from grouped dark exposures."""
        if not dark_records:
            logger.warning("No calibration darks found. Skipping bias decay correction.")
            return None

        logger.info("Constructing master dark bias model from %d dark sets.", len(dark_records))
        quad_curves = []

        for rec in dark_records:
            data, _ = load_fits(rec["filepath"])
            data = validate_sutr_cube(data)
            if data.ndim == 2:
                continue

            cube_p = remove_pedestal(data)
            q1, q2, q3, q4 = split_quadrants(cube_p)
            q_median = np.median(np.stack([q1, q2, q3, q4], axis=0), axis=0)
            row_curves = np.median(q_median, axis=2)
            row_curves_norm = row_curves - np.median(row_curves, axis=0)
            dark_1d = np.median(row_curves_norm, axis=1)
            quad_curves.append(dark_1d)

        if not quad_curves:
            return None

        master_decay_1d = np.median(np.array(quad_curves), axis=0)
        interp_rows = interpolate_row_clock_delays(master_decay_1d, num_rows=512)
        quadrant_model = np.repeat(interp_rows[:, :, np.newaxis], 512, axis=2)

        full_frame_model = assemble_quadrants(
            quadrant_model, quadrant_model, quadrant_model, quadrant_model
        )
        return full_frame_model.astype(np.float32)

    def reduce_science_group(
        self,
        group_name: str,
        group_data: dict[str, Any],
        dark_model: np.ndarray | None,
    ) -> list[Path]:
        """Calibrate and fit 2D SUTR slopes for all science frames in an observation group."""
        log_section(f"PROCESSING GROUP: {group_name}")
        saved_paths: list[Path] = []

        for item in group_data["science_frames"]:
            raw_path = Path(item["filepath"])
            logger.info("Processing frame: %s", raw_path.name)

            data, header = load_fits(raw_path)
            data = validate_sutr_cube(data)

            if data.ndim == 2:
                slope_map = data
            else:
                cube_p = remove_pedestal(data)
                if dark_model is not None:
                    n_frames = min(cube_p.shape[0], dark_model.shape[0])
                    cube_calibrated = cube_p[:n_frames] - dark_model[:n_frames]
                else:
                    cube_calibrated = cube_p

                slope_map, _ = fit_sutr_slope(cube_calibrated)

            out_header = header.copy()
            out_header["BUNIT"] = ("ADU/frame", "Count rate per readout frame")
            out_header["PIPELINE"] = ("autex", "Automated TIFR Extractor")
            out_header["GROUP_ID"] = group_name

            out_filename = self.output_dir / f"slope_{raw_path.stem}.fits"
            hdu = fits.PrimaryHDU(data=slope_map, header=out_header)
            hdu.writeto(out_filename, overwrite=True)
            logger.info("Wrote calibrated 2D slope image -> %s", out_filename.name)
            saved_paths.append(out_filename)

        return saved_paths

    def run_photometry_on_image(self, image_path: Path) -> Path:
        """Run background modeling, source detection, aperture photometry, and 2MASS zero-point calibration."""
        log_section(f"PHOTOMETRY: {image_path.name}")
        image, header = load_fits(image_path)
        if image.ndim != 2:
            raise ValueError(f"Photometry requires a 2D image. Given array with shape {image.shape}")

        cat_file = self.output_dir / f"phot_{image_path.stem}.csv"

        bkg_map, rms_map, _ = estimate_background_2d(image, box_size=(64, 64))
        image_sub = image - bkg_map

        prelim = detect_sources(image_sub, rms_map, expected_fwhm=3.5, threshold_sigma=5.0)
        raw_fwhm = measure_empirical_fwhm(image_sub, prelim)
        fwhm = float(np.clip(raw_fwhm, 2.8, 4.5))

        sources = detect_sources(
            image_sub,
            rms_map,
            expected_fwhm=fwhm,
            threshold_sigma=4.0,
            round_limits=(-0.8, 0.8),
            sharp_limits=(0.2, 1.0),
        )

        if not sources:
            logger.warning("Zero point sources detected in %s. Writing empty photometry table.", image_path.name)
            with open(cat_file, "w") as fp:
                fp.write("id,x,y,net_flux,flux_err,inst_mag,mag_err\n")
            return cat_file

        positions = [(s.x, s.y) for s in sources]
        records = extract_aperture_photometry(image, positions=positions, fwhm=fwhm)
        valid = [r for r in records if not np.isnan(r.instrumental_mag) and r.net_flux > 0]
        valid.sort(key=lambda r: r.net_flux, reverse=True)

        for idx, r in enumerate(valid, start=1):
            r.source_id = idx

        logger.info("Photometry extracted for %d valid sources (FWHM=%.2f px)", len(valid), fwhm)

        filter_band = str(header.get("FILTER", "Ks")).strip()
        zp_val = np.nan
        zp_err = np.nan

        try:
            wcs = WCS(header)
            if wcs.is_celestial:
                ra_center, dec_center = wcs.all_pix2world(512.0, 512.0, 0)
                catalog_stars = query_2mass_catalog(
                    float(ra_center), float(dec_center), radius_arcmin=6.0
                )
                zp_solution = calculate_photometric_zero_point(
                    valid, catalog_stars, wcs=wcs, filter_band=filter_band
                )
                zp_val = zp_solution.zero_point
                zp_err = zp_solution.zero_point_err
        except Exception as err:
            logger.debug("Astrometric WCS parsing skipped: %s", err)

        with open(cat_file, "w") as fp:
            if not np.isnan(zp_val):
                fp.write("id,x,y,net_flux,flux_err,inst_mag,mag_err,cal_mag,cal_mag_err\n")
                for r in valid:
                    cal_mag = r.instrumental_mag + zp_val
                    cal_err = np.sqrt(r.mag_err**2 + zp_err**2)
                    fp.write(f"{r.source_id},{r.x:.2f},{r.y:.2f},{r.net_flux:.2f},{r.flux_err:.2f},{r.instrumental_mag:.4f},{r.mag_err:.4f},{cal_mag:.4f},{cal_err:.4f}\n")
            else:
                fp.write("id,x,y,net_flux,flux_err,inst_mag,mag_err\n")
                for r in valid:
                    fp.write(f"{r.source_id},{r.x:.2f},{r.y:.2f},{r.net_flux:.2f},{r.flux_err:.2f},{r.instrumental_mag:.4f},{r.mag_err:.4f}\n")

        logger.info("Saved source photometry table -> %s", cat_file.name)

        fig, ax = plt.subplots(figsize=(10, 10))
        vals = image.flatten()
        vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
        im = ax.imshow(image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count Rate (ADU / frame)")

        for r in valid:
            circ = Circle((r.x, r.y), 1.1 * fwhm, color="red", fill=False, lw=1.2, alpha=0.85)
            ax.add_patch(circ)
            ax.text(r.x + 5, r.y + 5, f"#{r.source_id}", color="white", fontsize=7.5, weight="bold")

        title_str = f"autex Photometry: {len(valid)} Sources Detected (FWHM = {fwhm:.2f} px)"
        if not np.isnan(zp_val):
            title_str += f" | ZP ({filter_band}) = {zp_val:.2f} +/- {zp_err:.2f}"

        ax.set_title(title_str, fontsize=12, weight="bold")
        ax.set_xlabel("Detector Column (X)")
        ax.set_ylabel("Detector Row (Y)")

        out_plot = self.output_dir / f"phot_{image_path.stem}.png"
        plt.tight_layout()
        plt.savefig(out_plot, dpi=200)
        plt.close(fig)
        logger.info("Saved photometry overlay plot -> %s", out_plot.name)

        return cat_file

    def run_spectroscopy_on_pair(self, path_a: Path, path_b: Path) -> Path:
        """Run ABBA nod pair subtraction, 2D tracing, and 1D optimal extraction."""
        log_section(f"SPECTROSCOPY: NOD PAIR ({path_a.name} - {path_b.name})")
        img_a, hdr_a = load_fits(path_a)
        img_b, _ = load_fits(path_b)

        # 1. Nod Pair Subtraction (A - B)
        diff_2d = subtract_nod_pair(img_a, img_b)

        # 2. Trace Centroiding
        y_trace = fit_spectral_trace(diff_2d, poly_order=2)

        # 3. Optimal Extraction (Horne's Algorithm)
        flux_1d, flux_err_1d = extract_optimal_spectrum(diff_2d, y_trace=y_trace)
        pixels = np.arange(len(flux_1d))

        # 4. Dispersion Calibration (Standard TIRSPEC default nominal dispersion: ~1.0 to 2.5 um)
        disp_coeffs = [1.5e-3, 1.0]
        wavelengths = calibrate_wavelength(pixels, disp_coeffs)

        # 5. Save 1D Spectrum to CSV
        spec_csv = self.output_dir / f"spec_{path_a.stem}_minus_{path_b.stem}.csv"
        with open(spec_csv, "w") as fp:
            fp.write("pixel,wavelength_um,flux,flux_err\n")
            for px, lam, flx, err in zip(pixels, wavelengths, flux_1d, flux_err_1d):
                fp.write(f"{px},{lam:.4f},{flx:.3f},{err:.3f}\n")
        logger.info("Saved 1D extracted spectrum -> %s", spec_csv.name)

        # 6. Diagnostic 2D/1D Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 2]})

        # Panel 1: 2D Difference Frame + Overlaid Trace
        vmin, vmax = np.percentile(diff_2d, 5), np.percentile(diff_2d, 95)
        im = axes[0].imshow(diff_2d, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        axes[0].plot(pixels, y_trace, color="yellow", lw=1.5, label="Optimal Trace")
        axes[0].set_title(f"A - B Nod Difference Frame with Fitted Trace", fontsize=11, weight="bold")
        axes[0].set_ylabel("Spatial (Y)")
        axes[0].legend(loc="upper right")
        plt.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02)

        # Panel 2: 1D Extracted Spectrum
        axes[1].plot(wavelengths, flux_1d, color="black", lw=1.2, label=r"Net Flux $S(\lambda)$")
        axes[1].fill_between(wavelengths, flux_1d - flux_err_1d, flux_1d + flux_err_1d, color="gray", alpha=0.3)
        axes[1].set_title(f"Extracted 1D Spectrum (Horne's Algorithm)", fontsize=11, weight="bold")
        axes[1].set_xlabel(r"Wavelength ($\mu$m)")
        axes[1].set_ylabel("Flux (ADU / frame)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_plot = self.output_dir / f"spec_{path_a.stem}_diagnostic.png"
        plt.savefig(out_plot, dpi=200)
        plt.close(fig)
        logger.info("Saved spectroscopy diagnostic plot -> %s", out_plot.name)

        return spec_csv

    def run(self, raw_dir: str | Path, mode: str = "all") -> dict[str, Any]:
        """Execute the end-to-end reduction workflow."""
        log_section(f"AUTEX PIPELINE EXECUTION (MODE: {mode.upper()})")
        target_path = Path(raw_dir).expanduser().resolve()

        if mode == "photometry":
            if target_path.is_file() and target_path.suffix in [".fits", ".fit"]:
                fits_images = [target_path]
            else:
                fits_images = sorted(list(target_path.glob("*.fits*")))

            if not fits_images:
                logger.warning("No 2D FITS images found in %s for photometry.", target_path)
                return {}

            cat_paths = [self.run_photometry_on_image(p) for p in fits_images]
            log_section("PHOTOMETRY RUN COMPLETE")
            return {"photometry_catalogs": cat_paths}

        if mode == "spectroscopy":
            fits_images = sorted(list(target_path.glob("*.fits*"))) if target_path.is_dir() else [target_path]
            if len(fits_images) < 2:
                logger.warning("Spectroscopy mode requires at least 2 frames for an AB nod pair.")
                return {}
            spec_csv = self.run_spectroscopy_on_pair(fits_images[0], fits_images[1])
            log_section("SPECTROSCOPY RUN COMPLETE")
            return {"spectroscopy_catalog": spec_csv}

        self.catalog.scan_directory(target_path)
        pairing_tree = self.catalog.pair_science_and_calibrations()

        results: dict[str, list[Path]] = {}
        for group_name, group_data in pairing_tree.items():
            dark_records = group_data.get("matching_darks", [])
            dark_model = self.build_master_dark_model(dark_records)
            saved_images = self.reduce_science_group(group_name, group_data, dark_model)
            results[group_name] = saved_images

            if mode == "all":
                for img_path in saved_images:
                    self.run_photometry_on_image(img_path)

        log_section("PIPELINE REDUCTION COMPLETE")
        return results
