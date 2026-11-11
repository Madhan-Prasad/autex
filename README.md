# autex (Automated TIFR Extractor)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Astropy](https://img.shields.io/badge/powered%20by-Astropy-orange.svg)](https://www.astropy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/badge/release-v0.1.0-green.svg)](https://github.com/Madhan-Prasad/autex/releases)
[![Docs](https://img.shields.io/badge/docs-Sphinx-brightgreen.svg)](https://madhan-prasad.github.io/autex/)

**`autex`** (*Automated TIFR Extractor*) is a native Python data reduction and extraction pipeline developed for near-infrared (NIR) imaging and spectroscopy. It is designed and optimized specifically for **TIRSPEC** (TIFR Near Infrared Spectrometer and Imager) mounted on the 2.0-m Himalayan Chandra Telescope (HCT) at IAO, Hanle, Ladakh.

Engineered as a modern, scriptable replacement for legacy semi-automated pipelines (e.g., TOIREx), `autex` executes end-to-end processing without requiring manual terminal regex inputs or external text-editor intervention: from raw Sample-Up-The-Ramp (SUTR) Non-Destructive Read (NDR) detector frames to 2MASS-calibrated point-source aperture photometry and Horne optimal 1D spectral extraction.

---

## Key Capabilities

- **Electronic Pedestal & Systematics Removal:** Vectorized initial reset-frame ($D_0$) subtraction, 4-quadrant amplifier decoupling ($4 \times 512 \times 512$), and row-dependent readout clock delay compensation via cubic spline interpolation.
- **SUTR Temporal Slope Fitting:** Fast, vectorized temporal least-squares linear regression across non-destructive reads to extract detector count rates ($\text{ADU/frame}$) while mitigating reset decay anomalies.
- **Automated Source Photometry:** Dynamic 2D spatial background mesh modeling (`photutils.Background2D`), empirical seeing measurement (FWHM calculation), and circular aperture photometry with local sky annuli.
- **Astrometric Calibration & Zero-Point Fitting:** Automated 2MASS Point Source Catalog cross-matching with international VizieR mirror failover (`vizier.iucaa.in`, `cfa.harvard.edu`, `cds.unistra.fr`) to calibrate instrumental magnitudes onto standard $J, H, K_s$ systems.
- **Spectroscopy Engine:** Spatial $A - B$ dithered nod-pair subtraction for night-sky airglow cancellation, 2D centroid polynomial trace fitting, Horne variance-weighted optimal 1D extraction, and nominal wavelength dispersion mapping.
- **Unified CLI:** Single-command execution covering all reduction workflows: `--mode slope`, `--mode photometry`, `--mode spectroscopy`, or `--mode all`.

---

## Pipeline Architecture

```text
                  Raw SUTR FITS Reads / Cubes (NDR Ramps)
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │             Electronic Calibration Engine               │
       │  • D_0 Pedestal Subtraction (Reset Noise Elimination)   │
       │  • Amplifier Quadrant Splitting (4 x 512 x 512)         │
       │  • Inter-Row Readout Delay Spline Interpolation         │
       │  • Vectorized OLS SUTR Slope Fitting (ADU / frame)      │
       └────────────────────────────┬────────────────────────────┘
                                    │
                    Calibrated 2D Slope FITS Images
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│     Photometry & Astrometry     │   │       Spectroscopy Engine       │
│ • 2D Background Mesh (SExtractor│   │ • ABBA Nod Pair (A - B) Sky     │
│   style background & RMS)       │   │   Emission Cancellation         │
│ • Empirical Seeing (FWHM) Fit   │   │ • 2D Spatial Centroid Tracing   │
│ • Aperture Photometry + Annuli  │   │ • Horne Optimal 1D Extraction   │
│ • 2MASS Cross-Match & ZP Fitting│   │ • Wavelength Dispersion Mapping │
└─────────────────────────────────┘   └─────────────────────────────────┘
