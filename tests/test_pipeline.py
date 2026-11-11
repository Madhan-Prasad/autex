"""Automated end-to-end integration test for autex.pipeline."""

from pathlib import Path
import numpy as np
from astropy.io import fits
import pytest

from autex.config import Config
from autex.pipeline import ReductionPipeline


def test_full_pipeline_run(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir()

    # Generate synthetic 20-read SUTR cubes
    n_reads = 20
    shape = (n_reads, 1024, 1024)

    # 1. Dark exposure: baseline pedestal + slight relaxation
    dark_data = np.zeros(shape, dtype=np.float32)
    for i in range(n_reads):
        dark_data[i] = 4000.0 + 50.0 * np.exp(-i / 5.0)

    hdr_dark = fits.Header({"OBJECT": "DARK", "OBSTYPE": "DARK", "EXPTIME": 18.0})
    fits.writeto(raw_dir / "dark_001.fits", dark_data, hdr_dark)

    # 2. Science exposure: dark decay + linear starlight ramp
    sci_data = np.zeros(shape, dtype=np.float32)
    for i in range(n_reads):
        sci_data[i] = 4000.0 + 50.0 * np.exp(-i / 5.0) + (i * 12.0)

    hdr_sci = fits.Header({"OBJECT": "FU_Ori", "OBSTYPE": "OBJECT", "FILTER": "Ks", "EXPTIME": 18.0})
    fits.writeto(raw_dir / "sci_001.fits", sci_data, hdr_sci)

    # Prepare Config pointing to tmp directories
    cfg = Config({
        "sutr": {"total_reads": n_reads},
        "paths": {"output_dir": str(reduced_dir)},
    })

    pipeline = ReductionPipeline(cfg)
    results = pipeline.run(raw_dir)

    assert "FU_Ori_Ks_18.0s" in results
    saved_slopes = results["FU_Ori_Ks_18.0s"]
    assert len(saved_slopes) == 1

    # Verify output 2D slope image
    slope_file = saved_slopes[0]
    assert slope_file.is_file()

    with fits.open(slope_file) as hdul:
        slope_img = hdul[0].data
        assert slope_img.shape == (1024, 1024)
        # Measured slope should be close to 12.0 ADU/frame
        assert np.isclose(np.median(slope_img), 12.0, atol=0.5)
