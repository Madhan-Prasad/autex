"""Automated unit tests for autex.catalog module."""

from pathlib import Path
import numpy as np
from astropy.io import fits
import pytest

from autex.catalog import ObservationCatalog, infer_frame_type


def test_frame_type_inference():
    hdr_dark = fits.Header({"OBJECT": "Dark_Run", "OBSTYPE": "DARK"})
    assert infer_frame_type("Dark_Run", hdr_dark) == "DARK"

    hdr_flat = fits.Header({"OBJECT": "Dome_Flat", "IMAGETYP": "FLAT"})
    assert infer_frame_type("Dome_Flat", hdr_flat) == "FLAT"

    hdr_sci = fits.Header({"OBJECT": "FU_Ori", "OBSTYPE": "OBJECT"})
    assert infer_frame_type("FU_Ori", hdr_sci) == "SCIENCE"


def test_catalog_scanning_and_pairing(tmp_path: Path):
    # Create synthetic FITS test files in a temporary directory
    file_defs = [
        ("dark_1.fits", "DARK", "OPEN", 10.0),
        ("dark_2.fits", "DARK", "OPEN", 10.0),
        ("flat_Ks_1.fits", "FLAT", "Ks", 5.0),
        ("sci_fuori_1.fits", "FU_Ori", "Ks", 10.0),
        ("sci_fuori_2.fits", "FU_Ori", "Ks", 10.0),
    ]

    for fname, target, filt, exp in file_defs:
        hdr = fits.Header()
        hdr["OBJECT"] = target
        hdr["FILTER"] = filt
        hdr["EXPTIME"] = exp
        hdu = fits.PrimaryHDU(data=np.zeros((10, 10), dtype=np.float32), header=hdr)
        hdu.writeto(tmp_path / fname)

    catalog = ObservationCatalog()
    frames = catalog.scan_directory(tmp_path)
    assert len(frames) == 5

    darks = catalog.get_by_type("DARK")
    assert len(darks) == 2

    pairing = catalog.pair_science_and_calibrations()
    assert "FU_Ori_Ks_10.0s" in pairing
    fuori_group = pairing["FU_Ori_Ks_10.0s"]
    assert len(fuori_group["science_frames"]) == 2
    assert len(fuori_group["matching_darks"]) == 2
    assert len(fuori_group["matching_flats"]) == 1
