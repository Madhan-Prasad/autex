"""Automated unit tests for autex.config and autex.io modules."""

import numpy as np
from astropy.io import fits
from pathlib import Path
import pytest

from autex.config import Config
from autex.io import get_standard_header, validate_sutr_cube


def test_config_loader():
    cfg = Config.from_yaml("configs/default_config.yaml")
    assert cfg.get("instrument.name") == "TIRSPEC"
    assert cfg.get("instrument.detector.dim_x") == 1024
    assert cfg.get("instrument.detector.dim_y") == 1024
    assert cfg.get("sutr.total_reads") == 112
    assert cfg.get("nonexistent.key", "fallback") == "fallback"


def test_standard_header_extraction():
    hdr = fits.Header()
    hdr["ITIME"] = 5.0
    hdr["OBJECT"] = "FU_Ori"
    hdr["FILT1"] = "Ks"

    meta = get_standard_header(hdr)
    assert meta["exptime"] == 5.0
    assert meta["object"] == "FU_Ori"
    assert meta["filter"] == "Ks"


def test_sutr_cube_validation():
    # Synthetic cube with 112 reads of 1024x1024
    synthetic_cube = np.zeros((112, 1024, 1024), dtype=np.float32)
    validated = validate_sutr_cube(synthetic_cube, expected_frames=112, expected_dim=(1024, 1024))
    assert validated.shape == (112, 1024, 1024)

    # Fail on wrong spatial dimensions
    wrong_dim = np.zeros((112, 512, 512), dtype=np.float32)
    with pytest.raises(ValueError, match="Spatial dimensions"):
        validate_sutr_cube(wrong_dim, expected_dim=(1024, 1024))
