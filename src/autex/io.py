"""Standardized FITS file ingestion, NDR stacking, and data validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import numpy as np
from astropy.io import fits

from autex.logging import get_logger

logger = get_logger("io")

STANDARD_HEADER_MAP: dict[str, list[str]] = {
    "exptime": ["EXPTIME", "EXP_TIME", "ITIME"],
    "date_obs": ["DATE-OBS", "DATE", "OBS-DATE"],
    "filter": ["FILTER", "FILT1", "FILTER1"],
    "object": ["OBJECT", "OBJNAME", "TARGET"],
    "nreads": ["NREADS", "NDR", "RAMP_N"],
}


def load_fits(
    filepath: str | Path,
    ext: int = 0,
    memmap: bool = False,
) -> tuple[np.ndarray, fits.Header]:
    """Read a single FITS file and extract its primary array and header."""
    path = Path(filepath).expanduser().resolve()
    if not path.is_file():
        logger.error("FITS file not found: %s", path)
        raise FileNotFoundError(f"FITS file not found: {path}")

    with fits.open(path, memmap=memmap) as hdul:
        header = hdul[ext].header.copy()
        data = hdul[ext].data

        if data is None:
            for i in range(1, len(hdul)):
                if hdul[i].data is not None:
                    header = hdul[i].header.copy()
                    data = hdul[i].data
                    break

        if data is None:
            raise ValueError(f"No image data found in FITS: {path}")

        pixel_array = np.array(data, dtype=np.float32)

    return pixel_array, header


def stack_ndr_sequence(
    fits_files: list[Path | str],
) -> tuple[np.ndarray, fits.Header]:
    """Sort individual 2D NDR frame slices and stack them into a 3D SUTR cube.

    Parameters
    ----------
    fits_files : list[Path | str]
        List of paths belonging to a single sequence (e.g. NGC6604-92-debug-*.fits).

    Returns
    -------
    tuple[np.ndarray, fits.Header]
        3D cube of shape (N_reads, 1024, 1024) and base header.
    """
    def extract_frame_idx(p: Path | str) -> int:
        name = Path(p).stem
        nums = re.findall(r"\d+", name)
        return int(nums[-1]) if nums else 0

    sorted_files = sorted([Path(f) for f in fits_files], key=extract_frame_idx)
    logger.info("Assembling 3D SUTR cube from %d sequential 2D NDR slices.", len(sorted_files))

    first_data, base_header = load_fits(sorted_files[0])
    ny, nx = first_data.shape

    cube = np.zeros((len(sorted_files), ny, nx), dtype=np.float32)
    cube[0] = first_data

    for idx, fpath in enumerate(sorted_files[1:], start=1):
        data, _ = load_fits(fpath)
        cube[idx] = data

    logger.info("Successfully assembled cube with shape %s", cube.shape)
    return cube, base_header


def get_standard_header(header: fits.Header) -> dict[str, Any]:
    """Extract standard astronomical parameters regardless of telescope dialect."""
    standard_meta: dict[str, Any] = {}
    for std_key, candidates in STANDARD_HEADER_MAP.items():
        standard_meta[std_key] = None
        for candidate in candidates:
            if candidate in header:
                standard_meta[std_key] = header[candidate]
                break
    return standard_meta


def validate_sutr_cube(
    data: np.ndarray,
    expected_frames: int | None = None,
    expected_dim: tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """Validate 2D or 3D astronomical arrays."""
    if data.ndim == 2:
        return data

    if data.ndim == 3:
        n_frames, ny, nx = data.shape
        if (ny, nx) != expected_dim:
            raise ValueError(f"Spatial dimensions ({ny}, {nx}) do not match expected {expected_dim}")
        return data

    raise ValueError(f"Unsupported array dimensionality: ndim={data.ndim} (shape={data.shape})")
