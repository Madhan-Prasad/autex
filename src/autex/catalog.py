"""Deterministic observation file cataloging and calibration grouping engine for autex."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from astropy.io import fits

from autex.io import get_standard_header, load_fits
from autex.logging import get_logger

logger = get_logger("catalog")


@dataclass
class FrameMetadata:
    """Standardized representation of a single observed astronomical frame."""

    filepath: str
    filename: str
    target: str
    frame_type: str
    filter_band: str
    exptime: float
    date_obs: str
    nreads: int | None = None
    ra: float | None = None
    dec: float | None = None


def infer_frame_type(target_name: str, header: fits.Header) -> str:
    """Deterministically categorize a frame without manual regex terminal prompts.

    Parameters
    ----------
    target_name : str
        Target object string found in the FITS header.
    header : fits.Header
        Complete FITS header object.

    Returns
    -------
    str
        One of 'DARK', 'FLAT', 'LAMP', 'SKY', or 'SCIENCE'.
    """
    clean_target = str(target_name).strip().upper()
    obstype = str(header.get("OBSTYPE", "")).strip().upper()
    imagetyp = str(header.get("IMAGETYP", "")).strip().upper()

    combined_info = f"{clean_target} {obstype} {imagetyp}"

    if any(k in combined_info for k in ["DARK", "BIAS"]):
        return "DARK"
    if any(k in combined_info for k in ["FLAT", "LAMPON", "LAMPOFF", "DOMEFLAT", "SKYFLAT"]):
        return "FLAT"
    if any(k in combined_info for k in ["ARC", "LAMP", "NEON", "ARGON"]):
        return "LAMP"
    if any(k in combined_info for k in ["SKY", "OFF"]):
        return "SKY"

    return "SCIENCE"


class ObservationCatalog:
    """Scans, categorizes, and organizes raw observation frames for pipeline ingestion."""

    def __init__(self) -> None:
        self.frames: list[FrameMetadata] = []

    def scan_directory(self, raw_dir: str | Path, pattern: str = "*.fits*") -> list[FrameMetadata]:
        """Scan a folder, parse FITS metadata, and populate the catalog.

        Parameters
        ----------
        raw_dir : str | Path
            Directory containing raw observational FITS data.
        pattern : str, optional
            Glob pattern for identifying FITS files, by default '*.fits*'.

        Returns
        -------
        list[FrameMetadata]
            List of indexed and categorized frame metadata records.
        """
        directory = Path(raw_dir).expanduser().resolve()
        if not directory.is_dir():
            logger.error("Scan directory does not exist: %s", directory)
            raise NotADirectoryError(f"Directory not found: {directory}")

        fits_paths = sorted(directory.glob(pattern))
        logger.info("Scanning %d files matching '%s' in: %s", len(fits_paths), pattern, directory)

        self.frames = []
        for path in fits_paths:
            try:
                with fits.open(path) as hdul:
                    # Look for primary header with metadata
                    hdr = hdul[0].header
                    if len(hdul) > 1 and len(hdr) <= 8 and hdul[1].header is not None:
                        hdr = hdul[1].header

                    std_meta = get_standard_header(hdr)
                    target_name = std_meta.get("object") or "UNKNOWN"
                    frame_type = infer_frame_type(target_name, hdr)
                    exptime = float(std_meta.get("exptime") or 0.0)
                    date_obs = str(std_meta.get("date_obs") or "")
                    filter_band = str(std_meta.get("filter") or "OPEN")
                    nreads = std_meta.get("nreads")

                    record = FrameMetadata(
                        filepath=str(path),
                        filename=path.name,
                        target=target_name,
                        frame_type=frame_type,
                        filter_band=filter_band,
                        exptime=exptime,
                        date_obs=date_obs,
                        nreads=int(nreads) if nreads is not None else None,
                    )
                    self.frames.append(record)

            except Exception as err:
                logger.warning("Skipping unreadable or invalid FITS file %s: %s", path.name, err)

        logger.info("Successfully cataloged %d valid frames.", len(self.frames))
        return self.frames

    def get_by_type(self, frame_type: str) -> list[FrameMetadata]:
        """Filter catalog entries by their classified frame type."""
        target_type = frame_type.upper()
        return [f for f in self.frames if f.frame_type == target_type]

    def pair_science_and_calibrations(self) -> dict[str, Any]:
        """Automatically match science target observations with matching darks and flats.

        Returns
        -------
        dict[str, Any]
            Grouped tree pairing science exposures with corresponding calibration sets.
        """
        darks = self.get_by_type("DARK")
        flats = self.get_by_type("FLAT")
        science = self.get_by_type("SCIENCE")

        pairing_tree: dict[str, Any] = {}

        for frame in science:
            key = f"{frame.target}_{frame.filter_band}_{frame.exptime}s"
            if key not in pairing_tree:
                # Find matching darks (matching exposure time or longest available ramp)
                matched_darks = [
                    d for d in darks if abs(d.exptime - frame.exptime) < 0.1
                ]
                if not matched_darks and darks:
                    # Fallback to closest/longest dark
                    matched_darks = sorted(darks, key=lambda d: abs(d.exptime - frame.exptime))[:4]

                # Find matching flats with the same optical filter
                matched_flats = [f for f in flats if f.filter_band == frame.filter_band]

                pairing_tree[key] = {
                    "target": frame.target,
                    "filter": frame.filter_band,
                    "exptime": frame.exptime,
                    "science_frames": [],
                    "matching_darks": [asdict(d) for d in matched_darks],
                    "matching_flats": [asdict(f) for f in matched_flats],
                }

            pairing_tree[key]["science_frames"].append(asdict(frame))

        return pairing_tree
