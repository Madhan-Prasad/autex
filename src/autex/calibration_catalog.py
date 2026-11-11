"""Automated 2MASS catalog cross-matching and photometric zero-point calibration for autex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from astroquery.vizier import Vizier

from autex.logging import get_logger
from autex.photometry import PhotometryRecord

logger = get_logger("catalog_calib")

TWOMASS_FILTER_MAP: dict[str, str] = {
    "J": "Jmag",
    "H": "Hmag",
    "K": "Kmag",
    "KS": "Kmag",
}

# Standard VizieR mirrors across global astronomical centers
VIZIER_MIRRORS = [
    "vizier.cds.unistra.fr",
    "vizier.cfa.harvard.edu",
    "vizier.iucaa.in",
    "vizier.nao.ac.jp",
]


@dataclass
class ZeroPointSolution:
    """Photometric zero-point calibration solution."""
    zero_point: float
    zero_point_err: float
    num_matched_stars: int
    filter_band: str


def query_2mass_catalog(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 6.0,
    max_mag: float = 16.0,
) -> list[dict[str, Any]]:
    """Query 2MASS Point Source Catalog with automatic mirror failover.

    Parameters
    ----------
    ra_deg : float
        Right Ascension in decimal degrees.
    dec_deg : float
        Declination in decimal degrees.
    radius_arcmin : float, optional
        Search radius in arcminutes, by default 6.0.
    max_mag : float, optional
        Faint magnitude cutoff, by default 16.0.

    Returns
    -------
    list[dict[str, Any]]
        Catalog star records.
    """
    logger.info("Querying 2MASS PSC at (RA=%.4f, Dec=%.4f) with radius %.1f arcmin", ra_deg, dec_deg, radius_arcmin)
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

    columns = ["RAJ2000", "DEJ2000", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag"]
    tables = None

    for mirror in VIZIER_MIRRORS:
        try:
            logger.debug("Attempting query via VizieR mirror: %s", mirror)
            viz = Vizier(
                vizier_server=mirror,
                columns=columns,
                column_filters={"Kmag": f"<{max_mag}"},
                row_limit=1000,
                timeout=15,
            )
            tables = viz.query_region(coord, radius=radius_arcmin * u.arcmin, catalog="II/246/out")
            if tables and len(tables) > 0:
                logger.info("Successfully retrieved 2MASS catalog via %s", mirror)
                break
        except Exception as err:
            logger.debug("Mirror %s failed (%s), trying next...", mirror, err)
            continue

    if not tables or len(tables) == 0:
        logger.warning("All VizieR mirrors failed or returned empty results. Catalog calibration will be skipped.")
        return []

    table = tables[0]
    records = []
    for row in table:
        records.append({
            "ra": float(row["RAJ2000"]),
            "dec": float(row["DEJ2000"]),
            "Jmag": float(row["Jmag"]) if not np.isnan(row["Jmag"]) else None,
            "Hmag": float(row["Hmag"]) if not np.isnan(row["Hmag"]) else None,
            "Kmag": float(row["Kmag"]) if not np.isnan(row["Kmag"]) else None,
        })

    logger.info("Retrieved %d 2MASS reference stars.", len(records))
    return records


def calculate_photometric_zero_point(
    phot_records: list[PhotometryRecord],
    catalog_stars: list[dict[str, Any]],
    wcs: WCS,
    filter_band: str = "Ks",
    match_radius_arcsec: float = 2.5,
    sigma_clip_val: float = 2.5,
) -> ZeroPointSolution:
    """Cross-match detected sources with 2MASS stars and solve for the zero-point."""
    band_key = TWOMASS_FILTER_MAP.get(filter_band.upper(), "Kmag")
    valid_cat = [s for s in catalog_stars if s.get(band_key) is not None]

    if not phot_records or not valid_cat:
        logger.warning("Insufficient sources or catalog stars for zero-point fitting.")
        return ZeroPointSolution(zero_point=np.nan, zero_point_err=np.nan, num_matched_stars=0, filter_band=filter_band)

    det_x = np.array([r.x for r in phot_records])
    det_y = np.array([r.y for r in phot_records])
    det_ra, det_dec = wcs.all_pix2world(det_x, det_y, 0)
    det_coords = SkyCoord(ra=det_ra * u.deg, dec=det_dec * u.deg)

    cat_coords = SkyCoord(
        ra=[s["ra"] for s in valid_cat] * u.deg,
        dec=[s["dec"] for s in valid_cat] * u.deg,
    )

    idx_cat, sep2d, _ = det_coords.match_to_catalog_sky(cat_coords)
    matches = sep2d < (match_radius_arcsec * u.arcsec)

    zp_candidates = []
    for det_idx, is_match in enumerate(matches):
        if not is_match:
            continue
        rec = phot_records[det_idx]
        if np.isnan(rec.instrumental_mag):
            continue

        cat_mag = valid_cat[idx_cat[det_idx]][band_key]
        zp_val = cat_mag - rec.instrumental_mag
        zp_candidates.append(zp_val)

    if not zp_candidates:
        logger.warning("No detected stars cross-matched with 2MASS within %.1f arcsec.", match_radius_arcsec)
        return ZeroPointSolution(zero_point=np.nan, zero_point_err=np.nan, num_matched_stars=0, filter_band=filter_band)

    zp_arr = np.array(zp_candidates, dtype=np.float64)
    sig_clip = SigmaClip(sigma=sigma_clip_val)
    clipped = sig_clip(zp_arr)
    valid_zps = clipped.compressed()

    if len(valid_zps) == 0:
        valid_zps = zp_arr

    zp_mean = float(np.mean(valid_zps))
    zp_err = float(np.std(valid_zps) / np.sqrt(len(valid_zps))) if len(valid_zps) > 1 else 0.05

    logger.info("Photometric calibration solution [%s]: ZP = %.3f +/- %.3f (%d matched stars)",
                filter_band, zp_mean, zp_err, len(valid_zps))
    return ZeroPointSolution(
        zero_point=zp_mean,
        zero_point_err=zp_err,
        num_matched_stars=len(valid_zps),
        filter_band=filter_band,
    )
