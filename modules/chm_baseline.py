"""Baseline tree detection: local maxima on the LiDAR canopy height model.

The simplest established alternative to a 3D segmentation such as SegmentAnyTree
(Popescu & Wynne 2004; `lmf()` in the R package lidR): a cell of the smoothed
canopy height model (CHM) is a tree top if no cell within a window around it is
higher. The window is either fixed or grows with the height of the cell, because
taller trees have wider crowns.

Height of a top = the highest cell of the unsmoothed CHM within 0.5 m of it.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from .io_utils import get_logger

logger = get_logger("chm_baseline")

Window = Union[float, Callable[[np.ndarray], np.ndarray]]


def fill_gaps(chm: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Cells without a return take the highest of their 3 x 3 neighbours (repeated); what is left becomes 0."""
    filled = chm.astype(np.float32).copy()
    for _ in range(iterations):
        empty = ~np.isfinite(filled)
        if not empty.any():
            break
        grown = ndimage.maximum_filter(np.where(empty, -np.inf, filled), size=3)
        take = empty & np.isfinite(grown)
        filled[take] = grown[take]
    filled[~np.isfinite(filled)] = 0.0
    return filled


def _disk(radius_px: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (xx ** 2 + yy ** 2) <= radius_px ** 2


def local_maxima_tops(
    chm: np.ndarray,
    transform: rasterio.Affine,
    window_m: Window,
    min_height_m: float = 3.0,
    smoothing_sigma_m: float = 0.25,
    height_radius_m: float = 0.5,
) -> pd.DataFrame:
    """Tree tops as local maxima of the smoothed CHM.

    Parameters
    ----------
    chm : CHM array (NaN = no return).
    window_m : window diameter in m, or a function of the (smoothed) cell height
        that returns the diameter, e.g. ``lambda h: 1.5 + 0.08 * h``.
    min_height_m : lowest top that counts.
    smoothing_sigma_m : Gaussian smoothing before the search (0 = none).

    Returns
    -------
    DataFrame with top_id, x_top, y_top, height (unsmoothed CHM), height_smoothed, window_m.
    """
    res = abs(transform.a)
    filled = fill_gaps(chm)
    smooth = ndimage.gaussian_filter(filled, smoothing_sigma_m / res) if smoothing_sigma_m > 0 else filled
    tall = smooth >= min_height_m

    diameter = window_m(smooth) if callable(window_m) else np.full(smooth.shape, float(window_m))
    radius_px = np.maximum(np.round(diameter / 2 / res), 1).astype(np.int32)

    is_top = np.zeros(smooth.shape, dtype=bool)
    for r in np.unique(radius_px[tall]):
        cells = tall & (radius_px == r)
        is_top |= cells & (smooth >= ndimage.maximum_filter(smooth, footprint=_disk(int(r))))

    # A flat top can give several touching cells: keep one per group
    groups, n = ndimage.label(is_top, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return pd.DataFrame(columns=["top_id", "x_top", "y_top", "height", "height_smoothed", "window_m"])
    centres = np.array(ndimage.center_of_mass(is_top, groups, np.arange(1, n + 1)))
    rows, cols = np.round(centres[:, 0]).astype(int), np.round(centres[:, 1]).astype(int)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")

    peak = ndimage.maximum_filter(filled, footprint=_disk(max(int(round(height_radius_m / res)), 1)))
    tops = pd.DataFrame({
        "top_id": np.arange(n), "x_top": np.asarray(xs, dtype=float), "y_top": np.asarray(ys, dtype=float),
        "height": peak[rows, cols].astype(float), "height_smoothed": smooth[rows, cols].astype(float),
        "window_m": diameter[rows, cols].astype(float),
    })
    logger.info("Local maxima: %d tops (min height %.1f m)", len(tops), min_height_m)
    return tops
