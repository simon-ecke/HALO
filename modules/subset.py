"""Spatial subsetting utilities for fast development iteration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import laspy
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from .io_utils import get_logger

logger = get_logger("tree_height.subset")


def compute_subset_bounds(
    lidar_path: Path | str,
    size_m: float = 10.0,
) -> Tuple[float, float, float, float]:
    """Compute a square bounding box centered on the LiDAR dataset.

    Returns (xmin, ymin, xmax, ymax) in the LiDAR's native CRS.
    """
    with laspy.open(lidar_path) as src:
        header = src.header
        x_mid = (header.mins[0] + header.maxs[0]) / 2.0
        y_mid = (header.mins[1] + header.maxs[1]) / 2.0

    half = size_m / 2.0
    bounds = (x_mid - half, y_mid - half, x_mid + half, y_mid + half)

    logger.info(
        "Subset bounds (%.0f×%.0f m centered): xmin=%.1f, ymin=%.1f, xmax=%.1f, ymax=%.1f",
        size_m, size_m, *bounds,
    )
    return bounds


def subset_lidar(
    lidar_path: Path | str,
    bounds: Tuple[float, float, float, float],
    output_path: Optional[Path | str] = None,
) -> Path:
    """Clip a LiDAR file to a bounding box and write to a temporary LAZ.

    Parameters
    ----------
    lidar_path : Path
        Input LAZ/LAS file.
    bounds : tuple
        (xmin, ymin, xmax, ymax) in the LiDAR's CRS.
    output_path : Path, optional
        Where to write the clipped file. If None, writes next to the input
        with a '_subset' suffix.

    Returns
    -------
    Path
        Path to the clipped LAZ file.
    """
    lidar_path = Path(lidar_path)
    xmin, ymin, xmax, ymax = bounds

    if output_path is None:
        output_path = lidar_path.parent / f"{lidar_path.stem}_subset.laz"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    las = laspy.read(lidar_path)
    x = np.asarray(las.x, dtype=float)
    y = np.asarray(las.y, dtype=float)

    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    subset = las.points[mask]

    if len(subset) == 0:
        raise ValueError(
            f"No LiDAR points found in subset bounds. "
            f"Bounds: ({xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f})"
        )

    new_las = laspy.LasData(las.header)
    new_las.points = subset
    new_las.write(output_path)

    logger.info(
        "Subset LiDAR: %d/%d points (%.1f%%) -> %s",
        len(subset), len(las.points),
        100.0 * len(subset) / len(las.points),
        output_path.name,
    )
    return output_path


def subset_raster(
    raster_path: Path | str,
    bounds: Tuple[float, float, float, float],
    output_path: Optional[Path | str] = None,
) -> Path:
    """Clip a raster (DSM/DTM/nDSM) to a bounding box.

    Parameters
    ----------
    raster_path : Path
        Input GeoTIFF.
    bounds : tuple
        (xmin, ymin, xmax, ymax) in the raster's CRS.
    output_path : Path, optional
        Where to write the clipped raster. If None, writes with '_subset' suffix.

    Returns
    -------
    Path
        Path to the clipped GeoTIFF.
    """
    raster_path = Path(raster_path)
    xmin, ymin, xmax, ymax = bounds

    if output_path is None:
        output_path = raster_path.parent / f"{raster_path.stem}_subset.tif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        window = window.round_offsets().round_lengths()

        data = src.read(window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    logger.info(
        "Subset raster: %d×%d px -> %s",
        data.shape[2], data.shape[1], output_path.name,
    )
    return output_path
