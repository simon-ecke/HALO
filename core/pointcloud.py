"""Point cloud and raster I/O helpers (copied from modules/field_tree_matching.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import LineString, Point

import logging

logger = logging.getLogger("core.pointcloud")

Bounds = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------

def clip_laz_to_bounds(
    src_path: Path | str,
    dst_path: Path | str,
    bounds: Bounds,
    chunk_size: int = 10_000_000,
) -> dict[str, object]:
    """Clip a LAS/LAZ file to (xmin, ymin, xmax, ymax) without loading it whole.

    Unlike subset.subset_lidar this streams the file in chunks, so it works on
    multi-GB flights. All dimensions (incl. PredInstance) are kept.
    """
    src_path, dst_path = Path(src_path), Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bounds

    n_in = n_out = 0
    with laspy.open(src_path) as reader:
        with laspy.open(dst_path, mode="w", header=reader.header) as writer:
            for points in reader.chunk_iterator(chunk_size):
                x = np.asarray(points.x)
                y = np.asarray(points.y)
                mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
                n_in += len(points)
                if mask.any():
                    writer.write_points(points[mask])
                    n_out += int(mask.sum())

    logger.info("Clipped %s: %d/%d points -> %s", src_path.name, n_out, n_in, dst_path.name)
    return {"path": str(dst_path), "points_in": n_in, "points_out": n_out}


def load_instance_cloud(path: Path | str, instance_dim: str = "PredInstance") -> pd.DataFrame:
    """Load x, y, z and the per-point tree instance id from a (clipped) SAT LAZ."""
    las = laspy.read(path)
    names = {d.lower(): d for d in las.point_format.dimension_names}
    if instance_dim.lower() not in names:
        raise ValueError(f"{path} has no '{instance_dim}' dimension. Available: {list(names.values())}")

    return pd.DataFrame(
        {
            "x": np.asarray(las.x, dtype=np.float64),
            "y": np.asarray(las.y, dtype=np.float64),
            "z": np.asarray(las.z, dtype=np.float32),
            "instance": np.asarray(las[names[instance_dim.lower()]], dtype=np.int32),
        }
    )


# ---------------------------------------------------------------------------
# Ground model and height above ground
# ---------------------------------------------------------------------------

def write_ground_candidates_laz(
    points: pd.DataFrame,
    output_path: Path | str,
    cell_size_m: float = 0.25,
) -> Path:
    """Write the lowest point per grid cell to a small LAZ.

    Ground filtering (CSF in lidar_dtm.create_lidar_dtm) only needs the lowest
    returns, so thinning a 50M-point clip this way keeps the DTM step fast.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = points["x"].to_numpy()
    y = points["y"].to_numpy()
    z = points["z"].to_numpy()
    ix = np.floor((x - x.min()) / cell_size_m).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_size_m).astype(np.int64)
    cell = ix * (iy.max() + 1) + iy

    order = np.lexsort((z, cell))
    first = np.ones(len(order), dtype=bool)
    first[1:] = cell[order][1:] != cell[order][:-1]
    lowest = order[first]

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([np.floor(x.min()), np.floor(y.min()), 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x[lowest], y[lowest], z[lowest].astype(np.float64)
    las.write(output_path)

    logger.info("Ground candidates: %d lowest points (%.2f m cells) -> %s", len(lowest), cell_size_m, output_path.name)
    return output_path


def sample_raster_bilinear(raster_path: Path | str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear raster sampling for many points (vectorized, unlike rasterio's sample())."""
    with rasterio.open(raster_path) as src:
        band = src.read(1).astype(np.float64)
        nodata = src.nodata
        inv = ~src.transform

    if nodata is not None:
        band[np.isclose(band, nodata)] = np.nan

    cols, rows = inv * (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    # Pixel centres sit at +0.5 in pixel coordinates
    values = ndimage.map_coordinates(band, [rows - 0.5, cols - 0.5], order=1, mode="constant", cval=np.nan)
    return values


def add_height_above_ground(points: pd.DataFrame, dtm_path: Path | str) -> pd.DataFrame:
    """Add column 'h' = z minus DTM elevation (metres above ground)."""
    ground = sample_raster_bilinear(dtm_path, points["x"].to_numpy(), points["y"].to_numpy())
    points["h"] = (points["z"].to_numpy(dtype=np.float64) - ground).astype(np.float32)
    return points


def rasterize_max(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    bounds: Bounds,
    resolution_m: float,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Grid points to a max-value raster (e.g. a LiDAR CHM). Empty cells are NaN."""
    xmin, ymin, xmax, ymax = bounds
    ncols = int(np.ceil((xmax - xmin) / resolution_m))
    nrows = int(np.ceil((ymax - ymin) / resolution_m))

    col = ((x - xmin) / resolution_m).astype(np.int64)
    row = ((ymax - y) / resolution_m).astype(np.int64)
    valid = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows) & np.isfinite(values)

    grid = np.full(nrows * ncols, -np.inf, dtype=np.float32)
    np.maximum.at(grid, row[valid] * ncols + col[valid], values[valid].astype(np.float32))
    grid[np.isinf(grid)] = np.nan
    return grid.reshape(nrows, ncols), from_origin(xmin, ymax, resolution_m, resolution_m)


def write_raster(array: np.ndarray, transform: rasterio.Affine, path: Path | str, crs: str, nodata: float = -9999.0) -> Path:
    """Write a single-band float32 GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": array.shape[0], "width": array.shape[1], "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": nodata,
        "compress": "lzw", "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(np.isnan(array), nodata, array).astype(np.float32), 1)
    return path

