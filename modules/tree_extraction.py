"""Per-tree height extraction from ground truth, LiDAR, and nDSM."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def extract_ground_truth_tree_table(ground_truth_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return the canonical per-tree reference table from ground-truth points."""
    required = {"tree_id", "ground_truth_height", "x", "y", "geometry"}
    missing = required.difference(set(ground_truth_gdf.columns))
    if missing:
        raise ValueError(f"Ground-truth GeoDataFrame is missing required columns: {missing}")

    table = ground_truth_gdf[["tree_id", "ground_truth_height", "x", "y", "geometry"]].copy()
    table["tree_id"] = table["tree_id"].astype(str)
    return table


def extract_lidar_tree_heights(
    tree_points: pd.DataFrame,
    normalized_lidar_points: pd.DataFrame,
    top_coordinates: Optional[pd.DataFrame] = None,
    method: str = "percentile",
    search_radius_m: float = 1.5,
    percentile: float = 95.0,
    min_height_m: float = 0.5,
    max_height_m: Optional[float] = None,
) -> pd.DataFrame:
    """Extract one LiDAR canopy height estimate per tree using local neighborhoods.

    If top_coordinates is provided (from tree detection or harvest matching),
    extraction is performed at matched top positions instead of stem positions.
    """
    if normalized_lidar_points.empty:
        raise ValueError("Normalized LiDAR table is empty.")

    if search_radius_m <= 0:
        raise ValueError("search_radius_m must be > 0")

    valid_points = normalized_lidar_points[
        ["x", "y", "z_normalized"]
    ].dropna(subset=["x", "y", "z_normalized"])

    if valid_points.empty:
        raise ValueError("No valid normalized LiDAR points available after filtering NaN values.")

    coords = valid_points[["x", "y"]].to_numpy(dtype=float)
    heights_all = valid_points["z_normalized"].to_numpy(dtype=float)

    spatial_index = cKDTree(coords)
    area = math.pi * (search_radius_m**2)

    if top_coordinates is not None:
        search_points = _build_search_points(tree_points, top_coordinates)
    else:
        search_points = tree_points

    rows: list[dict[str, object]] = []

    for rec in search_points.itertuples(index=False):
        x_top = getattr(rec, "x_top", None)
        y_top = getattr(rec, "y_top", None)
        search_x = x_top if (x_top is not None and not np.isnan(x_top)) else rec.x
        search_y = y_top if (y_top is not None and not np.isnan(y_top)) else rec.y
        idx = spatial_index.query_ball_point([search_x, search_y], r=search_radius_m)

        if not idx:
            rows.append(
                {
                    "tree_id": rec.tree_id,
                    "lidar_height": np.nan,
                    "lidar_point_count": 0,
                    "lidar_point_density": 0.0,
                    "lidar_quality_flag": "no_points_in_radius",
                    "lidar_extraction_radius_m": search_radius_m,
                }
            )
            continue

        local_heights = heights_all[np.asarray(idx, dtype=int)]
        local_heights = local_heights[np.isfinite(local_heights)]
        local_heights = local_heights[local_heights >= min_height_m]
        if max_height_m is not None:
            local_heights = local_heights[local_heights <= max_height_m]

        if local_heights.size == 0:
            rows.append(
                {
                    "tree_id": rec.tree_id,
                    "lidar_height": np.nan,
                    "lidar_point_count": len(idx),
                    "lidar_point_density": len(idx) / area,
                    "lidar_quality_flag": "all_points_filtered",
                    "lidar_extraction_radius_m": search_radius_m,
                }
            )
            continue

        if method == "percentile":
            lidar_height = float(np.percentile(local_heights, percentile))
        elif method in {"local_max", "highest_filtered"}:
            lidar_height = float(np.max(local_heights))
        else:
            raise ValueError(
                "Unknown LiDAR extraction method. "
                "Use one of: 'percentile', 'local_max', 'highest_filtered'."
            )

        rows.append(
            {
                "tree_id": rec.tree_id,
                "lidar_height": lidar_height,
                "lidar_point_count": len(idx),
                "lidar_point_density": len(idx) / area,
                "lidar_quality_flag": "ok",
                "lidar_extraction_radius_m": search_radius_m,
            }
        )

    return pd.DataFrame(rows)


def _gaussian_smooth_with_nan(array: np.ndarray, sigma: float) -> np.ndarray:
    valid_mask = np.isfinite(array)
    filled = np.where(valid_mask, array, 0.0)

    smoothed_sum = gaussian_filter(filled, sigma=sigma)
    smoothed_weight = gaussian_filter(valid_mask.astype(float), sigma=sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = smoothed_sum / smoothed_weight
    smoothed[smoothed_weight == 0] = np.nan
    return smoothed


def _extract_circular_window_values(
    raster: np.ndarray,
    transform: rasterio.Affine,
    x: float,
    y: float,
    row: int,
    col: int,
    search_radius_m: float,
) -> np.ndarray:
    px_size = max(abs(transform.a), abs(transform.e))
    radius_px = int(np.ceil(search_radius_m / px_size))

    row_min = max(0, row - radius_px)
    row_max = min(raster.shape[0] - 1, row + radius_px)
    col_min = max(0, col - radius_px)
    col_max = min(raster.shape[1] - 1, col + radius_px)

    window = raster[row_min : row_max + 1, col_min : col_max + 1]
    if window.size == 0:
        return np.array([], dtype=float)

    row_grid, col_grid = np.indices(window.shape)
    row_grid = row_grid + row_min
    col_grid = col_grid + col_min

    # Newer rasterio versions return flattened arrays for 2D input
    xs, ys = rasterio.transform.xy(transform, row_grid, col_grid, offset="center")
    xs = np.asarray(xs, dtype=float).reshape(window.shape)
    ys = np.asarray(ys, dtype=float).reshape(window.shape)

    dist = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
    values = window[dist <= search_radius_m]
    return values[np.isfinite(values)]


def extract_ndsm_tree_heights(
    tree_points: pd.DataFrame,
    ndsm_path: Path | str,
    top_coordinates: Optional[pd.DataFrame] = None,
    method: str = "raster_percentile",
    search_radius_m: float = 1.5,
    percentile: float = 95.0,
    window_size_pixels: int = 5,
    smoothing_sigma: Optional[float] = None,
) -> pd.DataFrame:
    """Extract one photogrammetric canopy height estimate per tree from nDSM.

    If top_coordinates is provided (from tree detection or harvest matching),
    extraction is performed at matched top positions instead of stem positions.

    Methods:
        raster_percentile: Nth percentile within search radius (default, robust to outliers)
        raster_max: Maximum value within search radius
        focal_max: Maximum in a fixed pixel window
    """
    with rasterio.open(ndsm_path) as src:
        ndsm = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        width = src.width
        height = src.height

    if nodata is not None:
        ndsm = np.where(np.isclose(ndsm, nodata), np.nan, ndsm)

    if smoothing_sigma is not None and smoothing_sigma > 0:
        ndsm_work = _gaussian_smooth_with_nan(ndsm, sigma=smoothing_sigma)
    else:
        ndsm_work = ndsm

    half_window = max(1, window_size_pixels // 2)

    if top_coordinates is not None:
        search_points = _build_search_points(tree_points, top_coordinates)
    else:
        search_points = tree_points

    rows: list[dict[str, object]] = []

    for rec in search_points.itertuples(index=False):
        x_top = getattr(rec, "x_top", None)
        y_top = getattr(rec, "y_top", None)
        search_x = x_top if (x_top is not None and not np.isnan(x_top)) else rec.x
        search_y = y_top if (y_top is not None and not np.isnan(y_top)) else rec.y
        row, col = rasterio.transform.rowcol(transform, search_x, search_y)

        if row < 0 or row >= height or col < 0 or col >= width:
            rows.append(
                {
                    "tree_id": rec.tree_id,
                    "photogrammetry_height": np.nan,
                    "photo_nodata_hit": True,
                    "photo_quality_flag": "outside_raster_extent",
                    "photo_extraction_radius_m": search_radius_m,
                }
            )
            continue

        if method in ("raster_max", "raster_percentile"):
            values = _extract_circular_window_values(
                raster=ndsm_work,
                transform=transform,
                x=search_x,
                y=search_y,
                row=row,
                col=col,
                search_radius_m=search_radius_m,
            )
        elif method == "focal_max":
            row_min = max(0, row - half_window)
            row_max = min(height - 1, row + half_window)
            col_min = max(0, col - half_window)
            col_max = min(width - 1, col + half_window)
            window = ndsm_work[row_min : row_max + 1, col_min : col_max + 1]
            values = window[np.isfinite(window)]
        else:
            raise ValueError(
                "Unknown photogrammetry extraction method. "
                "Use 'raster_percentile', 'raster_max', or 'focal_max'."
            )

        if values.size == 0:
            rows.append(
                {
                    "tree_id": rec.tree_id,
                    "photogrammetry_height": np.nan,
                    "photo_nodata_hit": True,
                    "photo_quality_flag": "no_valid_raster_values",
                    "photo_extraction_radius_m": search_radius_m,
                }
            )
            continue

        if method == "raster_percentile":
            extracted_height = float(np.percentile(values, percentile))
        else:
            extracted_height = float(np.max(values))

        rows.append(
            {
                "tree_id": rec.tree_id,
                "photogrammetry_height": extracted_height,
                "photo_nodata_hit": False,
                "photo_quality_flag": "ok",
                "photo_extraction_radius_m": search_radius_m,
            }
        )

    return pd.DataFrame(rows)


def _build_search_points(
    tree_points: pd.DataFrame,
    top_coordinates: pd.DataFrame,
) -> pd.DataFrame:
    """Merge tree_points with matched top coordinates for extraction.

    Returns a DataFrame with columns from tree_points plus x_top, y_top
    from top_coordinates. Rows without a matched top keep x_top/y_top as NaN
    and will fall back to using stem x, y for extraction.
    """
    if "tree_id" not in top_coordinates.columns:
        raise ValueError("top_coordinates must contain a 'tree_id' column for merging.")

    if "x_removed" in top_coordinates.columns and "x_top" not in top_coordinates.columns:
        top_subset = top_coordinates[["tree_id", "x_removed", "y_removed"]].rename(
            columns={"x_removed": "x_top", "y_removed": "y_top"}
        )
    else:
        top_cols = ["tree_id"]
        if "x_top" in top_coordinates.columns:
            top_cols.append("x_top")
        if "y_top" in top_coordinates.columns:
            top_cols.append("y_top")
        top_subset = top_coordinates[top_cols]

    merged = tree_points.merge(top_subset, on="tree_id", how="left")
    return merged
