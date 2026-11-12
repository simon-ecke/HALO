"""Tree top detection and stem-to-top matching for individual tree analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.spatial import cKDTree
from shapely.geometry import Point

from .io_utils import get_logger

logger = get_logger("tree_detection")


def detect_lidar_tree_tops(
    normalized_lidar_points: pd.DataFrame,
    cell_size: float = 0.5,
    min_height_m: float = 2.0,
    search_radius_m: float = 1.5,
    min_points_per_tree: int = 5,
) -> pd.DataFrame:
    """Detect tree tops as local maxima in a normalized 3D LiDAR point cloud.

    Algorithm:
    1. Grid the point cloud into 2D cells and find the highest point per cell.
    2. Filter candidates by minimum height threshold.
    3. Apply non-maximum suppression: keep only points that are the tallest
       within search_radius_m of their position.

    Parameters
    ----------
    normalized_lidar_points : pd.DataFrame
        Must have columns x, y, z_normalized.
    cell_size : float
        Grid resolution for initial candidate detection (meters).
    min_height_m : float
        Minimum normalized height to consider as tree top.
    search_radius_m : float
        Radius for local maxima validation / non-maximum suppression.
    min_points_per_tree : int
        Minimum number of LiDAR points within search_radius to accept a detection.

    Returns
    -------
    pd.DataFrame
        Detected tree tops with columns: top_id, x_top, y_top, z_top,
        point_count, point_density, quality_flag.
    """
    required = {"x", "y", "z_normalized"}
    missing = required - set(normalized_lidar_points.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    pts = normalized_lidar_points[["x", "y", "z_normalized"]].dropna().copy()
    pts = pts[pts["z_normalized"] >= min_height_m]

    if pts.empty:
        logger.warning("No points above min_height_m=%.1f. No tops detected.", min_height_m)
        return _empty_tops_df()

    pts["cell_x"] = (pts["x"] // cell_size * cell_size).astype(float)
    pts["cell_y"] = (pts["y"] // cell_size * cell_size).astype(float)

    idx_max = pts.groupby(["cell_x", "cell_y"])["z_normalized"].idxmax()
    candidates = pts.loc[idx_max].reset_index(drop=True)

    candidates = candidates.sort_values("z_normalized", ascending=False).reset_index(drop=True)
    candidate_coords = candidates[["x", "y"]].to_numpy(dtype=float)
    candidate_heights = candidates["z_normalized"].to_numpy(dtype=float)

    kept = _non_maximum_suppression(candidate_coords, candidate_heights, search_radius_m)
    tops = candidates.iloc[kept].reset_index(drop=True)

    all_coords = pts[["x", "y"]].to_numpy(dtype=float)
    all_tree = cKDTree(all_coords)
    area = math.pi * search_radius_m ** 2

    rows = []
    for i, row in tops.iterrows():
        indices = all_tree.query_ball_point([row["x"], row["y"]], r=search_radius_m)
        point_count = len(indices)
        point_density = point_count / area

        quality_flag = "ok"
        if point_count < min_points_per_tree:
            quality_flag = "low_point_density"

        rows.append({
            "top_id": i,
            "x_top": row["x"],
            "y_top": row["y"],
            "z_top": row["z_normalized"],
            "point_count": point_count,
            "point_density": point_density,
            "quality_flag": quality_flag,
        })

    result = pd.DataFrame(rows)
    logger.info("Detected %d tree tops from LiDAR (min_height=%.1f m)", len(result), min_height_m)
    return result


def detect_ndsm_tree_tops(
    ndsm_path: Path | str,
    kernel_size: int = 5,
    min_height_m: float = 2.0,
    min_separation_m: float = 1.0,
    smoothing_sigma: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Detect tree tops as local maxima in a 2D nDSM raster.

    Algorithm:
    1. Optionally smooth nDSM with Gaussian filter.
    2. Apply scipy.ndimage.maximum_filter to find local maxima.
    3. Filter by minimum height threshold.
    4. Apply minimum separation constraint using cKDTree.

    Parameters
    ----------
    ndsm_path : Path or str
        Path to nDSM GeoTIFF.
    kernel_size : int
        Window size for maximum filter (pixels).
    min_height_m : float
        Minimum nDSM value to consider as tree top.
    min_separation_m : float
        Minimum distance between detected peaks (meters).
    smoothing_sigma : float, optional
        Gaussian smoothing sigma before detection.

    Returns
    -------
    gpd.GeoDataFrame
        Detected tree tops with columns: top_id, x_top, y_top, height,
        quality_flag, geometry.
    """
    with rasterio.open(ndsm_path) as src:
        ndsm = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    if nodata is not None:
        ndsm = np.where(np.isclose(ndsm, nodata), np.nan, ndsm)

    if smoothing_sigma is not None and smoothing_sigma > 0:
        valid_mask = np.isfinite(ndsm)
        filled = np.where(valid_mask, ndsm, 0.0)
        smoothed_sum = gaussian_filter(filled, sigma=smoothing_sigma)
        smoothed_weight = gaussian_filter(valid_mask.astype(float), sigma=smoothing_sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            ndsm_work = smoothed_sum / smoothed_weight
        ndsm_work[smoothed_weight == 0] = np.nan
    else:
        ndsm_work = ndsm

    local_max = maximum_filter(
        np.where(np.isfinite(ndsm_work), ndsm_work, -np.inf),
        size=kernel_size,
    )
    is_peak = (ndsm_work == local_max) & np.isfinite(ndsm_work) & (ndsm_work >= min_height_m)

    peak_rows, peak_cols = np.where(is_peak)
    if peak_rows.size == 0:
        logger.warning("No tree tops detected in nDSM (min_height=%.1f m).", min_height_m)
        return _empty_tops_gdf(crs)

    xs, ys = rasterio.transform.xy(transform, peak_rows, peak_cols, offset="center")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    heights = ndsm_work[peak_rows, peak_cols]

    candidates = np.column_stack([xs, ys])
    kept = _non_maximum_suppression(candidates, heights, min_separation_m)

    rows = []
    for idx, k in enumerate(kept):
        rows.append({
            "top_id": idx,
            "x_top": float(xs[k]),
            "y_top": float(ys[k]),
            "height": float(heights[k]),
            "quality_flag": "ok",
        })

    result = pd.DataFrame(rows)
    geometry = [Point(r["x_top"], r["y_top"]) for _, r in result.iterrows()]
    gdf = gpd.GeoDataFrame(result, geometry=geometry, crs=crs)

    logger.info("Detected %d tree tops from nDSM (min_height=%.1f m)", len(gdf), min_height_m)
    return gdf


def match_tops_to_stems(
    stems: pd.DataFrame,
    detected_tops: pd.DataFrame | gpd.GeoDataFrame,
    max_distance_m: float = 3.0,
    one_to_one: bool = True,
) -> pd.DataFrame:
    """Match detected tree tops to ground-truth stem positions.

    Parameters
    ----------
    stems : pd.DataFrame
        Ground-truth stem positions with columns: tree_id, x, y, ground_truth_height.
    detected_tops : pd.DataFrame or gpd.GeoDataFrame
        Detected tops with columns: top_id, x_top, y_top, and height/z_top.
    max_distance_m : float
        Maximum matching distance (meters).
    one_to_one : bool
        If True, enforce unique stem-top pairs (closest stem wins).

    Returns
    -------
    pd.DataFrame
        One row per stem with matched top information:
        tree_id, x_stem, y_stem, ground_truth_height, top_id, x_top, y_top,
        stem_to_top_distance_m, match_quality_flag.
    """
    required_stem = {"tree_id", "x", "y", "ground_truth_height"}
    missing = required_stem - set(stems.columns)
    if missing:
        raise ValueError(f"stems missing columns: {missing}")

    result = stems[["tree_id", "x", "y", "ground_truth_height"]].copy()
    result = result.rename(columns={"x": "x_stem", "y": "y_stem"})

    if detected_tops.empty:
        result["top_id"] = np.nan
        result["x_top"] = np.nan
        result["y_top"] = np.nan
        result["stem_to_top_distance_m"] = np.nan
        result["match_quality_flag"] = "no_top_in_range"
        return result

    top_x_col = "x_top"
    top_y_col = "y_top"
    top_coords = detected_tops[[top_x_col, top_y_col]].to_numpy(dtype=float)
    stem_coords = result[["x_stem", "y_stem"]].to_numpy(dtype=float)

    tree = cKDTree(top_coords)
    distances, indices = tree.query(stem_coords, k=1)

    result["_nearest_top_idx"] = indices
    result["stem_to_top_distance_m"] = distances.astype(float)
    result["top_id"] = np.nan
    result["x_top"] = np.nan
    result["y_top"] = np.nan
    result["match_quality_flag"] = "no_top_in_range"

    within_range = result["stem_to_top_distance_m"] <= max_distance_m

    if one_to_one and within_range.any():
        matched_subset = result[within_range].sort_values("stem_to_top_distance_m")
        seen_tops = set()
        keep_indices = []

        for idx, row in matched_subset.iterrows():
            top_idx = int(row["_nearest_top_idx"])
            if top_idx not in seen_tops:
                seen_tops.add(top_idx)
                keep_indices.append(idx)
            else:
                result.at[idx, "match_quality_flag"] = "ambiguous_match"

        within_range = result.index.isin(keep_indices)

    for idx in result.index[within_range]:
        top_idx = int(result.at[idx, "_nearest_top_idx"])
        top_row = detected_tops.iloc[top_idx]
        result.at[idx, "top_id"] = top_row["top_id"]
        result.at[idx, "x_top"] = top_row[top_x_col]
        result.at[idx, "y_top"] = top_row[top_y_col]
        result.at[idx, "match_quality_flag"] = "matched"

    result = result.drop(columns=["_nearest_top_idx"])

    n_matched = (result["match_quality_flag"] == "matched").sum()
    logger.info(
        "Top matching: %d/%d stems matched (%.1f%%)",
        n_matched, len(result), 100.0 * n_matched / max(len(result), 1),
    )

    return result


def _non_maximum_suppression(
    coords: np.ndarray,
    heights: np.ndarray,
    suppression_radius: float,
) -> list[int]:
    """Remove redundant peaks based on spatial proximity, keeping tallest."""
    order = np.argsort(-heights)
    kept = []
    suppressed = set()

    if len(coords) == 0:
        return kept

    tree = cKDTree(coords)

    for idx in order:
        if idx in suppressed:
            continue
        kept.append(idx)
        neighbors = tree.query_ball_point(coords[idx], r=suppression_radius)
        for nb in neighbors:
            if nb != idx:
                suppressed.add(nb)

    return kept


def _empty_tops_df() -> pd.DataFrame:
    """Return empty DataFrame with standard tops schema."""
    return pd.DataFrame(
        columns=["top_id", "x_top", "y_top", "z_top",
                 "point_count", "point_density", "quality_flag"]
    )


def _empty_tops_gdf(crs=None) -> gpd.GeoDataFrame:
    """Return empty GeoDataFrame with standard tops schema."""
    return gpd.GeoDataFrame(
        columns=["top_id", "x_top", "y_top", "height", "quality_flag", "geometry"],
        geometry="geometry",
        crs=crs,
    )
