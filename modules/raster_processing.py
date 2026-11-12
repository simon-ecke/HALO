"""Raster processing helpers: DTM diagnostics and nDSM generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .io_utils import get_logger


def create_ndsm(
    dsm_path: Path | str,
    dtm_path: Path | str,
    output_ndsm_path: Path | str,
    clip_negative_to_zero: bool = True,
) -> dict[str, object]:
    """Create nDSM by subtracting DTM from DSM on the DSM grid."""
    logger = get_logger("tree_height.raster_processing")

    dsm_path = Path(dsm_path)
    dtm_path = Path(dtm_path)
    output_ndsm_path = Path(output_ndsm_path)
    output_ndsm_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
        dsm = dsm_src.read(1).astype(np.float32)
        dsm_nodata = dsm_src.nodata

        dtm_on_dsm_grid = np.full(dsm.shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(dtm_src, 1),
            destination=dtm_on_dsm_grid,
            src_transform=dtm_src.transform,
            src_crs=dtm_src.crs,
            src_nodata=dtm_src.nodata,
            dst_transform=dsm_src.transform,
            dst_crs=dsm_src.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        if dsm_nodata is None:
            dsm_invalid = np.isnan(dsm)
        else:
            dsm_invalid = np.isclose(dsm, dsm_nodata) | np.isnan(dsm)

        invalid = dsm_invalid | np.isnan(dtm_on_dsm_grid)
        ndsm = dsm - dtm_on_dsm_grid

        if clip_negative_to_zero:
            ndsm = np.where(invalid, np.nan, np.maximum(ndsm, 0.0))
        else:
            ndsm = np.where(invalid, np.nan, ndsm)

        ndsm_nodata = -9999.0
        ndsm_out = np.where(np.isnan(ndsm), ndsm_nodata, ndsm).astype(np.float32)

        profile = dsm_src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=ndsm_nodata,
            compress="lzw",
            tiled=True,
        )

        with rasterio.open(output_ndsm_path, "w", **profile) as dst:
            dst.write(ndsm_out, 1)

    valid = np.isfinite(ndsm)
    summary = {
        "ndsm_path": str(output_ndsm_path),
        "valid_pixel_count": int(valid.sum()),
        "ndsm_min": float(np.nanmin(ndsm)) if valid.any() else float("nan"),
        "ndsm_max": float(np.nanmax(ndsm)) if valid.any() else float("nan"),
        "ndsm_mean": float(np.nanmean(ndsm)) if valid.any() else float("nan"),
    }

    logger.info("nDSM written to %s", output_ndsm_path)
    return summary


def raster_summary(raster_path: Path | str) -> dict[str, object]:
    """Return concise raster diagnostics for DTM decision support."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True)
        transform = src.transform
        pixel_size_x = abs(transform.a)
        pixel_size_y = abs(transform.e)

        valid_count = int(arr.count())
        total_count = int(arr.size)
        nodata_ratio = float(1 - (valid_count / total_count)) if total_count else 1.0

        values = arr.compressed()
        return {
            "path": str(raster_path),
            "crs": str(src.crs) if src.crs else None,
            "width": int(src.width),
            "height": int(src.height),
            "resolution_x": float(pixel_size_x),
            "resolution_y": float(pixel_size_y),
            "nodata": src.nodata,
            "nodata_ratio": nodata_ratio,
            "min": float(np.min(values)) if values.size else float("nan"),
            "max": float(np.max(values)) if values.size else float("nan"),
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "std": float(np.std(values)) if values.size else float("nan"),
            "bounds": {
                "left": float(src.bounds.left),
                "bottom": float(src.bounds.bottom),
                "right": float(src.bounds.right),
                "top": float(src.bounds.top),
            },
        }


def create_lidar_chm(
    normalized_df: pd.DataFrame,
    resolution_m: float,
    output_path: Path | str,
    crs: Optional[str] = None,
) -> Path:
    """Create a Canopy Height Model raster from normalized LiDAR points.

    Grids the point cloud and takes the maximum z_normalized per cell.
    """
    from rasterio.transform import from_origin

    logger = get_logger("tree_height.raster_processing")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pts = normalized_df[["x", "y", "z_normalized"]].dropna()
    if pts.empty:
        raise ValueError("No valid points for CHM generation.")

    xmin, xmax = float(pts["x"].min()), float(pts["x"].max())
    ymin, ymax = float(pts["y"].min()), float(pts["y"].max())

    cols = int(np.ceil((xmax - xmin) / resolution_m))
    rows = int(np.ceil((ymax - ymin) / resolution_m))

    col_idx = np.clip(((pts["x"].to_numpy() - xmin) / resolution_m).astype(int), 0, cols - 1)
    row_idx = np.clip(((ymax - pts["y"].to_numpy()) / resolution_m).astype(int), 0, rows - 1)

    chm = np.full((rows, cols), np.nan, dtype=np.float32)
    flat_idx = row_idx * cols + col_idx
    heights = pts["z_normalized"].to_numpy(dtype=np.float32)

    for i in range(len(flat_idx)):
        r, c = row_idx[i], col_idx[i]
        if np.isnan(chm[r, c]) or heights[i] > chm[r, c]:
            chm[r, c] = heights[i]

    transform = from_origin(xmin, ymax, resolution_m, resolution_m)
    nodata = -9999.0
    chm_out = np.where(np.isnan(chm), nodata, chm)

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
        "tiled": True,
    }
    if crs:
        profile["crs"] = crs

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(chm_out, 1)

    logger.info("LiDAR CHM written to %s (%d×%d px @ %.2f m)", output_path, cols, rows, resolution_m)
    return output_path


def compare_dtm_candidates(
    external_dtm_path: Optional[Path | str],
    lidar_dtm_path: Optional[Path | str],
) -> dict[str, object]:
    """Summarize available DTM candidates and their elevation differences."""
    records: list[dict[str, object]] = []

    if external_dtm_path:
        ext_summary = raster_summary(external_dtm_path)
        ext_summary["dtm_source"] = "external"
        records.append(ext_summary)

    if lidar_dtm_path:
        lidar_summary = raster_summary(lidar_dtm_path)
        lidar_summary["dtm_source"] = "lidar-derived"
        records.append(lidar_summary)

    diff_stats: dict[str, object] = {}

    if external_dtm_path and lidar_dtm_path:
        with rasterio.open(external_dtm_path) as ext_src, rasterio.open(lidar_dtm_path) as lidar_src:
            ext = ext_src.read(1).astype(np.float32)
            ext_nodata = ext_src.nodata

            lidar_resampled = np.full(ext.shape, np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(lidar_src, 1),
                destination=lidar_resampled,
                src_transform=lidar_src.transform,
                src_crs=lidar_src.crs,
                src_nodata=lidar_src.nodata,
                dst_transform=ext_src.transform,
                dst_crs=ext_src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

            if ext_nodata is None:
                ext_invalid = np.isnan(ext)
            else:
                ext_invalid = np.isclose(ext, ext_nodata) | np.isnan(ext)

            valid = (~ext_invalid) & np.isfinite(lidar_resampled)
            diff = lidar_resampled - ext
            diff_valid = diff[valid]

            if diff_valid.size > 0:
                diff_stats = {
                    "difference_name": "lidar_minus_external",
                    "count": int(diff_valid.size),
                    "mean": float(np.mean(diff_valid)),
                    "mae": float(np.mean(np.abs(diff_valid))),
                    "rmse": float(np.sqrt(np.mean(np.square(diff_valid)))),
                    "median": float(np.median(diff_valid)),
                    "std": float(np.std(diff_valid)),
                }

    return {
        "summary_table": pd.DataFrame.from_records(records),
        "difference_stats": diff_stats,
    }
