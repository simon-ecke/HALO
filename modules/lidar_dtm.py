"""LiDAR ground-point handling and DTM generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import laspy
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from scipy.interpolate import griddata

from .io_utils import get_logger


def _reproject_xy(
    x: np.ndarray,
    y: np.ndarray,
    source_crs: CRS,
    target_crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject planar point coordinates between CRS definitions."""
    if source_crs == target_crs:
        return x, y

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x_t, y_t = transformer.transform(x, y)
    return np.asarray(x_t, dtype=float), np.asarray(y_t, dtype=float)


def _pdal_is_available() -> bool:
    try:
        import pdal  # noqa: F401

        return True
    except Exception:
        return False


def read_lidar_crs(lidar_path: Path | str, assumed_crs: Optional[str] = None) -> CRS:
    """Read CRS from LAS/LAZ header, or use an assumed CRS if absent."""
    with laspy.open(lidar_path) as src:
        try:
            crs = src.header.parse_crs()
        except Exception:
            crs = None

    if crs is not None:
        return CRS.from_user_input(crs)

    if assumed_crs:
        return CRS.from_user_input(assumed_crs)

    raise ValueError(
        "LiDAR file has no CRS metadata. Set config.crs.lidar_assumed_crs explicitly."
    )


def _load_lidar_arrays(lidar_path: Path | str) -> dict[str, np.ndarray]:
    las = laspy.read(lidar_path)
    arrays: dict[str, np.ndarray] = {
        "x": np.asarray(las.x, dtype=float),
        "y": np.asarray(las.y, dtype=float),
        "z": np.asarray(las.z, dtype=float),
    }

    if "classification" in las.point_format.dimension_names:
        arrays["classification"] = np.asarray(las.classification, dtype=np.int16)

    return arrays


def _run_pdal_ground_filter(
    input_laz: Path,
    output_laz: Path,
    cloth_resolution: float = 0.5,
    rigidness: int = 2,
    class_threshold: float = 0.5,
    time_step: float = 0.65,
    smooth: bool = True,
) -> bool:
    """Run PDAL CSF (Cloth Simulation Filter) ground classification.

    CSF is more robust than SMRF on slopes and complex terrain, and widely
    used in forestry remote sensing. Parameters tuned for dense UAV-LiDAR.
    """
    import pdal

    pipeline = [
        {"type": "readers.las", "filename": str(input_laz)},
        {
            "type": "filters.csf",
            "step": cloth_resolution,
            "rigidness": rigidness,
            "threshold": class_threshold,
            "smooth": smooth,
        },
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {"type": "writers.las", "filename": str(output_laz)},
    ]

    pdal_pipeline = pdal.Pipeline(json.dumps(pipeline))
    count = pdal_pipeline.execute()
    return count > 0


def _ground_points_from_classification(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if classification is None:
        return np.array([]), np.array([]), np.array([])

    mask = classification == 2
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])

    return x[mask], y[mask], z[mask]


def _ground_points_grid_quantile(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size_m: float,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify likely ground points using per-cell lower quantiles."""
    x0 = float(x.min())
    y0 = float(y.min())
    ix = np.floor((x - x0) / cell_size_m).astype(np.int32)
    iy = np.floor((y - y0) / cell_size_m).astype(np.int32)

    df = pd.DataFrame({"ix": ix, "iy": iy, "x": x, "y": y, "z": z})

    grouped = (
        df.groupby(["ix", "iy"], sort=False)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            z=("z", lambda s: float(np.quantile(s.to_numpy(), quantile))),
        )
        .reset_index(drop=True)
    )

    return (
        grouped["x"].to_numpy(dtype=float),
        grouped["y"].to_numpy(dtype=float),
        grouped["z"].to_numpy(dtype=float),
    )


def _interpolate_dtm_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    resolution_m: float,
    interpolation_method: str = "linear",
) -> tuple[np.ndarray, rasterio.Affine]:
    minx, maxx = float(np.min(x)), float(np.max(x))
    miny, maxy = float(np.min(y)), float(np.max(y))

    grid_x = np.arange(minx, maxx + resolution_m, resolution_m)
    grid_y = np.arange(miny, maxy + resolution_m, resolution_m)
    xx, yy = np.meshgrid(grid_x, grid_y)

    dtm = griddata((x, y), z, (xx, yy), method=interpolation_method)

    if interpolation_method != "nearest" and np.isnan(dtm).any():
        nearest_fill = griddata((x, y), z, (xx, yy), method="nearest")
        nan_mask = np.isnan(dtm)
        dtm[nan_mask] = nearest_fill[nan_mask]

    dtm = np.flipud(dtm.astype(np.float32))
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    return dtm, transform


def create_lidar_dtm(
    lidar_path: Path | str,
    output_dtm_path: Path | str,
    resolution_m: float = 0.5,
    target_crs: Optional[str] = None,
    lidar_assumed_crs: Optional[str] = None,
    prefer_pdal: bool = True,
    fallback_grid_cell_size_m: float = 1.0,
    fallback_ground_quantile: float = 0.05,
    interpolation_method: str = "linear",
    csf_cloth_resolution: float = 0.5,
    csf_rigidness: int = 2,
    csf_class_threshold: float = 0.5,
    csf_time_step: float = 0.65,
    csf_smooth: bool = True,
) -> dict[str, object]:
    """Generate a DTM GeoTIFF from raw LiDAR.

    Priority for ground-point identification:
    1. PDAL CSF — Cloth Simulation Filter (if preferred and available)
    2. Existing LAS classification == 2
    3. Grid lower-quantile heuristic fallback
    """
    logger = get_logger("tree_height.lidar_dtm")
    lidar_path = Path(lidar_path)
    output_dtm_path = Path(output_dtm_path)
    output_dtm_path.parent.mkdir(parents=True, exist_ok=True)

    lidar_crs = read_lidar_crs(lidar_path, assumed_crs=lidar_assumed_crs)
    out_crs = CRS.from_user_input(target_crs) if target_crs else lidar_crs

    arrays = _load_lidar_arrays(lidar_path)
    x, y, z = arrays["x"], arrays["y"], arrays["z"]
    classification = arrays.get("classification")

    method_used = ""
    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray

    if prefer_pdal and _pdal_is_available():
        logger.info("Attempting PDAL-based CSF ground classification.")
        with tempfile.TemporaryDirectory(prefix="tree_height_pdal_") as tmp:
            ground_laz = Path(tmp) / "ground_only.laz"
            try:
                success = _run_pdal_ground_filter(
                    lidar_path,
                    ground_laz,
                    cloth_resolution=csf_cloth_resolution,
                    rigidness=csf_rigidness,
                    class_threshold=csf_class_threshold,
                    time_step=csf_time_step,
                )
                if success and ground_laz.exists():
                    ground_arrays = _load_lidar_arrays(ground_laz)
                    gx, gy, gz = (
                        ground_arrays["x"],
                        ground_arrays["y"],
                        ground_arrays["z"],
                    )
                    method_used = "pdal_csf"
                else:
                    gx, gy, gz = np.array([]), np.array([]), np.array([])
            except Exception as exc:
                logger.warning("PDAL ground classification failed: %s", exc)
                gx, gy, gz = np.array([]), np.array([]), np.array([])
    else:
        gx, gy, gz = np.array([]), np.array([]), np.array([])

    if gx.size == 0:
        gx, gy, gz = _ground_points_from_classification(x, y, z, classification)
        if gx.size > 0:
            method_used = "classification_class2"

    if gx.size == 0:
        logger.info("Falling back to grid-quantile ground estimation.")
        gx, gy, gz = _ground_points_grid_quantile(
            x=x,
            y=y,
            z=z,
            cell_size_m=fallback_grid_cell_size_m,
            quantile=fallback_ground_quantile,
        )
        method_used = "grid_quantile"

    if gx.size < 10:
        raise ValueError(
            "Insufficient ground points identified for DTM interpolation. "
            "Check LiDAR data quality or adjust ground extraction settings."
        )

    if lidar_crs != out_crs:
        logger.info(
            "Reprojecting ground points from %s to %s for DTM interpolation.",
            lidar_crs.to_string(),
            out_crs.to_string(),
        )
        gx, gy = _reproject_xy(gx, gy, source_crs=lidar_crs, target_crs=out_crs)

    dtm, transform = _interpolate_dtm_grid(
        x=gx,
        y=gy,
        z=gz,
        resolution_m=resolution_m,
        interpolation_method=interpolation_method,
    )

    nodata = -9999.0
    dtm_to_write = np.where(np.isnan(dtm), nodata, dtm).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": dtm_to_write.shape[0],
        "width": dtm_to_write.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": out_crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
        "tiled": True,
    }

    with rasterio.open(output_dtm_path, "w", **profile) as dst:
        dst.write(dtm_to_write, 1)

    summary = {
        "dtm_path": str(output_dtm_path),
        "ground_method": method_used,
        "ground_point_count": int(gx.size),
        "resolution_m": float(resolution_m),
        "crs": out_crs.to_string(),
        "pdal_available": _pdal_is_available(),
    }
    logger.info("LiDAR-derived DTM written to %s", output_dtm_path)
    return summary
