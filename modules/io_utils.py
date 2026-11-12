"""Input/output helpers and validation utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import laspy
import pandas as pd
from shapely.geometry import Point


def get_logger(name: str = "tree_height") -> logging.Logger:
    """Create a consistent logger for notebook and module usage."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def prepare_output_directories(output_root: Path | str) -> dict[str, Path]:
    """Create expected output folders and return their paths."""
    root = Path(output_root)
    paths = {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "rasters": root / "rasters",
    }
    for folder in paths.values():
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def assert_exists(path: Path | str, label: str) -> Path:
    """Raise an explicit error if a required path is missing."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _extract_z_from_point(point: Point) -> float:
    if point is None or point.is_empty:
        raise ValueError("Encountered empty geometry in ground-truth points.")
    if not point.has_z:
        raise ValueError(
            "Ground-truth point geometry does not contain Z. "
            "Provide ground_truth_height_field in config."
        )
    return float(point.z)


def load_ground_truth_points(
    path: Path | str,
    tree_id_field: str,
    height_field: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load ground-truth point data and return standardized columns.

    Output columns include:
    - tree_id
    - ground_truth_height
    - x
    - y
    - geometry
    """
    source = assert_exists(path, "Ground-truth vector")
    gdf = gpd.read_file(source)

    if gdf.empty:
        raise ValueError("Ground-truth vector is empty.")

    if tree_id_field not in gdf.columns:
        raise ValueError(f"tree_id field '{tree_id_field}' not found in {source}.")

    if not gdf.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
        raise ValueError("Ground-truth layer must contain point geometries.")

    # Convert MultiPoint to first point if needed for resilience.
    if (gdf.geometry.geom_type == "MultiPoint").any():
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.apply(
            lambda geom: list(geom.geoms)[0] if geom.geom_type == "MultiPoint" else geom
        )

    if height_field:
        if height_field not in gdf.columns:
            raise ValueError(f"Configured ground-truth height field '{height_field}' not found.")
        heights = pd.to_numeric(gdf[height_field], errors="coerce")
    else:
        heights = gdf.geometry.apply(_extract_z_from_point)

    out = gdf.copy()
    out["tree_id"] = out[tree_id_field].astype(str)
    out["ground_truth_height"] = pd.to_numeric(heights, errors="coerce")

    out = out.dropna(subset=["tree_id", "ground_truth_height"])
    if out.empty:
        raise ValueError("No valid tree records after applying tree_id and height filtering.")

    out["x"] = out.geometry.x
    out["y"] = out.geometry.y
    return out[["tree_id", "ground_truth_height", "x", "y", "geometry"]]


def read_lidar_overview(path: Path | str) -> dict[str, object]:
    """Return basic metadata from a LAS/LAZ header."""
    laz_path = assert_exists(path, "LiDAR point cloud")

    with laspy.open(laz_path) as src:
        header = src.header
        try:
            crs = header.parse_crs()
        except Exception:
            crs = None

        return {
            "path": str(laz_path),
            "point_count": int(header.point_count),
            "point_format": int(header.point_format.id),
            "version": str(header.version),
            "x_min": float(header.mins[0]),
            "y_min": float(header.mins[1]),
            "z_min": float(header.mins[2]),
            "x_max": float(header.maxs[0]),
            "y_max": float(header.maxs[1]),
            "z_max": float(header.maxs[2]),
            "crs": str(crs) if crs is not None else None,
        }


def export_dataframe_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Save a table to CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def export_geodataframe_gpkg(gdf: gpd.GeoDataFrame, path: Path | str, layer: str = "results") -> Path:
    """Save a GeoDataFrame to a GeoPackage layer."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, layer=layer, driver="GPKG")
    return out_path


def export_json(data: dict[str, object], path: Path | str) -> Path:
    """Save diagnostics to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
    return out_path
