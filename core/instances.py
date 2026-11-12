"""LiDAR tree instances: clip the SegmentAnyTree clouds to the area of interest, ground model, height above
ground, harvest evidence, one row per instance, point density per plot.

Public functions used by other stages:
    load_before_cloud(cfg) -> DataFrame   clipped before-harvest cloud (x, y, z, instance, h, removed)
    chm_path(cfg), dtm_path(cfg), instances_path(cfg) -> Path
    run_instances(cfg)                    writes lidar_instances.gpkg and point_density_per_parzelle.csv

Intermediate files: <output_dir>/work/ (clipped clouds, ground candidates, removed flags, cache_info.json) and
<output_dir>/rasters/ (DTM and CHM). cache_info.json records the area-of-interest grid and the source files of
each step, so a clip or a flag array is reused only when it belongs to the same grid. The voxel grid origin of
the harvest-evidence test is the AOI corner, as in the notebook; a shifted grid would change removed fractions.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull
from shapely.geometry import Point

from . import cache
from .dtm import create_lidar_dtm
from .io_field import load_corners
from .pointcloud import (add_height_above_ground, clip_laz_to_bounds, load_instance_cloud, rasterize_max,
                         write_ground_candidates_laz, write_raster)

logger = logging.getLogger("core.instances")

Bounds = tuple[float, float, float, float]

_CLOUDS: dict[str, pd.DataFrame] = {}     # prepared before-harvest cloud, one per output folder and process


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def work_dir(cfg) -> Path:
    return cfg.output_dir / "work"


def rasters_dir(cfg) -> Path:
    return cfg.output_dir / "rasters"


def dtm_path(cfg) -> Path:
    return rasters_dir(cfg) / "dtm_lidar_after.tif"


def chm_path(cfg) -> Path:
    """CHM of the before-harvest cloud (maximum height above ground per cell, grid.chm_resolution_m)."""
    return rasters_dir(cfg) / "chm_lidar_before.tif"


def chm_after_path(cfg) -> Path:
    return rasters_dir(cfg) / "chm_lidar_after.tif"


def instances_path(cfg) -> Path:
    return cfg.table("lidar_instances.gpkg")


def _density_path(cfg) -> Path:
    return cfg.table("point_density_per_parzelle.csv")


def _sat_after(cfg):
    return cfg.paths.get("sat_after")


# ---------------------------------------------------------------------------
# Area of interest and cache keys
# ---------------------------------------------------------------------------

def aoi_bounds(cfg, corners: Optional[pd.DataFrame] = None) -> Bounds:
    """(xmin, ymin, xmax, ymax): bounding box of the plot corners plus grid.aoi_buffer_m."""
    corners = load_corners(cfg) if corners is None else corners
    b = float(cfg.grid.aoi_buffer_m)
    return (float(corners["UTM_x"].min() - b), float(corners["UTM_y"].min() - b),
            float(corners["UTM_x"].max() + b), float(corners["UTM_y"].max() + b))


def _sig(path) -> list:
    st = Path(path).stat()
    return [str(path), st.st_size, st.st_mtime_ns]


def _info_path(cfg) -> Path:
    return work_dir(cfg) / "cache_info.json"


def _read_info(cfg) -> dict:
    try:
        return json.loads(_info_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_info(cfg, info: dict) -> None:
    work_dir(cfg).mkdir(parents=True, exist_ok=True)
    _info_path(cfg).write_text(json.dumps(info, indent=1), encoding="utf-8")


def _keys(cfg, aoi: Bounds) -> dict:
    """Cache keys of every intermediate step (cheap: file stats and config values only)."""
    keys = {"aoi": list(aoi)}
    keys["clip_before"] = {"src": _sig(cfg.paths.sat_before.path), "aoi": list(aoi)}
    after = _sat_after(cfg)
    keys["clip_after"] = {"src": _sig(after.path), "aoi": list(aoi)} if after is not None else None
    ground = keys["clip_after"] or keys["clip_before"]
    keys["dtm"] = {"ground": ground, "resolution_m": cfg.grid.dtm_resolution_m, "crs": cfg.site.crs}
    keys["removed"] = ({"before": keys["clip_before"], "after": keys["clip_after"], "dtm": keys["dtm"],
                        "voxel_m": cfg.grid.removed_voxel_m} if after is not None else None)
    keys["instances"] = {"removed": keys["removed"], "dtm": keys["dtm"], "before": keys["clip_before"],
                         "chm_resolution_m": cfg.grid.chm_resolution_m, "crs": cfg.site.crs,
                         "instance_dim": cfg.paths.sat_before.instance_dim}
    return keys


# ---------------------------------------------------------------------------
# Harvest evidence and per-instance metrics (from modules/field_tree_matching.py)
# ---------------------------------------------------------------------------

def flag_removed_points(
    before: pd.DataFrame,
    after: pd.DataFrame,
    bounds: Bounds,
    voxel_m: float = 0.4,
    max_height_m: float = 60.0,
) -> np.ndarray:
    """Flag before-harvest points with no after-harvest point nearby.

    Both clouds are voxelized in (x, y, height above ground). A before point is
    'removed' if neither its voxel nor any of the 26 neighbours holds an after
    point, i.e. nothing within roughly 1-2 voxel sizes.
    """
    xmin, ymin, xmax, ymax = bounds
    shape = (
        int(np.ceil((xmax - xmin) / voxel_m)) + 1,
        int(np.ceil((ymax - ymin) / voxel_m)) + 1,
        int(np.ceil(max_height_m / voxel_m)) + 1,
    )

    def voxel_index(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h = df["h"].to_numpy()
        i = ((df["x"].to_numpy() - xmin) / voxel_m).astype(np.int64)
        j = ((df["y"].to_numpy() - ymin) / voxel_m).astype(np.int64)
        k = np.where(np.isfinite(h), h / voxel_m, -1).astype(np.int64)
        ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1]) & (k >= 0) & (k < shape[2])
        return i, j, k, ok

    ai, aj, ak, aok = voxel_index(after)
    occupied = np.zeros(shape, dtype=bool)
    occupied[ai[aok], aj[aok], ak[aok]] = True
    occupied = ndimage.binary_dilation(occupied, structure=np.ones((3, 3, 3), dtype=bool))

    bi, bj, bk, bok = voxel_index(before)
    removed = np.zeros(len(before), dtype=bool)
    removed[bok] = ~occupied[bi[bok], bj[bok], bk[bok]]
    return removed


def _crown_base_height(xy: np.ndarray, h: np.ndarray, min_spread_m: float = 1.0) -> float:
    """Rough crown base: lowest 1 m layer (above 2 m) from which the tree stays wide."""
    top = int(np.floor(h.max()))
    wide = []
    for lower in range(2, top):
        sel = (h >= lower) & (h < lower + 1)
        if sel.sum() < 10:
            wide.append(False)
            continue
        layer = xy[sel]
        spread = np.percentile(np.hypot(*(layer - np.median(layer, axis=0)).T), 90)
        wide.append(spread > min_spread_m)
    for idx in range(len(wide) - 1):
        if wide[idx] and wide[idx + 1]:
            return float(idx + 2)
    return float("nan")


def _base_position(
    xy: np.ndarray,
    h: np.ndarray,
    stem_slice_m: tuple[float, float] = (1.0, 3.0),
) -> tuple[np.ndarray, str, float]:
    """Plan-view stem base of one tree: median of the points in the stem slice.

    Falls back to the lowest 5 % of the points when the slice is (almost) empty.
    Returns (xy, source, spread in metres).
    """
    in_slice = (h >= stem_slice_m[0]) & (h <= stem_slice_m[1])
    if in_slice.sum() >= 10:
        base_pts, source = xy[in_slice], "stem_slice"
    else:
        n_low = max(10, int(0.05 * len(h)))
        base_pts, source = xy[np.argsort(h)[:n_low]], "lowest_points"
    base = np.median(base_pts, axis=0)
    return base, source, float(np.median(np.hypot(*(base_pts - base).T)))


def summarize_instances(
    points: pd.DataFrame,
    removed: Optional[np.ndarray] = None,
    crs: Optional[str] = None,
    min_height_m: float = 3.0,
    min_points: int = 100,
    stem_slice_m: tuple[float, float] = (1.0, 3.0),
    removed_min_height_m: float = 2.0,
):
    """One row per tree instance: top, base, height, crown metrics, harvest evidence.

    points: DataFrame with x, y, z, h (height above ground), instance. removed: boolean array aligned with
    `points`, or None (removed_fraction is then NaN). Returns a GeoDataFrame (geometry = base position)
    with columns instance, x_top, y_top, z_top, height, x_base, y_base, base_source, base_spread_m, lean_m,
    crown_base_m, crown_area_m2, n_points, removed_fraction.
    """
    import geopandas as gpd

    inst = points["instance"].to_numpy()
    order = np.argsort(inst, kind="stable")
    ids, starts, counts = np.unique(inst[order], return_index=True, return_counts=True)

    x_all = points["x"].to_numpy()
    y_all = points["y"].to_numpy()
    z_all = points["z"].to_numpy()
    h_all = points["h"].to_numpy()

    rows = []
    for tid, start, count in zip(ids, starts, counts):
        if tid <= 0 or count < min_points:
            continue
        idx = order[start: start + count]
        h = h_all[idx]
        finite = np.isfinite(h)
        if not finite.any():
            continue
        idx, h = idx[finite], h[finite]
        height = float(h.max())
        if height < min_height_m:
            continue

        xy = np.column_stack([x_all[idx], y_all[idx]])
        top = int(np.argmax(h))

        base, base_source, base_spread = _base_position(xy, h, stem_slice_m)

        crown = xy[h > 0.5 * height]
        if len(crown) > 20000:
            crown = crown[:: len(crown) // 20000]
        try:
            crown_area = float(ConvexHull(crown).volume) if len(crown) >= 4 else np.nan
        except Exception:
            crown_area = np.nan

        removed_fraction = np.nan
        if removed is not None:
            upper = h > removed_min_height_m
            if upper.any():
                removed_fraction = float(removed[idx][upper].mean())

        rows.append({
            "instance": int(tid),
            "x_top": float(xy[top, 0]),
            "y_top": float(xy[top, 1]),
            "z_top": float(z_all[idx][top]),
            "height": height,
            "x_base": float(base[0]),
            "y_base": float(base[1]),
            "base_source": base_source,
            "base_spread_m": base_spread,
            "lean_m": float(np.hypot(xy[top, 0] - base[0], xy[top, 1] - base[1])),
            "crown_base_m": _crown_base_height(xy, h),
            "crown_area_m2": crown_area,
            "n_points": int(len(h)),
            "removed_fraction": removed_fraction,
        })

    result = pd.DataFrame(rows)
    geometry = [Point(r.x_base, r.y_base) for r in result.itertuples()]
    logger.info("Instances: %d trees summarized (min_height=%.1f m)", len(result), min_height_m)
    return gpd.GeoDataFrame(result, geometry=geometry, crs=crs)


def point_density(clouds: dict, plots, bounds: Bounds) -> pd.DataFrame:
    """Points per m2 and share of empty 1 m cells inside each plot polygon, per cloud label.

    clouds: {label: DataFrame with x, y}; plots: GeoDataFrame with 'parzelle' and geometry.
    """
    import rasterio
    from rasterio.features import geometry_mask

    ncols, nrows = int(np.ceil(bounds[2] - bounds[0])), int(np.ceil(bounds[3] - bounds[1]))
    grid_transform = rasterio.transform.from_origin(bounds[0], bounds[3], 1.0, 1.0)
    density = {}
    for label, cloud in clouds.items():
        col = np.clip((cloud["x"].to_numpy() - bounds[0]).astype(np.int64), 0, ncols - 1)
        row = np.clip((bounds[3] - cloud["y"].to_numpy()).astype(np.int64), 0, nrows - 1)
        counts = np.bincount(row * ncols + col, minlength=nrows * ncols).reshape(nrows, ncols)
        masks = [geometry_mask([geom], (nrows, ncols), grid_transform, invert=True) for geom in plots.geometry]
        density[f"pts_m2_{label}"] = [float(counts[m].mean()) for m in masks]
        density[f"empty_cells_{label}"] = [float((counts[m] == 0).mean()) for m in masks]
    return pd.DataFrame({"parzelle": plots["parzelle"], **density})


# ---------------------------------------------------------------------------
# Clouds
# ---------------------------------------------------------------------------

def _ensure_clip(cfg, name: str, src, key: dict, aoi: Bounds, info: dict) -> Path:
    dst = work_dir(cfg) / f"sat_{name}_aoi.laz"
    if dst.exists() and info.get(f"clip_{name}") == key:
        return dst
    t0 = time.time()
    tmp = dst.with_name(dst.stem + "_partial.laz")        # so an interrupted clip is never mistaken for a full one
    result = clip_laz_to_bounds(src, tmp, aoi)
    tmp.replace(dst)
    info[f"clip_{name}"] = key
    _write_info(cfg, info)
    print(f"  clip {name}: {result['points_out']:,} of {result['points_in']:,} points in {time.time() - t0:.0f} s")
    return dst


def _prepare(cfg, need_after: bool):
    """Clip, ground model, height above ground, removed flags. Returns (before, after or None).

    `before` has columns x, y, z, instance, h and removed; it is kept for the rest of the process. `after`
    (x, y, z, instance, h) is loaded only when needed and is not kept.
    """
    aoi = aoi_bounds(cfg)
    keys = _keys(cfg, aoi)
    info = _read_info(cfg)
    if info.get("aoi") != keys["aoi"]:
        info = {"aoi": keys["aoi"]}                       # another grid: nothing of the old steps may be reused
    work_dir(cfg).mkdir(parents=True, exist_ok=True)

    sat_after = _sat_after(cfg)
    clip_before = _ensure_clip(cfg, "before", cfg.paths.sat_before.path, keys["clip_before"], aoi, info)
    clip_after = (_ensure_clip(cfg, "after", sat_after.path, keys["clip_after"], aoi, info)
                  if sat_after is not None else None)

    dtm_ok = dtm_path(cfg).exists() and info.get("dtm") == keys["dtm"]
    removed_file = work_dir(cfg) / "removed_before.npy"
    removed_ok = sat_after is None or (removed_file.exists() and info.get("removed") == keys["removed"])

    key = str(cfg.output_dir)
    before = _CLOUDS.get(key)
    if before is not None and dtm_ok and removed_ok and not need_after:
        return before, None

    if before is None:
        t0 = time.time()
        before = load_instance_cloud(clip_before, instance_dim=cfg.paths.sat_before.instance_dim)
        print(f"  loaded {len(before):,} points before the harvest ({time.time() - t0:.0f} s)")
    after = None
    if sat_after is not None and (need_after or not dtm_ok or not removed_ok):
        t0 = time.time()
        after = load_instance_cloud(clip_after, instance_dim=sat_after.instance_dim)
        print(f"  loaded {len(after):,} points after the harvest ({time.time() - t0:.0f} s)")

    if not dtm_ok:
        ground = after if after is not None else before   # the after-harvest cloud is preferred (less canopy)
        candidates = write_ground_candidates_laz(ground, work_dir(cfg) / "ground_candidates.laz", cell_size_m=0.25)
        dtm_info = create_lidar_dtm(candidates, dtm_path(cfg), resolution_m=cfg.grid.dtm_resolution_m,
                                    lidar_assumed_crs=cfg.site.crs)
        print(f"  DTM: {dtm_info['ground_method']}, {dtm_info['ground_point_count']:,} ground points")
        info["dtm"] = keys["dtm"]
        _write_info(cfg, info)

    if "h" not in before.columns:
        add_height_above_ground(before, dtm_path(cfg))
    if after is not None and "h" not in after.columns:
        add_height_above_ground(after, dtm_path(cfg))

    if "removed" not in before.columns or not removed_ok:
        if sat_after is None:
            before["removed"] = False                      # no harvest evidence without the after-harvest flight
        else:
            if removed_ok:
                flags = np.load(removed_file)
            else:
                t0 = time.time()
                flags = flag_removed_points(before, after, aoi, voxel_m=cfg.grid.removed_voxel_m)
                np.save(removed_file, flags)
                info["removed"] = keys["removed"]
                _write_info(cfg, info)
                print(f"  removed points: {flags.mean():.1%} of the before-harvest cloud ({time.time() - t0:.0f} s)")
            before["removed"] = flags
    _CLOUDS[key] = before
    return before, after


def load_before_cloud(cfg) -> pd.DataFrame:
    """Before-harvest cloud clipped to the area of interest, as a DataFrame.

    Columns: x, y, z (float64, float64, float32), instance (int32, SegmentAnyTree id, 0 = no tree),
    h (float32, height above ground from the DTM), removed (bool, gone after the harvest; all False when
    paths.sat_after is not configured). Loaded once per process; do not modify the returned frame.
    """
    before, _ = _prepare(cfg, need_after=False)
    return before


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def run_instances(cfg) -> None:
    """Write lidar_instances.gpkg, point_density_per_parzelle.csv and the CHM rasters."""
    from .coords import parzelle_polygons

    corners = load_corners(cfg)
    aoi = aoi_bounds(cfg, corners)
    keys = _keys(cfg, aoi)
    info = _read_info(cfg)
    outputs = [instances_path(cfg), _density_path(cfg), chm_path(cfg)]
    if (not getattr(cache, "_FORCE", False) and all(p.exists() for p in outputs)
            and info.get("aoi") == keys["aoi"] and info.get("instances") == keys["instances"]):
        print("  using cached instance table")
        return

    before, after = _prepare(cfg, need_after=True)
    removed = before["removed"].to_numpy() if _sat_after(cfg) is not None else None

    t0 = time.time()
    instances = summarize_instances(before, removed, crs=cfg.site.crs)
    print(f"  instances: {len(instances)} trees ({time.time() - t0:.0f} s); "
          f"{(before['instance'] > 0).mean():.1%} of points belong to a tree instance")
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    if instances_path(cfg).exists():
        instances_path(cfg).unlink()
    instances.to_file(instances_path(cfg), layer="instance_base", driver="GPKG")

    plots = parzelle_polygons(corners, crs=cfg.site.crs)
    clouds = {"before": before} if after is None else {"before": before, "after": after}
    point_density(clouds, plots, aoi).to_csv(_density_path(cfg), index=False)

    res = cfg.grid.chm_resolution_m
    chm_before, transform = rasterize_max(before["x"].to_numpy(), before["y"].to_numpy(), before["h"].to_numpy(), aoi, res)
    write_raster(chm_before, transform, chm_path(cfg), cfg.site.crs)
    if after is not None:
        chm_after, _ = rasterize_max(after["x"].to_numpy(), after["y"].to_numpy(), after["h"].to_numpy(), aoi, res)
        write_raster(chm_after, transform, chm_after_path(cfg), cfg.site.crs)

    (work_dir(cfg) / "run_info.json").write_text(json.dumps({
        "flight_before": Path(cfg.paths.sat_before.path).name,
        "flight_after": Path(_sat_after(cfg).path).name if _sat_after(cfg) is not None else None,
        "points_before": int(len(before)), "points_after": int(len(after)) if after is not None else None,
        "instances": int(len(instances)),
        "removed_share": round(float(removed.mean()), 4) if removed is not None else None}, indent=2))
    info = _read_info(cfg)
    info["instances"] = keys["instances"]
    _write_info(cfg, info)
