"""Match field-measured trees to LiDAR tree instances (SegmentAnyTree).

Tree identity is established from the before-harvest LiDAR instance segmentation:
a field stem is matched to the instance whose *base* is closest, using harvest
evidence (instance gone in the after-harvest cloud) as a second, height-independent
criterion. Heights are never used to choose a match, only to flag it.
"""

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

from .io_utils import get_logger

logger = get_logger("tree_height.field_tree_matching")

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


# ---------------------------------------------------------------------------
# Harvest evidence: which before-harvest points are gone afterwards
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


# ---------------------------------------------------------------------------
# Per-instance tree metrics
# ---------------------------------------------------------------------------

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
) -> gpd.GeoDataFrame:
    """One row per tree instance: top, base, height, crown metrics, harvest evidence.

    Parameters
    ----------
    points : DataFrame with x, y, z, h (height above ground), instance.
    removed : boolean array aligned with `points` (from flag_removed_points).
    stem_slice_m : height range used to locate the stem base in plan view.

    Returns
    -------
    GeoDataFrame (geometry = base position) with columns instance, x_top, y_top,
    z_top, height, x_base, y_base, base_source, base_spread_m, lean_m,
    crown_base_m, crown_area_m2, n_points, removed_fraction.
    """
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
        idx = order[start : start + count]
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


# ---------------------------------------------------------------------------
# Matching field stems to instances
# ---------------------------------------------------------------------------

def match_field_trees_to_instances(
    field: pd.DataFrame,
    instances: pd.DataFrame,
    max_distance_m: float = 3.0,
    removed_threshold: float = 0.5,
    removal_penalty_m: float = 2.0,
    high_confidence_distance_m: float = 1.5,
    height_gate_rel: float = 0.25,
) -> pd.DataFrame:
    """Assign each field stem to at most one LiDAR instance (global optimum).

    Cost of a pair = distance stem -> instance base, plus `removal_penalty_m` when
    the harvest evidence disagrees (tree recorded as felled in 2026 but instance
    still present afterwards, or vice versa). Field heights do NOT enter the cost;
    they only set `height_gate_ok`, which lowers the confidence class.

    Parameters
    ----------
    field : DataFrame with tree_id, utm_x, utm_y, removed_2026 and optionally
            height_vertex / height_massband.
    instances : output of summarize_instances().

    Returns
    -------
    DataFrame, one row per field tree: tree_id, instance, base_dist_m,
    runner_up_instance, runner_up_dist_m, n_candidates, removal_consistent,
    lidar_height, x_top, y_top, x_base, y_base, removed_fraction, lean_m,
    crown_base_lidar_m, field_height, height_gate_ok, confidence, method.
    """
    field = field.reset_index(drop=True)
    instances = instances.reset_index(drop=True)
    n_field, n_inst = len(field), len(instances)

    stem_xy = field[["utm_x", "utm_y"]].to_numpy(dtype=float)
    base_xy = instances[["x_base", "y_base"]].to_numpy(dtype=float)
    inst_removed = instances["removed_fraction"].to_numpy(dtype=float) > removed_threshold
    field_removed = field["removed_2026"].to_numpy(dtype=bool)

    forbidden = 1e6
    cost = np.full((n_field, n_inst + n_field), forbidden)
    # Dummy columns: leaving a tree unmatched costs as much as the worst allowed match
    cost[np.arange(n_field), n_inst + np.arange(n_field)] = max_distance_m + removal_penalty_m

    neighbours = cKDTree(base_xy).query_ball_point(stem_xy, r=max_distance_m)
    for i, cand in enumerate(neighbours):
        for j in cand:
            dist = float(np.hypot(*(stem_xy[i] - base_xy[j])))
            cost[i, j] = dist + (removal_penalty_m if inst_removed[j] != field_removed[i] else 0.0)

    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = {i: j for i, j in zip(row_ind, col_ind) if j < n_inst and cost[i, j] < forbidden}

    if "height_vertex" in field.columns:
        field_height = field["height_vertex"].fillna(field.get("height_massband")).to_numpy(dtype=float)
    else:
        field_height = np.full(n_field, np.nan)

    rows = []
    for i in range(n_field):
        cand = sorted(neighbours[i], key=lambda j: cost[i, j])
        row = {
            "tree_id": field.at[i, "tree_id"],
            "instance": pd.NA, "base_dist_m": np.nan,
            "runner_up_instance": pd.NA, "runner_up_dist_m": np.nan,
            "n_candidates": len(cand),
            "removal_consistent": pd.NA,
            "lidar_height": np.nan, "x_top": np.nan, "y_top": np.nan,
            "x_base": np.nan, "y_base": np.nan,
            "removed_fraction": np.nan, "lean_m": np.nan, "crown_base_lidar_m": np.nan,
            "field_height": field_height[i],
            "height_gate_ok": pd.NA,
            "confidence": "unmatched", "method": "none",
        }

        j = assigned.get(i)
        others = [c for c in cand if c != j]
        if others:
            row["runner_up_instance"] = int(instances.at[others[0], "instance"])
            row["runner_up_dist_m"] = float(np.hypot(*(stem_xy[i] - base_xy[others[0]])))

        if j is not None:
            inst = instances.loc[j]
            dist = float(np.hypot(*(stem_xy[i] - base_xy[j])))
            consistent = bool(inst_removed[j] == field_removed[i])
            gate = pd.NA
            if np.isfinite(field_height[i]):
                gate = bool(abs(inst["height"] - field_height[i]) <= height_gate_rel * field_height[i])

            row.update({
                "instance": int(inst["instance"]), "base_dist_m": dist,
                "removal_consistent": consistent,
                "lidar_height": float(inst["height"]),
                "x_top": float(inst["x_top"]), "y_top": float(inst["y_top"]),
                "x_base": float(inst["x_base"]), "y_base": float(inst["y_base"]),
                "removed_fraction": float(inst["removed_fraction"]),
                "lean_m": float(inst["lean_m"]),
                "crown_base_lidar_m": float(inst["crown_base_m"]),
                "height_gate_ok": gate,
                "method": "sat_instance",
            })
            is_high = dist <= high_confidence_distance_m and consistent and gate is not False
            row["confidence"] = "high" if is_high else "check"

        rows.append(row)

    result = pd.DataFrame(rows)
    logger.info("Matching: %s", result["confidence"].value_counts().to_dict())
    return result


def match_stems_with_attributes(
    field: pd.DataFrame,
    instances: pd.DataFrame,
    max_distance_m: float = 3.0,
    dbh_tolerance_mm: Optional[float] = None,
    dbh_penalty_m: float = 1.0,
    height_tolerance_m: Optional[float] = None,
    height_penalty_m: float = 1.0,
) -> pd.DataFrame:
    """One-to-one assignment of field stems to instance bases, optionally with DBH and height as tie-breakers.

    Cost of a pair = distance stem -> instance base
      + `dbh_penalty_m` if both DBHs are known and differ by more than `dbh_tolerance_mm`
      + `height_penalty_m` if both heights are known and differ by more than `height_tolerance_m`.
    A missing value on either side never costs anything. With both tolerances None this is
    the position-only assignment. The harvest record is not used.

    Parameters
    ----------
    field : DataFrame with tree_id, utm_x, utm_y and optionally dbh_mm, field_height_m.
    instances : DataFrame with instance, x_base, y_base and optionally dbh_lidar_mm, height.

    Returns
    -------
    DataFrame, one row per field tree: tree_id, instance, base_dist_m, cost, n_candidates,
    dbh_diff_mm and height_diff_m of the assigned pair (LiDAR minus field), dbh_penalised, height_penalised.
    """
    field = field.reset_index(drop=True)
    instances = instances.reset_index(drop=True)
    n_field, n_inst = len(field), len(instances)
    stem_xy = field[["utm_x", "utm_y"]].to_numpy(dtype=float)
    base_xy = instances[["x_base", "y_base"]].to_numpy(dtype=float)

    def column(table, name):
        return table[name].to_numpy(dtype=float) if name in table.columns else np.full(len(table), np.nan)

    field_dbh, inst_dbh = column(field, "dbh_mm"), column(instances, "dbh_lidar_mm")
    field_height, inst_height = column(field, "field_height_m"), column(instances, "height")

    forbidden = 1e6
    cost = np.full((n_field, n_inst + n_field), forbidden)
    cost[np.arange(n_field), n_inst + np.arange(n_field)] = max_distance_m      # leaving a tree unmatched
    pairs = {}
    neighbours = cKDTree(base_xy).query_ball_point(stem_xy, r=max_distance_m)
    for i, cand in enumerate(neighbours):
        for j in cand:
            dist = float(np.hypot(*(stem_xy[i] - base_xy[j])))
            dbh_diff, height_diff = inst_dbh[j] - field_dbh[i], inst_height[j] - field_height[i]
            dbh_bad = dbh_tolerance_mm is not None and np.isfinite(dbh_diff) and abs(dbh_diff) > dbh_tolerance_mm
            height_bad = height_tolerance_m is not None and np.isfinite(height_diff) and abs(height_diff) > height_tolerance_m
            cost[i, j] = dist + dbh_penalty_m * dbh_bad + height_penalty_m * height_bad
            pairs[(i, j)] = (dist, dbh_diff, height_diff, bool(dbh_bad), bool(height_bad))

    row_ind, col_ind = linear_sum_assignment(cost)
    rows = []
    for i, j in zip(row_ind, col_ind):
        row = {"tree_id": field.at[i, "tree_id"], "instance": pd.NA, "base_dist_m": np.nan, "cost": np.nan,
               "n_candidates": len(neighbours[i]), "dbh_diff_mm": np.nan, "height_diff_m": np.nan,
               "dbh_penalised": False, "height_penalised": False}
        if j < n_inst and cost[i, j] < max_distance_m:
            dist, dbh_diff, height_diff, dbh_bad, height_bad = pairs[(i, j)]
            row.update({"instance": int(instances.at[j, "instance"]), "base_dist_m": dist, "cost": float(cost[i, j]),
                        "dbh_diff_mm": dbh_diff, "height_diff_m": height_diff,
                        "dbh_penalised": dbh_bad, "height_penalised": height_bad})
        rows.append(row)
    result = pd.DataFrame(rows)
    result["instance"] = result["instance"].astype("Int64")
    return result


def apply_match_overrides(
    matches: pd.DataFrame,
    instances: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """Apply manual corrections: columns tree_id, instance (0 or empty = no LiDAR tree)."""
    matches = matches.copy()
    by_instance = instances.set_index("instance")

    for rec in overrides.itertuples(index=False):
        hit = matches.index[matches["tree_id"] == str(rec.tree_id)]
        if len(hit) == 0:
            logger.warning("Override for unknown tree_id %s ignored.", rec.tree_id)
            continue
        i = hit[0]

        if pd.isna(rec.instance) or int(rec.instance) <= 0:
            matches.loc[i, ["instance", "lidar_height", "x_top", "y_top", "x_base", "y_base"]] = [pd.NA] + [np.nan] * 5
            matches.loc[i, ["confidence", "method"]] = ["manual", "none"]
            continue

        inst_id = int(rec.instance)
        if inst_id not in by_instance.index:
            logger.warning("Override for tree %s: instance %d not found, ignored.", rec.tree_id, inst_id)
            continue
        inst = by_instance.loc[inst_id]

        # Free the instance if another tree currently holds it
        holder = matches.index[(matches["instance"] == inst_id) & (matches.index != i)]
        for h in holder:
            matches.loc[h, ["instance", "lidar_height", "x_top", "y_top", "x_base", "y_base"]] = [pd.NA] + [np.nan] * 5
            matches.loc[h, ["confidence", "method"]] = ["unmatched", "none"]

        matches.loc[i, "instance"] = inst_id
        for col, src in [("lidar_height", "height"), ("x_top", "x_top"), ("y_top", "y_top"),
                         ("x_base", "x_base"), ("y_base", "y_base"),
                         ("removed_fraction", "removed_fraction"), ("lean_m", "lean_m"),
                         ("crown_base_lidar_m", "crown_base_m")]:
            matches.loc[i, col] = float(inst[src])
        if "sat_height" in matches.columns:
            matches.loc[i, "sat_height"] = float(inst["height"])
        matches.loc[i, ["confidence", "method"]] = ["manual", "sat_instance"]

    return matches


# ---------------------------------------------------------------------------
# Manual inspection: clips for a 3D viewer, and hand-segmented trees
# ---------------------------------------------------------------------------

def export_inspection_clips(
    result: pd.DataFrame,
    before: pd.DataFrame,
    removed: np.ndarray,
    after: pd.DataFrame,
    before_laz: Path | str,
    after_laz: Path | str,
    output_dir: Path | str,
    tree_ids: Sequence[str],
    radius_m: float = 10.0,
) -> pd.DataFrame:
    """Write a cylinder of the before/after cloud around each tree for 3D inspection.

    `before` / `after` must be the tables loaded from `before_laz` / `after_laz`
    (same point order), so height above ground and the removed flag line up.
    All original dimensions are kept; added per point:
    height_above_ground, dist_to_stem (horizontal, m), and for the before cloud
    removed (0/1) and matched (1 = instance currently assigned to this tree).

    Returns one row per tree (also written as inspect_trees.csv) with the stem
    position, field heights, the assigned / runner-up instance and the instances
    that belong to neighbouring field trees.
    """
    import copy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trees = result.set_index("tree_id")

    def write_clip(las, table, idx, cx, cy, path, extra):
        out = laspy.LasData(copy.deepcopy(las.header))
        out.points = las.points[idx]
        dims = {
            "height_above_ground": table["h"].to_numpy()[idx].astype(np.float32),
            "dist_to_stem": np.hypot(table["x"].to_numpy()[idx] - cx, table["y"].to_numpy()[idx] - cy).astype(np.float32),
            **extra,
        }
        out.add_extra_dims([laspy.ExtraBytesParams(name=name, type=values.dtype) for name, values in dims.items()])
        for name, values in dims.items():
            out[name] = values
        out.write(path)

    def cylinder(table, cx, cy):
        x, y = table["x"].to_numpy(), table["y"].to_numpy()
        idx = np.flatnonzero((np.abs(x - cx) <= radius_m) & (np.abs(y - cy) <= radius_m))
        return idx[np.hypot(x[idx] - cx, y[idx] - cy) <= radius_m]

    las_before = laspy.read(before_laz)
    las_after = laspy.read(after_laz)
    if len(las_before.points) != len(before) or len(las_after.points) != len(after):
        raise ValueError("Point tables and LAZ files differ in length; they must come from the same clips.")

    rows = []
    for tree_id in tree_ids:
        tree = trees.loc[tree_id]
        cx, cy = float(tree["utm_x"]), float(tree["utm_y"])

        idx_b = cylinder(before, cx, cy)
        matched = np.zeros(len(idx_b), dtype=np.uint8)
        if pd.notna(tree.get("instance")):
            matched = (before["instance"].to_numpy()[idx_b] == int(tree["instance"])).astype(np.uint8)
        write_clip(las_before, before, idx_b, cx, cy, output_dir / f"tree_{tree_id}_before.laz",
                   {"removed": removed[idx_b].astype(np.uint8), "matched": matched})

        idx_a = cylinder(after, cx, cy)
        write_clip(las_after, after, idx_a, cx, cy, output_dir / f"tree_{tree_id}_after.laz", {})

        others = result[(result["tree_id"] != tree_id) & result["instance"].notna()]
        others = others.assign(stem_dist=np.hypot(others["utm_x"] - cx, others["utm_y"] - cy))
        others = others[others["stem_dist"] <= radius_m].sort_values("stem_dist")
        neighbours = "; ".join(
            f"{int(r.instance)}={r.tree_id} ({r.stem_dist:.1f} m away, {r.sat_height:.1f} m tall)"
            for r in others.itertuples()
        )

        rows.append({
            "tree_id": tree_id, "utm_x": cx, "utm_y": cy,
            "height_vertex": tree.get("height_vertex"), "height_massband": tree.get("height_massband"),
            "dbh_mm_2025": tree.get("dbh_mm_2025"),
            "confidence": tree.get("confidence"), "method": tree.get("method"),
            "instance": tree.get("instance"), "runner_up_instance": tree.get("runner_up_instance"),
            "lidar_height": tree.get("lidar_height"), "sat_height": tree.get("sat_height"),
            "points_before": len(idx_b), "points_after": len(idx_a),
            "neighbour_instances": neighbours,
        })

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "inspect_trees.csv", index=False)
    logger.info("Inspection clips: %d trees, radius %.1f m -> %s", len(table), radius_m, output_dir)
    return table


def load_manual_segments(
    folder: Path | str,
    dtm_path: Path | str,
    stem_slice_m: tuple[float, float] = (1.0, 3.0),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Read hand-segmented trees saved as tree_<Parzelle>_<Nr>.laz / .las.

    Heights are recomputed from z and the DTM, so the files only need x, y, z.

    Returns
    -------
    (summary, points): summary has one row per tree (tree_id, lidar_height,
    x_top, y_top, x_base, y_base, lean_m, n_points, file); points maps
    tree_id -> DataFrame x, y, h for plotting.
    """
    import re

    columns = ["tree_id", "lidar_height", "x_top", "y_top", "x_base", "y_base", "lean_m", "n_points", "file"]
    folder = Path(folder)
    rows, points = [], {}
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".laz", ".las"}) if folder.exists() else []

    for path in files:
        found = re.search(r"tree_(\d+_\d+)", path.stem)
        if not found:
            logger.warning("Manual segment %s ignored: name must contain tree_<Parzelle>_<Nr>.", path.name)
            continue
        tree_id = found.group(1)
        if tree_id in points:
            logger.warning("Manual segment %s ignored: tree %s already loaded.", path.name, tree_id)
            continue

        las = laspy.read(path)
        x, y = np.asarray(las.x, dtype=np.float64), np.asarray(las.y, dtype=np.float64)
        h = np.asarray(las.z, dtype=np.float64) - sample_raster_bilinear(dtm_path, x, y)
        ok = np.isfinite(h)
        if not ok.any():
            logger.warning("Manual segment %s ignored: no points inside the DTM.", path.name)
            continue
        x, y, h = x[ok], y[ok], h[ok]

        top = int(np.argmax(h))
        base, _, _ = _base_position(np.column_stack([x, y]), h, stem_slice_m)
        rows.append({
            "tree_id": tree_id, "lidar_height": float(h[top]),
            "x_top": float(x[top]), "y_top": float(y[top]),
            "x_base": float(base[0]), "y_base": float(base[1]),
            "lean_m": float(np.hypot(x[top] - base[0], y[top] - base[1])),
            "n_points": int(len(h)), "file": path.name,
        })
        points[tree_id] = pd.DataFrame({"x": x, "y": y, "h": h})

    return pd.DataFrame(rows, columns=columns), points


def apply_manual_segments(matches: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Use hand-segmented trees as the LiDAR result (method 'manual_segmentation').

    These are reference segments made by a person, not a SegmentAnyTree result;
    the automatic outcome stays available in the auto_* / sat_height columns.
    """
    matches = matches.copy()
    for rec in manual.itertuples(index=False):
        hit = matches.index[matches["tree_id"] == rec.tree_id]
        if len(hit) == 0:
            logger.warning("Manual segment for unknown tree_id %s ignored.", rec.tree_id)
            continue
        i = hit[0]
        for col in ["lidar_height", "x_top", "y_top", "x_base", "y_base", "lean_m"]:
            matches.loc[i, col] = getattr(rec, col)
        matches.loc[i, ["method", "confidence"]] = ["manual_segmentation", "manual"]
    return matches


# ---------------------------------------------------------------------------
# Fallback for suppressed trees: stem-anchored column profile
# ---------------------------------------------------------------------------

def column_profile_top(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    centre: tuple[float, float],
    radius_m: float = 1.0,
    max_gap_m: float = 1.5,
    min_height_m: float = 2.0,
    min_points: int = 50,
) -> Optional[dict[str, float]]:
    """Top of the lowest continuous point layer above a stem.

    For a tree standing under a taller canopy the layer ends at its own top,
    because the air between the two crowns shows up as a gap in the heights.
    Points below `min_height_m` (ground vegetation) are ignored. Returns None
    if fewer than `min_points` remain near the stem.
    """
    cx, cy = centre
    near = (np.abs(x - cx) <= radius_m) & (np.abs(y - cy) <= radius_m)
    near[near] = np.hypot(x[near] - cx, y[near] - cy) <= radius_m
    near[near] = np.isfinite(h[near]) & (h[near] >= min_height_m)
    if near.sum() < min_points:
        return None

    idx = np.flatnonzero(near)
    order = idx[np.argsort(h[idx])]
    heights = h[order]

    gaps = np.flatnonzero(np.diff(heights) > max_gap_m)
    last = gaps[0] if len(gaps) else len(heights) - 1
    return {
        "height": float(heights[last]),
        "x_top": float(x[order[last]]),
        "y_top": float(y[order[last]]),
        "n_points": int(last + 1),
        "hit_gap": bool(len(gaps) > 0),
    }


def add_height_gate(matches: pd.DataFrame, field: pd.DataFrame, height_gate_rel: float = 0.25) -> pd.DataFrame:
    """Set `height_gate_ok` on a matching that was made without field heights.

    The gate is a flag, not a selection: a matched tree whose LiDAR height differs
    from the field height (Vertex, else Maßband) by more than `height_gate_rel` of
    it drops from 'high' to 'check'. Trees without a field height keep NA.
    """
    matches = matches.copy()
    heights = field.set_index("tree_id")
    field_height = matches["tree_id"].map(heights["height_vertex"].fillna(heights["height_massband"])).to_numpy(dtype=float)
    lidar = matches["lidar_height"].to_numpy(dtype=float)
    known = np.isfinite(field_height) & np.isfinite(lidar)
    gate = pd.array([pd.NA] * len(matches), dtype="boolean")
    gate[known] = np.abs(lidar[known] - field_height[known]) <= height_gate_rel * field_height[known]
    matches["field_height"] = field_height
    matches["height_gate_ok"] = gate
    failed = matches["height_gate_ok"].isin([False]) & (matches["confidence"] == "high")
    matches.loc[failed, "confidence"] = "check"
    return matches


def add_column_profile_fallback(
    matches: pd.DataFrame,
    field: pd.DataFrame,
    points: pd.DataFrame,
    removed: np.ndarray,
    radius_m: float = 1.0,
    max_gap_m: float = 1.5,
) -> pd.DataFrame:
    """Column-profile height for trees that are unmatched or failed the height gate.

    Only points that are still unexplained are used: points of instances assigned
    to another field tree are excluded, and for trees felled in 2026 only removed
    points count. If a layer is found, `lidar_height` and the top position switch
    to it and `method` becomes 'column_profile' (confidence 'check').

    A tree that failed the height gate and has no such layer cannot be separated
    from its neighbours: it becomes 'unresolved' with no LiDAR height. The SAT
    result is always kept in `sat_height`.
    """
    matches = matches.copy()
    matches["sat_height"] = matches["lidar_height"]
    stems = field.set_index("tree_id")

    x, y, h = points["x"].to_numpy(), points["y"].to_numpy(), points["h"].to_numpy()
    inst = points["instance"].to_numpy()
    needs = (matches["confidence"] == "unmatched") | (matches["height_gate_ok"].isin([False]))
    needs &= matches["confidence"] != "manual"  # manual overrides are final

    for i in matches.index[needs]:
        tree_id = matches.at[i, "tree_id"]
        stem = stems.loc[tree_id]
        cx, cy = float(stem["utm_x"]), float(stem["utm_y"])

        owned_by_others = matches.loc[matches.index != i, "instance"].dropna().astype(int).to_numpy()
        box = np.flatnonzero((np.abs(x - cx) <= radius_m) & (np.abs(y - cy) <= radius_m))
        keep = ~np.isin(inst[box], owned_by_others)
        if stem["removed_2026"]:
            keep &= removed[box]
        box = box[keep]
        profile = column_profile_top(x[box], y[box], h[box], (cx, cy), radius_m, max_gap_m)

        if profile is not None:
            matches.loc[i, ["lidar_height", "x_top", "y_top"]] = [profile["height"], profile["x_top"], profile["y_top"]]
            matches.loc[i, ["x_base", "y_base"]] = [cx, cy]
            matches.loc[i, ["method", "confidence"]] = ["column_profile", "check"]
        elif matches.at[i, "confidence"] != "unmatched":
            matches.loc[i, ["lidar_height", "x_top", "y_top"]] = np.nan
            matches.loc[i, ["method", "confidence"]] = ["none", "unresolved"]

    return matches


# ---------------------------------------------------------------------------
# Visibility from above (is the top part of the outer canopy surface?)
# ---------------------------------------------------------------------------

def visible_from_above(
    chm: np.ndarray,
    transform: rasterio.Affine,
    x_top: np.ndarray,
    y_top: np.ndarray,
    h_top: np.ndarray,
    radius_m: float = 1.0,
    tolerance_m: float = 1.0,
) -> np.ndarray:
    """True where a tree top lies within `tolerance_m` of the canopy surface around it.

    A top well below the highest returns within `radius_m` is overtopped and cannot
    be seen in a photogrammetric surface model. NaN tops give False.
    """
    res = transform.a
    r_px = max(1, int(round(radius_m / res)))
    yy, xx = np.ogrid[-r_px : r_px + 1, -r_px : r_px + 1]
    footprint = (xx**2 + yy**2) <= r_px**2
    surface = ndimage.maximum_filter(np.where(np.isfinite(chm), chm, -np.inf), footprint=footprint)

    cols, rows = ~transform * (np.asarray(x_top, dtype=float), np.asarray(y_top, dtype=float))
    ok = np.isfinite(cols) & np.isfinite(rows)
    rows_i = np.clip(np.where(ok, rows, 0).astype(int), 0, chm.shape[0] - 1)
    cols_i = np.clip(np.where(ok, cols, 0).astype(int), 0, chm.shape[1] - 1)
    above = surface[rows_i, cols_i] - np.asarray(h_top, dtype=float)
    return ok & np.isfinite(above) & (above <= tolerance_m)


def open_ground_offset(
    chm: np.ndarray,
    chm_transform: rasterio.Affine,
    ndsm_path: Path | str,
    radius_m: float = 1.5,
    max_canopy_m: float = 0.3,
) -> dict[str, float]:
    """Vertical offset of a photogrammetric nDSM against the LiDAR ground, on open ground.

    Open ground = LiDAR CHM below `max_canopy_m` everywhere within `radius_m`. Any
    offset there appears one-to-one as bias in the photogrammetry heights.
    """
    r_px = int(round(radius_m / chm_transform.a))
    surface = ndimage.maximum_filter(np.where(np.isfinite(chm), chm, np.inf), size=2 * r_px + 1)
    rows, cols = np.where(surface < max_canopy_m)
    xs, ys = rasterio.transform.xy(chm_transform, rows, cols, offset="center")
    ground = sample_raster_bilinear(ndsm_path, np.asarray(xs), np.asarray(ys))
    ground = ground[np.isfinite(ground)]
    if len(ground) == 0:
        return {"open_ground_cells": 0, "median_m": np.nan, "p05_m": np.nan, "p95_m": np.nan}
    return {"open_ground_cells": len(ground), "median_m": np.median(ground),
            "p05_m": np.percentile(ground, 5), "p95_m": np.percentile(ground, 95)}


# ---------------------------------------------------------------------------
# GIS layers
# ---------------------------------------------------------------------------

def build_match_layers(
    field: pd.DataFrame,
    matches: pd.DataFrame,
    crs: str,
    extra_columns: Sequence[str] = (),
) -> dict[str, gpd.GeoDataFrame]:
    """Stems, tops and stem->top lines as GeoDataFrames for checking in GIS."""
    table = field.merge(matches, on="tree_id", how="left", suffixes=("", "_match"))
    keep = [c for c in table.columns if c in set(matches.columns) | set(extra_columns) | {
        "tree_id", "parzelle", "nr", "utm_x", "utm_y", "dbh_mm_2025", "removed_2026",
        "height_vertex", "height_massband", "crown_base_m", "has_height"}]
    table = table[keep]

    stems = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table["utm_x"], table["utm_y"]), crs=crs)

    has_top = table["x_top"].notna()
    tops = gpd.GeoDataFrame(
        table[has_top], geometry=gpd.points_from_xy(table.loc[has_top, "x_top"], table.loc[has_top, "y_top"]), crs=crs
    )
    lines = gpd.GeoDataFrame(
        table[has_top],
        geometry=[LineString([(r.utm_x, r.utm_y), (r.x_top, r.y_top)]) for r in table[has_top].itertuples()],
        crs=crs,
    )
    return {"stems": stems, "tops": tops, "stem_to_top": lines}


# ---------------------------------------------------------------------------
# Stem diameter at breast height (DBH) from a point slice
# ---------------------------------------------------------------------------

def _circles_from_triples(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Circumscribed circles of many point triples at once. Returns (n, 3): cx, cy, r (NaN if collinear)."""
    ax, ay = p1[:, 0], p1[:, 1]
    bx, by = p2[:, 0], p2[:, 1]
    cx, cy = p3[:, 0], p3[:, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = np.hypot(ax - ux, ay - uy)
    bad = np.abs(d) < 1e-12
    return np.column_stack([np.where(bad, np.nan, ux), np.where(bad, np.nan, uy), np.where(bad, np.nan, r)])


def fit_circle_ransac(
    xy: np.ndarray,
    centre_hint: Optional[tuple[float, float]] = None,
    n_iter: int = 1000,
    tolerance_m: float = 0.025,
    radius_range_m: tuple[float, float] = (0.03, 0.45),
    max_centre_offset_m: float = 0.75,
    interior_weight: float = 2.0,
    seed: int = 0,
) -> Optional[dict[str, float]]:
    """Robust circle fit: RANSAC on point triples, then least-squares refinement on the inliers.

    RANSAC keeps the circle that most points lie on (within `tolerance_m`), so
    branches, needles and stray points do not pull the fit. Because a stem is
    opaque, a correct circle has no points inside it: each point lying more than
    `tolerance_m` inside a candidate circle costs `interior_weight` inliers. This
    stops large circles that enclose the real stem. The refinement minimizes the
    geometric distance of the inliers to the circle. Returns None if no plausible
    circle is found.
    """
    from scipy.optimize import least_squares

    n = len(xy)
    if n < 5:
        return None
    rng = np.random.default_rng(seed)
    triples = np.array([rng.choice(n, 3, replace=False) for _ in range(n_iter)])
    circles = _circles_from_triples(xy[triples[:, 0]], xy[triples[:, 1]], xy[triples[:, 2]])

    ok = np.isfinite(circles[:, 2]) & (circles[:, 2] >= radius_range_m[0]) & (circles[:, 2] <= radius_range_m[1])
    if centre_hint is not None:
        ok &= np.hypot(circles[:, 0] - centre_hint[0], circles[:, 1] - centre_hint[1]) <= max_centre_offset_m
    circles = circles[ok]
    if len(circles) == 0:
        return None

    # Score every candidate circle (candidates x points): points on it minus penalty for points inside it
    signed = np.hypot(xy[None, :, 0] - circles[:, 0, None], xy[None, :, 1] - circles[:, 1, None]) - circles[:, 2, None]
    score = (np.abs(signed) <= tolerance_m).sum(axis=1) - interior_weight * (signed < -tolerance_m).sum(axis=1)
    best = int(np.argmax(score))
    params = circles[best]

    for _ in range(2):  # refine, re-select inliers, refine again
        inliers = np.abs(np.hypot(xy[:, 0] - params[0], xy[:, 1] - params[1]) - params[2]) <= tolerance_m
        if inliers.sum() < 5:
            return None
        pts = xy[inliers]
        fit = least_squares(lambda p: np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) - p[2], params)
        params = fit.x

    if not (radius_range_m[0] <= params[2] <= radius_range_m[1]):
        return None
    dist = np.hypot(xy[:, 0] - params[0], xy[:, 1] - params[1]) - params[2]
    inliers = np.abs(dist) <= tolerance_m
    angles = np.arctan2(xy[inliers, 1] - params[1], xy[inliers, 0] - params[0])
    sectors = np.unique(np.floor((angles + np.pi) / (2 * np.pi) * 36).astype(int) % 36)

    return {
        "cx": float(params[0]), "cy": float(params[1]), "radius_m": float(params[2]),
        "n_inliers": int(inliers.sum()),
        "n_interior": int((dist < -tolerance_m).sum()),
        "rmse_m": float(np.sqrt(np.mean(dist[inliers] ** 2))),
        "arc_coverage": float(len(sectors) / 36.0),
    }


def fit_ellipse(
    xy: np.ndarray,
    circle: dict[str, float],
    band_m: float = 0.05,
    tolerance_m: float = 0.025,
    max_axis_ratio: float = 1.5,
) -> Optional[dict[str, float]]:
    """Ellipse fit started from a circle fit (see fit_circle_ransac).

    Uses the points within `band_m` of the circle, so an oval stem's points are not
    cut off, and a robust least-squares fit (soft-L1 loss) of centre, size,
    axis ratio and orientation. The axis ratio is limited to `max_axis_ratio`,
    because on a partial arc an unconstrained ellipse can become arbitrarily
    elongated. Equivalent diameters:
    - perimeter-equivalent = ellipse circumference / pi (what a girth tape measures)
    - area-equivalent = 2 * sqrt(a * b)
    """
    from scipy.optimize import least_squares

    cx, cy, r = circle["cx"], circle["cy"], circle["radius_m"]
    near = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r) <= band_m
    # Work relative to the circle centre: with UTM-sized coordinates the numerical
    # derivatives of the optimizer break down and the fit never leaves its start.
    pts = xy[near] - np.array([cx, cy])
    if len(pts) < 8:
        return None

    def residuals(p, pts):
        x0, y0, rm, k, theta = p
        a, b = rm * np.exp(k / 2), rm * np.exp(-k / 2)
        c, s = np.cos(theta), np.sin(theta)
        u = c * (pts[:, 0] - x0) + s * (pts[:, 1] - y0)
        v = -s * (pts[:, 0] - x0) + c * (pts[:, 1] - y0)
        t = np.arctan2(v / b, u / a)
        return np.hypot(u - a * np.cos(t), v - b * np.sin(t))

    k_max = np.log(max_axis_ratio)
    lower = [-0.5, -0.5, 0.02, -k_max, -np.pi]
    upper = [0.5, 0.5, 0.6, k_max, np.pi]
    params = np.array([0.0, 0.0, r, 0.05, 0.0])
    for _ in range(2):  # robust fit, keep the points close to the ellipse, fit again
        fit = least_squares(residuals, params, args=(pts,), bounds=(lower, upper), loss="soft_l1",
                            f_scale=0.01, x_scale=[0.05, 0.05, 0.05, 0.1, 0.5])
        params = fit.x
        close = residuals(params, pts) <= tolerance_m
        if close.sum() < 8:
            return None
        pts = pts[close]

    x0, y0, rm, k, theta = params
    x0, y0 = x0 + cx, y0 + cy
    a, b = rm * np.exp(abs(k) / 2), rm * np.exp(-abs(k) / 2)
    h = ((a - b) / (a + b)) ** 2
    perimeter = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))  # Ramanujan
    res = residuals(params, pts)
    return {
        "cx": float(x0), "cy": float(y0), "a_m": float(a), "b_m": float(b), "theta": float(theta),
        "axis_ratio": float(a / b),
        "d_perimeter_m": float(perimeter / np.pi),
        "d_area_m": float(2 * np.sqrt(a * b)),
        "n_inliers": int(len(pts)),
        "rmse_m": float(np.sqrt(np.mean(res**2))),
    }


def estimate_dbh(
    points: pd.DataFrame,
    centres: pd.DataFrame,
    breast_height_m: float = 1.3,
    slice_half_height_m: float = 0.25,
    search_radius_m: float = 1.0,
    min_inliers: int = 15,
    min_arc_coverage: float = 0.25,
    max_rmse_m: float = 0.02,
    max_interior_share: float = 0.2,
    return_slices: bool = False,
    **ransac_kwargs,
):
    """LiDAR DBH per tree from a thin point slice at breast height.

    Parameters
    ----------
    points : DataFrame with x, y, h (height above ground).
    centres : DataFrame with tree_id, x, y: where to look for the stem.

    A fit is 'ok' if it has at least `min_inliers` points on the circle, covers
    at least `min_arc_coverage` of the circumference, has a residual RMSE of at
    most `max_rmse_m`, and has few points inside the circle (at most
    `max_interior_share` of the inliers). Otherwise 'weak'; 'no_fit' if no circle
    was found.

    For every circle fit an ellipse is fitted as well (fit_ellipse); its
    perimeter-equivalent diameter is in dbh_ellipse_mm.

    Returns
    -------
    DataFrame with tree_id, dbh_lidar_mm, x_stem, y_stem, n_slice, n_inliers,
    n_interior, rmse_mm, arc_coverage, dbh_quality ('ok' / 'weak' / 'no_fit'),
    dbh_ellipse_mm, dbh_ellipse_area_mm, ellipse_a_mm, ellipse_b_mm,
    ellipse_axis_ratio, ellipse_theta, ellipse_cx, ellipse_cy, ellipse_rmse_mm;
    with return_slices=True also a dict tree_id -> slice points (N x 2 array).
    """
    h = points["h"].to_numpy()
    in_slice = np.flatnonzero(np.abs(h - breast_height_m) <= slice_half_height_m)
    slice_xy = np.column_stack([points["x"].to_numpy()[in_slice], points["y"].to_numpy()[in_slice]])
    index = cKDTree(slice_xy)

    rows, slices = [], {}
    for rec in centres.itertuples(index=False):
        near = index.query_ball_point([rec.x, rec.y], r=search_radius_m)
        xy = slice_xy[near]
        if return_slices:
            slices[rec.tree_id] = xy
        fit = fit_circle_ransac(xy, centre_hint=(rec.x, rec.y), **ransac_kwargs) if len(xy) else None

        row = {"tree_id": rec.tree_id, "dbh_lidar_mm": np.nan, "x_stem": np.nan, "y_stem": np.nan,
               "n_slice": len(xy), "n_inliers": 0, "n_interior": 0, "rmse_mm": np.nan, "arc_coverage": 0.0,
               "dbh_quality": "no_fit"}
        if fit is not None:
            good = (fit["n_inliers"] >= min_inliers and fit["arc_coverage"] >= min_arc_coverage
                    and fit["rmse_m"] <= max_rmse_m and fit["n_interior"] <= max_interior_share * fit["n_inliers"])
            row.update({
                "dbh_lidar_mm": 2000.0 * fit["radius_m"], "x_stem": fit["cx"], "y_stem": fit["cy"],
                "n_inliers": fit["n_inliers"], "n_interior": fit["n_interior"],
                "rmse_mm": 1000.0 * fit["rmse_m"], "arc_coverage": fit["arc_coverage"],
                "dbh_quality": "ok" if good else "weak",
            })
            ellipse = fit_ellipse(xy, fit, tolerance_m=ransac_kwargs.get("tolerance_m", 0.025))
            if ellipse is not None:
                row.update({
                    "dbh_ellipse_mm": 1000.0 * ellipse["d_perimeter_m"],
                    "dbh_ellipse_area_mm": 1000.0 * ellipse["d_area_m"],
                    "ellipse_a_mm": 1000.0 * ellipse["a_m"], "ellipse_b_mm": 1000.0 * ellipse["b_m"],
                    "ellipse_axis_ratio": ellipse["axis_ratio"], "ellipse_theta": ellipse["theta"],
                    "ellipse_cx": ellipse["cx"], "ellipse_cy": ellipse["cy"],
                    "ellipse_rmse_mm": 1000.0 * ellipse["rmse_m"],
                })
        rows.append(row)

    result = pd.DataFrame(rows)
    logger.info("DBH: %s", result["dbh_quality"].value_counts().to_dict())
    return (result, slices) if return_slices else result


def stem_profiles(
    points: pd.DataFrame,
    centres: pd.DataFrame,
    slice_heights_m: Sequence[float] = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5),
    slice_half_height_m: float = 0.25,
    search_radius_m: float = 1.0,
    track_radius_m: float = 0.6,
    max_track_step_m: float = 0.25,
    min_inliers: int = 15,
    min_arc_coverage: float = 0.2,
    max_rmse_m: float = 0.02,
    max_points: int = 1500,
    n_iter: int = 300,
) -> pd.DataFrame:
    """Follow each stem upwards through thin slices and describe its form.

    Parameters
    ----------
    points : DataFrame with x, y, h (height above ground) and instance.
    centres : DataFrame with tree_id, x, y: where to look for the stem in the lowest slice.

    In every slice a circle is fitted (fit_circle_ransac) near the centre of the
    slice below; the first slice searches around the given position. A slice
    counts if the circle has at least `min_inliers` points, covers
    `min_arc_coverage` of the circumference and has an RMSE of at most `max_rmse_m`.

    Returns
    -------
    DataFrame with tree_id and
    - stem_slices: number of slices with a stem circle (0 = no stem found)
    - stem_x, stem_y, stem_offset_m: stem centre in the lowest fitted slice and its distance to the given position
    - stem_points: points on the fitted circles
    - lean_deg: tilt of the straight line through the slice centres (needs 3 slices)
    - sweep_mm: largest horizontal distance of a slice centre from that line (needs 4 slices)
    - taper_mm_per_m: decrease of the diameter with height (needs 3 slices)
    - stem_instance: instance id most of the circle points carry (0 or below = not part of any tree instance)
    - stem_instance_share: share of the circle points with that id
    """
    h = points["h"].to_numpy()
    x_all, y_all, inst_all = points["x"].to_numpy(), points["y"].to_numpy(), points["instance"].to_numpy()
    slices = []
    for height in slice_heights_m:
        idx = np.flatnonzero(np.abs(h - height) <= slice_half_height_m)
        xy = np.column_stack([x_all[idx], y_all[idx]])
        slices.append((height, xy, inst_all[idx], cKDTree(xy)))

    rng = np.random.default_rng(0)
    rows = []
    for rec in centres.itertuples(index=False):
        hint, radius, max_offset = (rec.x, rec.y), search_radius_m, 0.75 * search_radius_m
        fitted, owners = [], []
        for height, xy, inst, index in slices:
            near = np.asarray(index.query_ball_point(hint, r=radius), dtype=int)
            if len(near) > max_points:
                near = rng.choice(near, max_points, replace=False)
            fit = fit_circle_ransac(xy[near], centre_hint=hint, n_iter=n_iter, max_centre_offset_m=max_offset) if len(near) >= 5 else None
            if fit is None or fit["n_inliers"] < min_inliers or fit["arc_coverage"] < min_arc_coverage or fit["rmse_m"] > max_rmse_m:
                continue
            fitted.append((height, fit["cx"], fit["cy"], 2000.0 * fit["radius_m"]))
            on_circle = np.abs(np.hypot(xy[near, 0] - fit["cx"], xy[near, 1] - fit["cy"]) - fit["radius_m"]) <= 0.025
            owners.append(inst[near][on_circle])
            hint, radius, max_offset = (fit["cx"], fit["cy"]), track_radius_m, max_track_step_m

        row = {"tree_id": rec.tree_id, "stem_slices": len(fitted), "stem_x": np.nan, "stem_y": np.nan, "stem_offset_m": np.nan,
               "stem_points": 0, "lean_deg": np.nan, "sweep_mm": np.nan, "taper_mm_per_m": np.nan,
               "stem_instance": np.nan, "stem_instance_share": np.nan}
        if fitted:
            fitted = np.array(fitted)
            owners = np.concatenate(owners)
            ids, counts = np.unique(np.where(owners > 0, owners, 0), return_counts=True)
            row.update({"stem_x": fitted[0, 1], "stem_y": fitted[0, 2],
                        "stem_offset_m": float(np.hypot(fitted[0, 1] - rec.x, fitted[0, 2] - rec.y)),
                        "stem_points": int(len(owners)), "stem_instance": int(ids[np.argmax(counts)]),
                        "stem_instance_share": float(counts.max() / counts.sum())})
            if len(fitted) >= 3:
                design = np.column_stack([fitted[:, 0], np.ones(len(fitted))])
                (slope_x, _), (slope_y, _) = [np.linalg.lstsq(design, fitted[:, k], rcond=None)[0] for k in (1, 2)]
                row["lean_deg"] = float(np.degrees(np.arctan(np.hypot(slope_x, slope_y))))
                row["taper_mm_per_m"] = float(-np.linalg.lstsq(design, fitted[:, 3], rcond=None)[0][0])
                if len(fitted) >= 4:
                    residual = [fitted[:, k] - design @ np.linalg.lstsq(design, fitted[:, k], rcond=None)[0] for k in (1, 2)]
                    row["sweep_mm"] = float(1000.0 * np.hypot(*residual).max())
        rows.append(row)

    result = pd.DataFrame(rows)
    logger.info("Stem profiles: %d of %d trees with a stem, median %s slices",
                int((result["stem_slices"] > 0).sum()), len(result), result["stem_slices"].median())
    return result


def column_structure(
    points: pd.DataFrame,
    centres: pd.DataFrame,
    radius_m: float = 1.5,
    stem_radius_m: float = 0.5,
    stem_zone_m: tuple[float, float] = (1.0, 3.0),
    min_height_m: float = 2.0,
    layer_gap_m: float = 2.0,
) -> pd.DataFrame:
    """Vertical structure of the point cloud in a column around each position.

    Parameters
    ----------
    points : DataFrame with x, y, h (height above ground).
    centres : DataFrame with tree_id, x, y.

    Returns
    -------
    DataFrame with tree_id and
    - column_points: points above `min_height_m` within `radius_m`
    - column_top_m: highest of them
    - canopy_layers: groups of these points separated by height ranges of at least
      `layer_gap_m` without any return (1 = closed column, 2 or more = layered)
    - largest_gap_m: tallest height range without a return below the column top
    - stem_zone_points: points between stem_zone_m[0] and stem_zone_m[1] within
      `stem_radius_m` (how well the lower stem is hit)
    """
    cx, cy = centres["x"].to_numpy(dtype=float), centres["y"].to_numpy(dtype=float)
    x, y, h = points["x"].to_numpy(), points["y"].to_numpy(), points["h"].to_numpy()

    # Keep only points in 1 m cells that touch a column, then index those
    x0, y0 = cx.min() - radius_m - 1, cy.min() - radius_m - 1
    ncols, nrows = int(cx.max() - x0 + radius_m + 2), int(cy.max() - y0 + radius_m + 2)
    wanted = np.zeros((nrows, ncols), dtype=bool)
    reach = int(np.ceil(radius_m)) + 1
    for col, row in zip((cx - x0).astype(int), (cy - y0).astype(int)):
        wanted[max(row - reach, 0):row + reach + 1, max(col - reach, 0):col + reach + 1] = True
    col, row = (x - x0).astype(np.int64), (y - y0).astype(np.int64)
    inside = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows) & np.isfinite(h)
    inside[inside] = wanted[row[inside], col[inside]]
    idx = np.flatnonzero(inside)
    xy = np.column_stack([x[idx], y[idx]])
    heights = h[idx]
    index = cKDTree(xy)

    rows = []
    for tree_id, px, py in zip(centres["tree_id"], cx, cy):
        near = np.asarray(index.query_ball_point([px, py], r=radius_m), dtype=int)
        hh = heights[near]
        stem = near[(hh >= stem_zone_m[0]) & (hh <= stem_zone_m[1])]
        stem_points = int((np.hypot(xy[stem, 0] - px, xy[stem, 1] - py) <= stem_radius_m).sum())
        hh = np.sort(hh[hh >= min_height_m])
        row = {"tree_id": tree_id, "column_points": int(len(hh)), "column_top_m": np.nan, "canopy_layers": 0,
               "largest_gap_m": np.nan, "stem_zone_points": stem_points}
        if len(hh):
            gaps = np.diff(hh)
            row.update({"column_top_m": float(hh[-1]), "canopy_layers": int((gaps >= layer_gap_m).sum()) + 1,
                        "largest_gap_m": float(gaps.max()) if len(gaps) else 0.0})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-tree QA figure
# ---------------------------------------------------------------------------

def _read_rgb_window(path: Path | str, bounds: Bounds) -> np.ndarray:
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        rgb = src.read(indexes=[1, 2, 3], window=window, boundless=True, fill_value=0)
    return rgb.transpose(1, 2, 0)


def plot_match_qa(
    tree: pd.Series,
    points: pd.DataFrame,
    removed: np.ndarray,
    chm: np.ndarray,
    chm_transform: rasterio.Affine,
    ortho_before: Path | str,
    ortho_after: Path | str,
    output_path: Optional[Path | str] = None,
    half_size_m: float = 10.0,
    slab_half_width_m: float = 1.5,
    manual_points: Optional[pd.DataFrame] = None,
    refined_points: Optional[pd.DataFrame] = None,
    other_stems: Optional[pd.DataFrame] = None,
):
    """One figure per field tree: orthos before/after, LiDAR CHM, vertical profile.

    `tree` is a row of the merged field + match table (utm_x, utm_y, x_top, y_top,
    x_base, y_base, instance, lidar_height, height_vertex, height_massband, ...).
    The profile shows a vertical slab through stem and top: matched instance in
    red, other removed points orange, everything else grey. `manual_points`
    (x, y, h of a hand-segmented tree) are drawn in blue on top. `refined_points`
    (x, y, h of a post-processed instance, see stem_seeded_refinement) are drawn
    in purple: in the profile, and as crown outline and top in the three maps.
    `other_stems` (x, y) are further field stems, drawn as white circles in the
    maps and as dotted lines in the profile where they lie inside the slab.
    """
    import matplotlib.pyplot as plt

    sx, sy = float(tree["utm_x"]), float(tree["utm_y"])
    has_top = pd.notna(tree.get("x_top"))
    bounds = (sx - half_size_m, sy - half_size_m, sx + half_size_m, sy + half_size_m)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6))

    for ax, path, title in [(axes[0], ortho_before, "Ortho before"), (axes[1], ortho_after, "Ortho after")]:
        ax.imshow(_read_rgb_window(path, bounds), extent=extent)
        ax.set_title(title)

    col0, row0 = ~chm_transform * (bounds[0], bounds[3])
    col1, row1 = ~chm_transform * (bounds[2], bounds[1])
    crop = chm[max(int(row0), 0) : int(row1) + 1, max(int(col0), 0) : int(col1) + 1]
    im = axes[2].imshow(crop, extent=extent, cmap="viridis", vmin=0, vmax=38)
    axes[2].set_title("LiDAR CHM before")
    fig.colorbar(im, ax=axes[2], fraction=0.046, label="m")

    for ax in axes[:3]:
        ax.plot(sx, sy, "x", color="red", markersize=11, markeredgewidth=2.5, label="field stem")
        if has_top:
            ax.plot(tree["x_base"], tree["y_base"], "+", color="cyan", markersize=11, markeredgewidth=2, label="LiDAR base")
            ax.plot(tree["x_top"], tree["y_top"], "^", color="yellow", markeredgecolor="black", markersize=9, label="LiDAR top")
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_xticks([])
        ax.set_yticks([])
    if refined_points is not None and len(refined_points) >= 3:
        r_xy = refined_points[["x", "y"]].to_numpy()
        r_top = refined_points.loc[refined_points["h"].idxmax()]
        outline = r_xy[ConvexHull(r_xy).vertices]
        outline = np.vstack([outline, outline[:1]])
        for ax in axes[:3]:
            ax.plot(outline[:, 0], outline[:, 1], color="#a020f0", linewidth=1.6, label="post-processed crown")
            ax.plot(r_top["x"], r_top["y"], "*", color="#a020f0", markeredgecolor="white", markersize=15, label="post-processed top")
    if other_stems is not None and len(other_stems):
        for ax in axes[:3]:
            ax.plot(other_stems["x"], other_stems["y"], "o", markerfacecolor="none", markeredgecolor="white",
                    markersize=7, markeredgewidth=1.5, label="other field stems")
    axes[0].legend(loc="upper left", fontsize=8)

    # Vertical slab through stem and top (east-west if they coincide)
    direction = np.array([1.0, 0.0])
    if has_top:
        delta = np.array([tree["x_top"] - sx, tree["y_top"] - sy])
        if np.hypot(*delta) > 0.5:
            direction = delta / np.hypot(*delta)
    normal = np.array([-direction[1], direction[0]])

    x, y, h = points["x"].to_numpy(), points["y"].to_numpy(), points["h"].to_numpy()
    box = (np.abs(x - sx) <= half_size_m) & (np.abs(y - sy) <= half_size_m)
    idx = np.flatnonzero(box)
    rel = np.column_stack([x[idx] - sx, y[idx] - sy])
    in_slab = np.abs(rel @ normal) <= slab_half_width_m
    idx, along = idx[in_slab], (rel @ direction)[in_slab]

    inst = points["instance"].to_numpy()[idx]
    is_match = inst == tree["instance"] if pd.notna(tree.get("instance")) else np.zeros(len(idx), dtype=bool)
    is_removed = removed[idx] & ~is_match
    rest = ~is_match & ~is_removed

    ax = axes[3]
    step = max(1, int(rest.sum() // 40000))
    ax.scatter(along[rest][::step], h[idx][rest][::step], s=0.3, color="0.6", label="other points")
    step = max(1, int(is_removed.sum() // 40000))
    ax.scatter(along[is_removed][::step], h[idx][is_removed][::step], s=0.3, color="orange", label="removed (other trees)")
    step = max(1, int(is_match.sum() // 40000))
    ax.scatter(along[is_match][::step], h[idx][is_match][::step], s=0.5, color="red", label="SAT instance")

    if manual_points is not None and len(manual_points):
        m_rel = np.column_stack([manual_points["x"].to_numpy() - sx, manual_points["y"].to_numpy() - sy])
        m_slab = np.abs(m_rel @ normal) <= slab_half_width_m
        step = max(1, int(m_slab.sum() // 40000))
        ax.scatter((m_rel @ direction)[m_slab][::step], manual_points["h"].to_numpy()[m_slab][::step],
                   s=0.5, color="blue", label="manual segment")

    if refined_points is not None and len(refined_points):
        r_rel = np.column_stack([refined_points["x"].to_numpy() - sx, refined_points["y"].to_numpy() - sy])
        r_slab = np.abs(r_rel @ normal) <= slab_half_width_m
        ax.scatter((r_rel @ direction)[r_slab], refined_points["h"].to_numpy()[r_slab], s=1.2, color="#a020f0",
                   label=f"post-processed instance ({int(r_slab.sum()):,} of {len(refined_points):,} points in slab)")
        ax.axhline(refined_points["h"].max(), color="#a020f0", linewidth=1.2, linestyle="--",
                   label=f"post-processed {refined_points['h'].max():.1f} m")

    ax.axvline(0, color="red", linewidth=0.8, linestyle=":")
    if other_stems is not None and len(other_stems):
        o_rel = np.column_stack([other_stems["x"].to_numpy() - sx, other_stems["y"].to_numpy() - sy])
        for pos in (o_rel @ direction)[np.abs(o_rel @ normal) <= slab_half_width_m]:
            ax.axvline(pos, color="0.3", linewidth=0.8, linestyle=":")
    for col, color, label in [("height_vertex", "blue", "Vertex"), ("height_massband", "green", "Maßband"),
                              ("height_standing", "darkorange", "standing height")]:
        if pd.notna(tree.get(col)):
            ax.axhline(tree[col], color=color, linewidth=1.2, label=f"{label} {tree[col]:.1f} m")
    if pd.notna(tree.get("lidar_height")):
        ax.axhline(tree["lidar_height"], color="black", linewidth=1.2, linestyle="--", label=f"LiDAR {tree['lidar_height']:.1f} m")
    ax.set_xlim(-half_size_m, half_size_m)
    ax.set_ylim(0, 40)
    ax.set_xlabel("distance along stem → top direction (m)")
    ax.set_ylabel("height above ground (m)")
    ax.set_title(f"Profile, {2 * slab_half_width_m:.0f} m slab")
    ax.legend(loc="upper right", fontsize=7, markerscale=10)

    dist = f"{tree['base_dist_m']:.2f} m" if pd.notna(tree.get("base_dist_m")) else "n/a"
    record = "n/a" if pd.isna(tree.get("removed_2026", np.nan)) else bool(tree.get("removed_2026"))
    fig.suptitle(
        f"Tree {tree['tree_id']}  |  {tree.get('confidence', '')}  |  method {tree.get('method', '')}  |  "
        f"instance {tree.get('instance', '')}  |  stem→base {dist}  |  "
        f"removed 2026: {record}, LiDAR removed fraction {tree.get('removed_fraction', np.nan):.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=110, bbox_inches="tight")
    return fig
