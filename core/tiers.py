"""Result tiers: automatic, manually picked, hand-segmented.

The detection class of `core.matching` never uses a field height. Here the height is added as
flags and as separate tiers, so that SegmentAnyTree numbers never contain manual work:

- height gate (flag only): LiDAR and field height differ by more than `matching.height_gate_rel`
- column profile fallback for trees without an instance or with a failed gate
- manual overrides (another instance, or no LiDAR tree) and hand-segmented trees, both optional

`auto_height`, `auto_confidence` and `auto_method` keep the automatic result of every tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

TIER_COLUMNS = ["tree_id", "instance", "lidar_height", "sat_height", "x_top", "y_top", "x_base", "y_base",
                "field_height", "height_gate_ok", "confidence", "method"]
_CLASS_TO_CONFIDENCE = {"detected": "high", "uncertain": "check", "not detected": "unmatched"}


def matches_from_detection(detection: pd.DataFrame) -> pd.DataFrame:
    """Matcher output rebuilt from field_trees_detection.csv (confidence and method are implied by the class)."""
    out = detection[["tree_id", "instance", "lidar_height", "x_top", "y_top", "x_base", "y_base"]].copy()
    out["instance"] = out["instance"].astype("Int64")
    out["confidence"] = detection["detection"].astype(str).map(_CLASS_TO_CONFIDENCE)
    out["method"] = np.where(out["instance"].notna(), "sat_instance", "none")
    return out


# ---------------------------------------------------------------------------
# Height gate and column profile fallback
# ---------------------------------------------------------------------------

def add_height_gate(matches: pd.DataFrame, field: pd.DataFrame, height_gate_rel: float) -> pd.DataFrame:
    """Set `height_gate_ok` on a matching that was made without field heights.

    The gate is a flag, not a selection: a matched tree whose LiDAR height differs from the field
    height (Vertex, else tape) by more than `height_gate_rel` of it drops from 'high' to 'check'.
    Trees without a field height keep NA.
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


def column_profile_top(x, y, h, centre, radius_m, max_gap_m, min_height_m, min_points):
    """Top of the lowest continuous point layer above a stem (None if fewer than `min_points` points)."""
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
    return {"height": float(heights[last]), "x_top": float(x[order[last]]), "y_top": float(y[order[last]]),
            "n_points": int(last + 1), "hit_gap": bool(len(gaps) > 0)}


def add_column_profile_fallback(matches: pd.DataFrame, field: pd.DataFrame, points: pd.DataFrame, removed: np.ndarray,
                                radius_m: float, max_gap_m: float, min_height_m: float, min_points: int) -> pd.DataFrame:
    """Column-profile height for trees that are unmatched or failed the height gate.

    Only points that are still unexplained are used: points of instances assigned to another field
    tree are excluded, and for trees felled in the harvest only removed points count. If a layer is
    found, `lidar_height` and the top position switch to it and `method` becomes 'column_profile'
    (confidence 'check'). A tree that failed the gate and has no such layer becomes 'unresolved'
    with no LiDAR height. The SegmentAnyTree result is always kept in `sat_height`.

    points: x, y, h (height above ground), instance.
    """
    matches = matches.copy()
    matches["sat_height"] = matches["lidar_height"]
    stems = field.set_index("tree_id")

    needs = (matches["confidence"] == "unmatched") | (matches["height_gate_ok"].isin([False]))
    needs &= matches["confidence"] != "manual"      # manual overrides are final
    todo = matches.index[needs]
    if len(todo) == 0:
        return matches

    # Keep only the points near a stem that needs a profile (the cloud has hundreds of millions of points)
    centres = stems.loc[matches.loc[todo, "tree_id"], ["utm_x", "utm_y"]].to_numpy(dtype=float)
    x_all, y_all = points["x"].to_numpy(), points["y"].to_numpy()
    index = cKDTree(centres)
    keep = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), 5_000_000):
        sl = slice(start, start + 5_000_000)
        d, _ = index.query(np.column_stack([x_all[sl], y_all[sl]]), p=np.inf, distance_upper_bound=radius_m + 1e-9)
        keep[sl] = np.isfinite(d)
    rows = np.flatnonzero(keep)
    x, y = x_all[rows], y_all[rows]
    h, inst, gone = points["h"].to_numpy()[rows], points["instance"].to_numpy()[rows], np.asarray(removed)[rows]

    for i in todo:
        tree_id = matches.at[i, "tree_id"]
        stem = stems.loc[tree_id]
        cx, cy = float(stem["utm_x"]), float(stem["utm_y"])

        owned_by_others = matches.loc[matches.index != i, "instance"].dropna().astype(int).to_numpy()
        box = np.flatnonzero((np.abs(x - cx) <= radius_m) & (np.abs(y - cy) <= radius_m))
        sel = ~np.isin(inst[box], owned_by_others)
        if stem["removed_2026"]:
            sel &= gone[box]
        box = box[sel]
        profile = column_profile_top(x[box], y[box], h[box], (cx, cy), radius_m, max_gap_m, min_height_m, min_points)

        if profile is not None:
            matches.loc[i, ["lidar_height", "x_top", "y_top"]] = [profile["height"], profile["x_top"], profile["y_top"]]
            matches.loc[i, ["x_base", "y_base"]] = [cx, cy]
            matches.loc[i, ["method", "confidence"]] = ["column_profile", "check"]
        elif matches.at[i, "confidence"] != "unmatched":
            matches.loc[i, ["lidar_height", "x_top", "y_top"]] = np.nan
            matches.loc[i, ["method", "confidence"]] = ["none", "unresolved"]

    return matches


# ---------------------------------------------------------------------------
# Manual corrections (own tiers)
# ---------------------------------------------------------------------------

def apply_match_overrides(matches: pd.DataFrame, instances: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Manual corrections: columns tree_id, instance (0 or empty = no LiDAR tree). Method stays 'manual' / sat_instance."""
    matches = matches.copy()
    by_instance = instances.set_index("instance")
    cleared = ["instance", "lidar_height", "x_top", "y_top", "x_base", "y_base"]

    for rec in overrides.itertuples(index=False):
        hit = matches.index[matches["tree_id"] == str(rec.tree_id)]
        if len(hit) == 0:
            continue
        i = hit[0]

        if pd.isna(rec.instance) or int(rec.instance) <= 0:
            matches.loc[i, cleared] = [pd.NA] + [np.nan] * 5
            matches.loc[i, ["confidence", "method"]] = ["manual", "none"]
            continue
        inst_id = int(rec.instance)
        if inst_id not in by_instance.index:
            continue
        inst = by_instance.loc[inst_id]

        # Free the instance if another tree currently holds it
        for h in matches.index[(matches["instance"] == inst_id) & (matches.index != i)]:
            matches.loc[h, cleared] = [pd.NA] + [np.nan] * 5
            matches.loc[h, ["confidence", "method"]] = ["unmatched", "none"]

        matches.loc[i, "instance"] = inst_id
        for col, src in [("lidar_height", "height"), ("x_top", "x_top"), ("y_top", "y_top"),
                         ("x_base", "x_base"), ("y_base", "y_base"),
                         ("removed_fraction", "removed_fraction"), ("lean_m", "lean_m"),
                         ("crown_base_lidar_m", "crown_base_m")]:
            if col in matches.columns:
                matches.loc[i, col] = float(inst[src])
        if "sat_height" in matches.columns:
            matches.loc[i, "sat_height"] = float(inst["height"])
        matches.loc[i, ["confidence", "method"]] = ["manual", "sat_instance"]

    return matches


def load_manual_segments(folder, dtm_path, stem_slice_m) -> pd.DataFrame:
    """Hand-segmented trees saved as tree_<Parzelle>_<Nr>.laz / .las. Heights come from z and the DTM.

    Returns one row per tree: tree_id, lidar_height, x_top, y_top, x_base, y_base, lean_m, n_points, file.
    """
    import laspy

    from core.pointcloud import sample_raster_bilinear

    columns = ["tree_id", "lidar_height", "x_top", "y_top", "x_base", "y_base", "lean_m", "n_points", "file"]
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".laz", ".las"}) if folder.exists() else []
    rows, seen = [], set()
    for path in files:
        found = re.search(r"tree_(\d+_\d+)", path.stem)
        if not found or found.group(1) in seen:
            continue
        tree_id = found.group(1)
        las = laspy.read(path)
        x, y = np.asarray(las.x, dtype=np.float64), np.asarray(las.y, dtype=np.float64)
        h = np.asarray(las.z, dtype=np.float64) - sample_raster_bilinear(dtm_path, x, y)
        ok = np.isfinite(h)
        if not ok.any():
            continue
        x, y, h = x[ok], y[ok], h[ok]
        seen.add(tree_id)

        top = int(np.argmax(h))
        in_slice = (h >= stem_slice_m[0]) & (h <= stem_slice_m[1])
        xy = np.column_stack([x, y])
        if in_slice.sum() >= 10:
            base_pts = xy[in_slice]
        else:
            base_pts = xy[np.argsort(h)[:max(10, int(0.05 * len(h)))]]
        base = np.median(base_pts, axis=0)
        rows.append({"tree_id": tree_id, "lidar_height": float(h[top]), "x_top": float(x[top]), "y_top": float(y[top]),
                     "x_base": float(base[0]), "y_base": float(base[1]),
                     "lean_m": float(np.hypot(x[top] - base[0], y[top] - base[1])), "n_points": int(len(h)),
                     "file": path.name})
    return pd.DataFrame(rows, columns=columns)


def apply_manual_segments(matches: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Use hand-segmented trees as the LiDAR result (method 'manual_segmentation', confidence 'manual').

    These are reference segments made by a person, not SegmentAnyTree results; the automatic
    outcome stays in the auto_* / sat_height columns.
    """
    matches = matches.copy()
    for rec in manual.itertuples(index=False):
        hit = matches.index[matches["tree_id"] == rec.tree_id]
        if len(hit) == 0:
            continue
        i = hit[0]
        for col in ["lidar_height", "x_top", "y_top", "x_base", "y_base", "lean_m"]:
            matches.loc[i, col] = getattr(rec, col)
        matches.loc[i, ["method", "confidence"]] = ["manual_segmentation", "manual"]
    return matches


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def automatic_tier(matches, field, points, removed, *, height_gate_rel, radius_m, max_gap_m, min_height_m, min_points):
    """match_tiers_auto.csv: height gate, then column profile fallback."""
    auto = add_height_gate(matches, field, height_gate_rel)
    auto = add_column_profile_fallback(auto, field, points, removed, radius_m, max_gap_m, min_height_m, min_points)
    return auto[TIER_COLUMNS].reset_index(drop=True)


def final_tier(auto: pd.DataFrame, instances: pd.DataFrame, overrides=None, manual=None) -> pd.DataFrame:
    """match_tiers.csv: the automatic tier plus the auto_* columns, then manual overrides and segments.

    Without overrides and manual segments the final tier equals the automatic tier.
    """
    tiers = auto.copy()
    tiers["instance"] = tiers["instance"].astype("Int64")
    tiers["auto_height"], tiers["auto_confidence"], tiers["auto_method"] = tiers["lidar_height"], tiers["confidence"], tiers["method"]
    if overrides is not None and len(overrides):
        tiers = apply_match_overrides(tiers, instances, overrides)
    if manual is not None and len(manual):
        tiers = apply_manual_segments(tiers, manual)
    return tiers


def run_tiers(cfg) -> None:
    """Stage 9: writes match_tiers_auto.csv and match_tiers.csv."""
    import geopandas as gpd

    from core import instances as core_instances
    from core.cache import cached_table

    field = pd.read_csv(cfg.table("field_trees.csv"), dtype={"tree_id": str})
    detection = pd.read_csv(cfg.table("field_trees_detection.csv"), dtype={"tree_id": str})
    instances = gpd.read_file(cfg.table("lidar_instances.gpkg"), layer="instance_base")
    cp = cfg.matching.column_profile

    def compute():
        cloud = core_instances.load_before_cloud(cfg)
        if "removed" not in cloud.columns:
            raise RuntimeError("load_before_cloud must return a boolean column 'removed' (point gone after the harvest)")
        return automatic_tier(matches_from_detection(detection), field, cloud, cloud["removed"].to_numpy(bool),
                              height_gate_rel=cfg.matching.height_gate_rel, radius_m=cp.radius_m, max_gap_m=cp.max_gap_m,
                              min_height_m=cp.min_height_m, min_points=cp.min_points)

    inputs = [cfg.table("field_trees.csv"), cfg.table("field_trees_detection.csv")]
    auto = cached_table(cfg, "match_tiers_auto.csv", ["matching", "grid"], inputs, compute)
    auto["tree_id"] = auto["tree_id"].astype(str)
    if not auto["tree_id"].equals(detection["tree_id"]):
        raise RuntimeError("match_tiers_auto.csv belongs to another matching: rerun with --force")

    overrides = None
    path = cfg.paths.get("match_overrides")
    if path is not None and Path(path).is_file():
        overrides = pd.read_csv(path, dtype={"tree_id": str})
    manual = None
    folder = cfg.paths.get("manual_segments_dir")
    if folder is not None and Path(folder).is_dir():
        manual = load_manual_segments(folder, core_instances.dtm_path(cfg), cfg.matching.stem_slice_m)

    tiers = final_tier(auto, instances, overrides, manual)
    tiers.to_csv(cfg.table("match_tiers.csv"), index=False)
    print(f"tiers: automatic {tiers['auto_confidence'].value_counts().to_dict()}")
