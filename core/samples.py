"""Side-view points around every sample tree (stage `samples`), for the dashboard.

sample_points() is a pure function (cloud and trees in, DataFrame out). Only run_samples() reads and writes files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GROUPS = ("matched instance", "other instance", "no instance")


def sample_points(cloud: pd.DataFrame, trees: pd.DataFrame, radius_m: float = 4.0, max_points: int = 1500,
                  seed: int = 0) -> pd.DataFrame:
    """Points within radius_m (horizontal) of each tree's stem position.

    cloud: columns x, y, h (height above ground), instance (0 = no tree).
    trees: columns tree_id, x, y (stem position in the cloud's coordinates), instance (matched instance id or NaN).
    Returns tree_id, x_rel, y_rel, h, group; at most max_points random points per tree (fixed seed).
    """
    xy = np.column_stack([cloud["x"].to_numpy(float), cloud["y"].to_numpy(float)])
    h = cloud["h"].to_numpy(float)
    inst = cloud["instance"].to_numpy()
    index = cKDTree(xy)
    rng = np.random.default_rng(seed)
    parts = []
    for row in trees.itertuples(index=False):
        idx = np.asarray(index.query_ball_point([row.x, row.y], r=radius_m), dtype=np.int64)
        if len(idx) > max_points:
            idx = np.sort(rng.choice(idx, size=max_points, replace=False))
        i = inst[idx]
        matched = pd.notna(row.instance) & (i == row.instance)
        group = np.where(matched, GROUPS[0], np.where(i > 0, GROUPS[1], GROUPS[2]))
        parts.append(pd.DataFrame({"tree_id": row.tree_id, "x_rel": xy[idx, 0] - row.x, "y_rel": xy[idx, 1] - row.y,
                                   "h": h[idx], "group": group}))
    if not parts:
        return pd.DataFrame(columns=["tree_id", "x_rel", "y_rel", "h", "group"])
    return pd.concat(parts, ignore_index=True)


def run_samples(cfg) -> None:
    """Write sample_tree_points.csv for every tree with a field height."""
    from core import instances as core_instances

    detection = pd.read_csv(cfg.table("field_trees_detection.csv"), dtype={"tree_id": str})
    tiers = pd.read_csv(cfg.table("match_tiers.csv"), dtype={"tree_id": str}).set_index("tree_id")
    trees = detection.loc[detection["has_height"].astype(bool), ["tree_id", "utm_x", "utm_y"]].rename(
        columns={"utm_x": "x", "utm_y": "y"})
    matched = tiers["instance"].where(tiers["method"].astype(str).ne("none"))
    trees["instance"] = trees["tree_id"].map(matched)

    s = cfg.raw.get("samples") or {}
    radius = float(s.get("radius_m", 4.0))
    max_points = int(s.get("max_points_per_tree", 1500))
    cloud = core_instances.load_before_cloud(cfg)
    out = sample_points(cloud, trees, radius, max_points, seed=0)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.table("sample_tree_points.csv"), index=False)
    print(f"samples: {trees['tree_id'].nunique()} trees, {len(out)} points")
