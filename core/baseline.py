"""Baseline: tree tops as local maxima of the LiDAR canopy height model (CHM).

A cell of the smoothed CHM is a tree top if no cell within a window around it is higher. The window
is fixed or grows with the cell height. A CHM method has no stem base, so the comparison is made on
the tops: the tops of every method and the tops of the SegmentAnyTree instances are assigned to the
field stems with the same one-to-one assignment as the instance matching (no harvest penalty).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree

from core.matching import match_stems_to_instances

SAT_METHOD = "SegmentAnyTree instance tops"
SHIFTS = [(0, 0, "none"), (0, 1, "north"), (1, 0, "east"), (0, -1, "south"), (-1, 0, "west")]


def fill_gaps(chm: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Cells without a return take the highest of their 3 x 3 neighbours (repeated); what is left becomes 0."""
    filled = chm.astype(np.float32).copy()
    for _ in range(iterations):
        empty = ~np.isfinite(filled)
        if not empty.any():
            break
        grown = ndimage.maximum_filter(np.where(empty, -np.inf, filled), size=3)
        take = empty & np.isfinite(grown)
        filled[take] = grown[take]
    filled[~np.isfinite(filled)] = 0.0
    return filled


def _disk(radius_px: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (xx ** 2 + yy ** 2) <= radius_px ** 2


def local_maxima_tops(chm: np.ndarray, transform, window_m, min_height_m: float, smoothing_sigma_m: float,
                      height_radius_m: float = 0.5) -> pd.DataFrame:
    """Tree tops as local maxima of the smoothed CHM.

    window_m: window diameter in m, or a function of the (smoothed) cell height returning the diameter.
    Height of a top = the highest cell of the unsmoothed CHM within `height_radius_m` of it.
    Returns top_id, x_top, y_top, height, height_smoothed, window_m.
    """
    res = abs(transform.a)
    filled = fill_gaps(chm)
    smooth = ndimage.gaussian_filter(filled, smoothing_sigma_m / res) if smoothing_sigma_m > 0 else filled
    tall = smooth >= min_height_m

    diameter = window_m(smooth) if callable(window_m) else np.full(smooth.shape, float(window_m))
    radius_px = np.maximum(np.round(diameter / 2 / res), 1).astype(np.int32)

    is_top = np.zeros(smooth.shape, dtype=bool)
    for r in np.unique(radius_px[tall]):
        cells = tall & (radius_px == r)
        is_top |= cells & (smooth >= ndimage.maximum_filter(smooth, footprint=_disk(int(r))))

    # A flat top can give several touching cells: keep one per group
    groups, n = ndimage.label(is_top, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return pd.DataFrame(columns=["top_id", "x_top", "y_top", "height", "height_smoothed", "window_m"])
    centres = np.array(ndimage.center_of_mass(is_top, groups, np.arange(1, n + 1)))
    rows, cols = np.round(centres[:, 0]).astype(int), np.round(centres[:, 1]).astype(int)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")

    peak = ndimage.maximum_filter(filled, footprint=_disk(max(int(round(height_radius_m / res)), 1)))
    return pd.DataFrame({
        "top_id": np.arange(n), "x_top": np.asarray(xs, dtype=float), "y_top": np.asarray(ys, dtype=float),
        "height": peak[rows, cols].astype(float), "height_smoothed": smooth[rows, cols].astype(float),
        "window_m": diameter[rows, cols].astype(float),
    })


def window_function(diameter_m: float, per_height_m: float = 0.0):
    """Fixed window diameter, or one that grows with the cell height."""
    if not per_height_m:
        return float(diameter_m)
    return lambda h: diameter_m + per_height_m * h


def baseline_tops(chm: np.ndarray, transform, windows, instances: pd.DataFrame, plots, *, min_height_m: float,
                  smoothing_m: float) -> pd.DataFrame:
    """Tops of every window plus the SegmentAnyTree instance tops, with the Parzelle of each top.

    windows: list of (name, diameter_m, per_height_m).
    """
    import geopandas as gpd

    all_tops = [local_maxima_tops(chm, transform, window_function(d, p), min_height_m, smoothing_m).assign(method=name)
                for name, d, p in windows]
    all_tops.append(instances[["instance", "x_top", "y_top", "height"]].rename(columns={"instance": "top_id"})
                    .assign(method=SAT_METHOD))
    all_tops = pd.concat(all_tops, ignore_index=True)
    located = gpd.sjoin(gpd.GeoDataFrame(all_tops, geometry=gpd.points_from_xy(all_tops["x_top"], all_tops["y_top"]),
                                         crs=plots.crs),
                        plots[["parzelle", "geometry"]], predicate="within", how="left")
    all_tops["parzelle"] = located.loc[~located.index.duplicated(), "parzelle"].to_numpy()
    return all_tops


def baseline_matches(all_tops: pd.DataFrame, stems: pd.DataFrame, *, top_match_max_m: float,
                     top_found_distance_m: float, control_shift_m: float) -> pd.DataFrame:
    """One-to-one assignment of the tops of each method to the field stems, plus the shifted-stem chance control.

    stems: tree_id, utm_x, utm_y, removed_2026 (no height columns).
    """
    rows = []
    for name, tops in all_tops.groupby("method", sort=False):
        candidates = pd.DataFrame({"instance": tops["top_id"], "x_base": tops["x_top"], "y_base": tops["y_top"],
                                   "x_top": tops["x_top"], "y_top": tops["y_top"], "height": tops["height"],
                                   "removed_fraction": 0.0, "lean_m": 0.0, "crown_base_m": np.nan})
        for dx, dy, shift in SHIFTS:
            shifted = stems.assign(utm_x=stems["utm_x"] + dx * control_shift_m, utm_y=stems["utm_y"] + dy * control_shift_m)
            reach = cKDTree(shifted[["utm_x", "utm_y"]].to_numpy()).query(
                candidates[["x_base", "y_base"]].to_numpy(), distance_upper_bound=top_match_max_m)[0]
            # No field heights in the stems: the height gate is not used (rel. 0)
            assigned = match_stems_to_instances(
                shifted, candidates[np.isfinite(reach)], max_distance_m=top_match_max_m, removed_threshold=0.5,
                removal_penalty_m=0.0, high_confidence_distance_m=top_found_distance_m, height_gate_rel=0.0)
            rows.append(assigned[["tree_id", "instance", "base_dist_m", "lidar_height"]]
                        .rename(columns={"instance": "top_id", "base_dist_m": "top_dist_m", "lidar_height": "top_height"})
                        .assign(method=name, shift=shift))
    return pd.concat(rows, ignore_index=True)


def run_baseline(cfg) -> None:
    """Stage 11: writes chm_baseline_tops.csv and chm_baseline_matches.csv."""
    from core import instances as core_instances
    from core.matching import load_plots

    plots, _ = load_plots(cfg)                    # Excel first
    import geopandas as gpd

    field = pd.read_csv(cfg.table("field_trees.csv"), dtype={"tree_id": str})
    instances = gpd.read_file(cfg.table("lidar_instances.gpkg"), layer="instance_base")
    with rasterio.open(core_instances.chm_path(cfg)) as src:
        chm = src.read(1).astype(np.float32)
        chm[chm == src.nodata] = np.nan
        transform = src.transform

    b = cfg.baseline
    windows = [(w.name, w.diameter_m, w.get("per_height_m", 0.0)) for w in b.windows]
    tops = baseline_tops(chm, transform, windows, instances, plots, min_height_m=b.min_height_m, smoothing_m=b.smoothing_m)
    matches = baseline_matches(tops, field[["tree_id", "utm_x", "utm_y", "removed_2026"]],
                               top_match_max_m=b.top_match_max_m, top_found_distance_m=b.top_found_distance_m,
                               control_shift_m=cfg.matching.control_shift_m)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    tops.to_csv(cfg.table("chm_baseline_tops.csv"), index=False)
    matches.to_csv(cfg.table("chm_baseline_matches.csv"), index=False)
    found = matches[matches["shift"] == "none"].groupby("method", sort=False)["top_dist_m"].apply(
        lambda d: (d <= b.top_found_distance_m).mean())
    print(pd.DataFrame({"tops": tops.groupby("method", sort=False).size(), "found": found.round(3)}).to_string())
