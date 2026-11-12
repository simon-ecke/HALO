"""The stem at every field position (stage 6) and the LiDAR diameter at breast height (stage 8).

Pure functions take data and thresholds and return DataFrames; the RANSAC seed and the order of the
calls are those of the notebook so that results are identical. Only run_stems() and run_dbh() read
inputs and write tables. All thresholds come from the site config.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from core.cache import cached_table
from core.causes import plot_extent, read_corners

logger = logging.getLogger("core.dbh")

DETECTION_CLASSES = ['detected', 'uncertain', 'not detected']
STEM_OWNERS = ['own instance', 'instance of another field tree', 'instance without a field tree', 'not in any instance',
               'no stem at the position']


# ---------------------------------------------------------------------------
# Circle and ellipse fit
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

    RANSAC keeps the circle that most points lie on (within `tolerance_m`). Because a stem is
    opaque, each point lying more than `tolerance_m` inside a candidate circle costs
    `interior_weight` inliers. Returns None if no plausible circle is found.
    """
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

    Uses the points within `band_m` of the circle and a robust least-squares fit (soft-L1 loss) of
    centre, size, axis ratio and orientation. The axis ratio is limited to `max_axis_ratio`.
    Perimeter-equivalent diameter = ellipse circumference / pi (what a girth tape measures);
    area-equivalent diameter = 2 * sqrt(a * b).
    """
    cx, cy, r = circle["cx"], circle["cy"], circle["radius_m"]
    near = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r) <= band_m
    # Work relative to the circle centre: with UTM-sized coordinates the numerical derivatives break down.
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
    breast_height_m: float,
    slice_half_height_m: float,
    search_radius_m: float,
    ellipse_axis_ratio: float,
    min_inliers: int = 15,
    min_arc_coverage: float = 0.25,
    max_rmse_m: float = 0.02,
    max_interior_share: float = 0.2,
    **ransac_kwargs,
) -> pd.DataFrame:
    """LiDAR DBH per tree from a thin point slice at breast height.

    points: x, y, h (height above ground). centres: tree_id, x, y (where to look for the stem).

    A fit is 'ok' if it has at least `min_inliers` points on the circle, covers at least
    `min_arc_coverage` of the circumference, has an RMSE of at most `max_rmse_m` and few points inside
    the circle; otherwise 'weak'; 'no_fit' if no circle was found. For every circle fit an ellipse is
    fitted as well; its perimeter-equivalent diameter is dbh_ellipse_mm.
    """
    h = points["h"].to_numpy()
    in_slice = np.flatnonzero(np.abs(h - breast_height_m) <= slice_half_height_m)
    slice_xy = np.column_stack([points["x"].to_numpy()[in_slice], points["y"].to_numpy()[in_slice]])
    index = cKDTree(slice_xy)

    rows = []
    for rec in centres.itertuples(index=False):
        near = index.query_ball_point([rec.x, rec.y], r=search_radius_m)
        xy = slice_xy[near]
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
            ellipse = fit_ellipse(xy, fit, tolerance_m=ransac_kwargs.get("tolerance_m", 0.025),
                                  max_axis_ratio=ellipse_axis_ratio)
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
    return result


def dbh_table(points: pd.DataFrame, trees: pd.DataFrame, *, breast_height_m: float, slice_half_height_m: float,
              search_radius_m: float, ellipse_axis_ratio: float) -> pd.DataFrame:
    """LiDAR DBH for every field tree: look at the LiDAR stem base, or at the field position if the tree has no instance.

    trees: field_trees_detection.csv (tree_id, utm_x, utm_y, x_base, y_base).
    """
    use_lidar = trees['x_base'].notna()
    centres = pd.DataFrame({'tree_id': trees['tree_id'],
                            'x': np.where(use_lidar, trees['x_base'], trees['utm_x']),
                            'y': np.where(use_lidar, trees['y_base'], trees['utm_y'])})
    dbh = estimate_dbh(points, centres, breast_height_m=breast_height_m, slice_half_height_m=slice_half_height_m,
                       search_radius_m=search_radius_m, ellipse_axis_ratio=ellipse_axis_ratio)
    values = pd.DataFrame({'tree_id': trees['tree_id'],
                           'dbh_centre_from': np.where(use_lidar, 'LiDAR stem base', 'field position')})
    return values.merge(dbh, on='tree_id', how='left')


# ---------------------------------------------------------------------------
# Stem form and column structure
# ---------------------------------------------------------------------------

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

    points: x, y, h (height above ground) and instance. centres: tree_id, x, y (where to look for the
    stem in the lowest slice). In every slice a circle is fitted near the centre of the slice below.

    Returns tree_id, stem_slices, stem_x, stem_y, stem_offset_m, stem_points, lean_deg, sweep_mm,
    taper_mm_per_m, stem_instance (0 = not part of any tree instance) and stem_instance_share.
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
    logger.info("Stem profiles: %d of %d trees with a stem", int((result["stem_slices"] > 0).sum()), len(result))
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

    points: x, y, h. centres: tree_id, x, y.

    Returns tree_id, column_points (points above `min_height_m` within `radius_m`), column_top_m,
    canopy_layers (groups of points separated by at least `layer_gap_m` without a return),
    largest_gap_m and stem_zone_points (points in the stem zone within `stem_radius_m`).
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


def field_positions(trees: pd.DataFrame) -> pd.DataFrame:
    """tree_id, x, y of every field stem (detected or not, so all classes are treated alike)."""
    return trees[['tree_id']].assign(x=trees['utm_x'], y=trees['utm_y'])


def stem_form_table(cloud: pd.DataFrame, trees: pd.DataFrame, *, stem_zone_m: tuple[float, float] = (1.0, 9.0)) -> pd.DataFrame:
    """stem_form_per_tree: stem profile at every field position, from the cloud between the stem-zone heights."""
    stem_zone = cloud[(cloud['h'] > stem_zone_m[0]) & (cloud['h'] < stem_zone_m[1])]
    return stem_profiles(stem_zone, field_positions(trees))


# ---------------------------------------------------------------------------
# Tree profile
# ---------------------------------------------------------------------------

def tree_profile(struct: pd.DataFrame, trees: pd.DataFrame, stem_form: pd.DataFrame, plot_area_m2: pd.Series,
                 plot_width: pd.Series, plot_depth: pd.Series, *, dbh_plausible_mm, standing_height_dm_above: float,
                 competition_radius_m: float, species_groups: dict) -> pd.DataFrame:
    """tree_profile_all: everything known about a tree in one row.

    struct: structure_per_tree.csv. trees: field_trees_detection.csv. stem_form: stem_form_per_tree.csv.
    plot_area_m2, plot_width, plot_depth: Series indexed by integer plot id.
    """
    assert struct['tree_id'].equals(trees['tree_id'])
    profile = struct.merge(
        trees[['tree_id', 'instance', 'local_x', 'local_y', 'block', 'has_height', 'height_vertex', 'height_massband',
               'height_standing_sheet', 'dbh_2020_mm', 'dbh_year']], on='tree_id', how='left'
    ).merge(stem_form, on='tree_id', how='left')
    assert profile['tree_id'].equals(struct['tree_id'])

    # Field height where one was measured: felled length first, else the height taken on the standing tree
    standing = profile['height_standing_sheet'].where(profile['height_standing_sheet'] <= standing_height_dm_above,
                                                      profile['height_standing_sheet'] / 10)
    profile['field_height_m'] = profile[['height_vertex', 'height_massband']].mean(axis=1).fillna(standing)

    # Diameter growth since the 2020 inventory
    dbh_2020 = profile['dbh_2020_mm'].where(profile['dbh_2020_mm'].between(*dbh_plausible_mm))
    profile['dbh_increment_mm_yr'] = (profile['dbh_mm'] - dbh_2020) / (profile['dbh_year'] - 2020)

    # Neighbourhood and position
    stem_xy = trees[['utm_x', 'utm_y']].to_numpy()
    stem_index = cKDTree(stem_xy)
    profile['stems_within_5m'] = stem_index.query_ball_point(stem_xy, competition_radius_m, return_length=True) - 1
    profile['parzelle_stems_per_ha'] = profile['parzelle'].map(struct.groupby('parzelle').size() / (plot_area_m2 / 1e4))
    width, depth = profile['parzelle'].map(plot_width), profile['parzelle'].map(plot_depth)
    profile['edge_dist_m'] = np.minimum.reduce(
        [profile['local_x'], width - profile['local_x'], profile['local_y'], depth - profile['local_y']]).clip(min=0)

    # Which instance do the LiDAR points of the stem at the field position belong to?
    own_instance = pd.to_numeric(profile['instance'], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    matched_instances = pd.to_numeric(trees['instance'], errors='coerce').dropna().astype(int)
    other_tree = profile['stem_instance'].isin(matched_instances)
    profile['stem_owner'] = pd.Categorical(np.select(
        [profile['stem_slices'] == 0, profile['stem_instance'] == 0, profile['stem_instance'].to_numpy() == own_instance, other_tree],
        [STEM_OWNERS[4], STEM_OWNERS[3], STEM_OWNERS[0], STEM_OWNERS[1]], default=STEM_OWNERS[2]), categories=STEM_OWNERS)

    profile['species_group'] = profile['species'].map(species_groups).fillna('other')
    return profile.drop(columns=['height_vertex', 'height_massband', 'height_standing_sheet'])


def missed_trees(profile: pd.DataFrame) -> pd.DataFrame:
    """Profile rows of the trees that are not detected, sorted by class, case, plot and tree."""
    missed = profile[profile['detection'] != 'detected']
    # classes sort in the order of the matching (uncertain before not detected), not alphabetically
    order = missed['detection'].map({name: i for i, name in enumerate(DETECTION_CLASSES)})
    return missed.assign(_order=order).sort_values(['_order', 'case', 'parzelle', 'tree_id']).drop(columns='_order')


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stems(cfg) -> None:
    """Write column_structure_per_tree.csv, stem_form_per_tree.csv, tree_profile_all.csv, missed_trees_profile.csv."""
    corners = read_corners(cfg)        # Excel first
    from core import instances         # owned by another module; imported late

    detection_path = cfg.table('field_trees_detection.csv')
    struct_path = cfg.table('structure_per_tree.csv')
    canopy_path = cfg.table('canopy_per_parzelle.csv')
    cloud_files = [cfg.paths.sat_before.path]
    trees = pd.read_csv(detection_path, dtype={'tree_id': str})
    memo = {}

    def cloud():
        if 'cloud' not in memo:
            memo['cloud'] = instances.load_before_cloud(cfg)
        return memo['cloud']

    cached_table(cfg, 'stem_form_per_tree.csv', ['grid'], [detection_path] + cloud_files,
                 lambda: stem_form_table(cloud(), trees))
    cached_table(cfg, 'column_structure_per_tree.csv', ['grid'], [detection_path] + cloud_files,
                 lambda: column_structure(cloud(), field_positions(trees)))

    stem_form_path = cfg.table('stem_form_per_tree.csv')
    width, depth = plot_extent(corners)

    def compute_profile():
        struct = pd.read_csv(struct_path, dtype={'tree_id': str})
        stem_form = pd.read_csv(stem_form_path, dtype={'tree_id': str})
        area = pd.read_csv(canopy_path).set_index('parzelle')['area_m2']
        return tree_profile(
            struct, trees, stem_form, area, width, depth, dbh_plausible_mm=cfg.cleaning.dbh_plausible_mm,
            standing_height_dm_above=cfg.cleaning.standing_height_dm_above,
            competition_radius_m=cfg.structure.competition_radius_m,
            species_groups=cfg.structure.species_groups.to_dict())

    deps = ['structure', 'cleaning', 'field_columns']
    inputs = [detection_path, struct_path, canopy_path, stem_form_path, cfg.paths.plot_corners]
    profile = cached_table(cfg, 'tree_profile_all.csv', deps, inputs, compute_profile)
    cached_table(cfg, 'missed_trees_profile.csv', deps, inputs + [cfg.table('tree_profile_all.csv')],
                 lambda: missed_trees(pd.read_csv(cfg.table('tree_profile_all.csv'), dtype={'tree_id': str})))


def run_dbh(cfg) -> None:
    """Write height_dbh_all_trees.csv (LiDAR diameter at breast height; no photogrammetry)."""
    from core import instances

    detection_path = cfg.table('field_trees_detection.csv')

    def compute():
        trees = pd.read_csv(detection_path, dtype={'tree_id': str})
        return dbh_table(instances.load_before_cloud(cfg), trees, breast_height_m=cfg.dbh.breast_height_m,
                         slice_half_height_m=cfg.dbh.slice_half_height_m, search_radius_m=cfg.dbh.search_radius_m,
                         ellipse_axis_ratio=cfg.dbh.ellipse_axis_ratio)

    cached_table(cfg, 'height_dbh_all_trees.csv', ['dbh', 'grid'], [detection_path, cfg.paths.sat_before.path], compute)
