"""Stem-seeded refinement of single tree instances.

For a field tree whose SegmentAnyTree instance is wrong (typically a small tree
under a tall canopy that got a piece of a neighbour's crown), the tree is rebuilt
from its stem:

1. trace the lower stem upward from the stem base, as far as it is hit by returns,
2. extend that axis as a straight line (the thin upper stem and the inside of a
   shaded crown are usually not hit at all),
3. give every point to the tree whose stem it is connected to by the shortest
   path through the point cloud: the neighbours' stems compete for the points,
4. point clusters that are connected to no stem at all go to the tree if they lie
   around its extended axis (branch whorls seen only at their tips),
5. the tree ends at a gap in the heights of its points. If a field height is
   given, the gap closest to it is taken.

Field data used: stem position (via the matched instance base), stem diameter,
and optionally the height for step 5. A height obtained with the field height is
not an independent LiDAR measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import laspy
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .field_tree_matching import sample_raster_bilinear
from .io_utils import get_logger

logger = get_logger("tree_height.stem_seeded_refinement")


# ---------------------------------------------------------------------------
# Point cloud access
# ---------------------------------------------------------------------------

def load_points_near(
    laz_path: Path | str,
    centres: Sequence[tuple[float, float]],
    radius_m: float,
    dtm_path: Path | str,
    instance_dim: str = "PredInstance",
    chunk_size: int = 10_000_000,
) -> pd.DataFrame:
    """Read only the points within `radius_m` (square window) of any centre, without loading the file whole.

    Returns x, y, z, h (height above the DTM) and instance.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    index = cKDTree(centres)
    lo, hi = centres.min(axis=0) - radius_m, centres.max(axis=0) + radius_m
    parts = []
    with laspy.open(laz_path) as reader:
        names = {d.lower(): d for d in reader.header.point_format.dimension_names}
        dim = names[instance_dim.lower()]
        for points in reader.chunk_iterator(chunk_size):
            x, y = np.asarray(points.x), np.asarray(points.y)
            keep = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
            if keep.any():
                # square window around the nearest centre (Chebyshev distance)
                dist, _ = index.query(np.column_stack([x[keep], y[keep]]), p=np.inf, distance_upper_bound=radius_m + 1e-9)
                keep[np.flatnonzero(keep)] = np.isfinite(dist)
            if keep.any():
                parts.append(pd.DataFrame({
                    "x": x[keep], "y": y[keep], "z": np.asarray(points.z)[keep],
                    "instance": np.asarray(points[dim])[keep].astype(np.int32)}))
    cloud = pd.concat(parts, ignore_index=True)
    cloud["h"] = cloud["z"] - sample_raster_bilinear(dtm_path, cloud["x"].to_numpy(), cloud["y"].to_numpy())
    logger.info("Loaded %d points around %d trees from %s", len(cloud), len(centres), Path(laz_path).name)
    return cloud


# ---------------------------------------------------------------------------
# Stem axis
# ---------------------------------------------------------------------------

def trace_stem_axis(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    seed_xy: tuple[float, float],
    layer_m: float = 0.5,
    start_height_m: float = 0.5,
    search_radius_m: float = 0.3,
    min_points: int = 5,
    max_gap_m: float = 1.5,
    max_lean_m_per_m: float = 0.5,
    max_height_m: float = 45.0,
) -> pd.DataFrame:
    """Follow a stem upward in height layers, as far as it is hit by returns.

    In each layer the axis position is predicted from the layer below and the
    current lean; the points within `search_radius_m` of the prediction move
    the axis to their median. Layers with fewer than `min_points` are bridged
    by extrapolating the lean; the trace ends when more than `max_gap_m` in a
    row has to be bridged, or at `max_height_m`.

    Returns one row per layer up to the last supported one: h_low, h_high, x, y,
    n (points near the axis), supported.
    """
    sel = (np.abs(x - seed_xy[0]) < 8.0) & (np.abs(y - seed_xy[1]) < 8.0) & (h >= start_height_m)
    x, y, h = x[sel], y[sel], h[sel]

    centre = np.array(seed_xy, dtype=float)
    lean = np.zeros(2)                       # horizontal shift per layer
    gap, rows = 0.0, []
    for h_low in np.arange(start_height_m, max_height_m, layer_m):
        layer = (h >= h_low) & (h < h_low + layer_m)
        predicted = centre + lean
        near = np.hypot(x[layer] - predicted[0], y[layer] - predicted[1]) < search_radius_m
        supported = int(near.sum()) >= min_points
        if supported:
            new = np.array([np.median(x[layer][near]), np.median(y[layer][near])])
            lean = 0.5 * lean + 0.5 * (new - centre) / (1.0 + gap / layer_m)
            limit = max_lean_m_per_m * layer_m
            if np.hypot(*lean) > limit:
                lean *= limit / np.hypot(*lean)
            centre, gap = new, 0.0
        else:
            centre, gap = predicted, gap + layer_m
        rows.append({"h_low": h_low, "h_high": h_low + layer_m, "x": centre[0], "y": centre[1],
                     "n": int(near.sum()), "supported": supported})
        if gap > max_gap_m:
            break

    axis = pd.DataFrame(rows)
    if axis["supported"].any():
        axis = axis.iloc[: int(np.flatnonzero(axis["supported"].to_numpy())[-1]) + 1]
    return axis.reset_index(drop=True)


def extend_axis(axis: pd.DataFrame, to_height_m: float, layer_m: float = 0.5) -> pd.DataFrame:
    """Continue a traced axis as a straight line up to `to_height_m`.

    The direction is fitted to the supported layers of the trace (weighted by
    their point counts); the line starts at the last traced position. Added rows
    have supported = False and n = 0.
    """
    sup = axis[axis["supported"]]
    mid = ((sup["h_low"] + sup["h_high"]) / 2).to_numpy()
    if len(sup) >= 3:
        weights = np.sqrt(sup["n"].to_numpy())
        slope_x = np.polyfit(mid, sup["x"].to_numpy(), 1, w=weights)[0]
        slope_y = np.polyfit(mid, sup["y"].to_numpy(), 1, w=weights)[0]
    else:
        slope_x = slope_y = 0.0
    last_h = float(axis["h_high"].iloc[-1]) - layer_m / 2
    x0, y0 = float(axis["x"].iloc[-1]), float(axis["y"].iloc[-1])
    rows = [{"h_low": h_low, "h_high": h_low + layer_m,
             "x": x0 + slope_x * (h_low + layer_m / 2 - last_h), "y": y0 + slope_y * (h_low + layer_m / 2 - last_h),
             "n": 0, "supported": False}
            for h_low in np.arange(float(axis["h_high"].iloc[-1]), to_height_m, layer_m)]
    return pd.concat([axis, pd.DataFrame(rows)], ignore_index=True) if rows else axis


def axis_position(axis: pd.DataFrame, heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis x, y at the given heights (linear between layer centres; NaN outside the axis range)."""
    mid = (axis["h_low"].to_numpy() + axis["h_high"].to_numpy()) / 2
    inside = (heights >= axis["h_low"].iloc[0]) & (heights <= axis["h_high"].iloc[-1])
    ax_x = np.where(inside, np.interp(heights, mid, axis["x"].to_numpy()), np.nan)
    ax_y = np.where(inside, np.interp(heights, mid, axis["y"].to_numpy()), np.nan)
    return ax_x, ax_y


# ---------------------------------------------------------------------------
# Refinement of one tree
# ---------------------------------------------------------------------------

def refine_tree_from_stem(
    cloud: pd.DataFrame,
    seed_xy: tuple[float, float],
    dbh_mm: float,
    competitor_seeds: Optional[pd.DataFrame] = None,
    expected_height_m: Optional[float] = None,
    crown_base_m: Optional[float] = None,
    search_above_m: float = 4.0,
    work_radius_m: float = 6.0,
    voxel_m: float = 0.04,
    link_m: float = 0.25,
    max_crown_radius_m: float = 3.5,
    orphan_radius_m: float = 2.0,
    top_gap_m: float = 0.5,
    stray_radius_m: float = 0.3,
    stray_min_neighbours: int = 4,
    min_height_m: float = 0.3,
    **trace_kwargs,
) -> dict[str, object]:
    """Rebuild one tree from its stem base.

    Parameters
    ----------
    cloud : DataFrame with x, y, h (height above ground); from load_points_near().
    seed_xy : stem base (the base of the matched instance, or a stem found at the field position).
    dbh_mm : field stem diameter; sets how far from the axis a point still counts as stem.
    competitor_seeds : DataFrame with x, y, dbh_mm of the neighbouring stems. Their
        stems are traced as well and compete for the points.
    expected_height_m : field height. The point cloud is searched up to this height
        plus `search_above_m`, and of the gaps in the heights of the tree's points
        the one closest to the field height becomes the top. Without it the search
        goes twice `search_above_m` above the traced stem and the first gap is the top.
    max_crown_radius_m : limit for points connected to the tree's own stem.
    orphan_radius_m : limit for point clusters connected to no stem.
    crown_base_m : field crown base. Clusters connected to no stem are only taken
        from 0.5 m below it upward (below the crown a tree has no branches that
        could show up as separate clusters). Without it they are taken above 2 m.
    top_gap_m : height range without points that counts as a gap.
    stray_radius_m, stray_min_neighbours : a point with fewer neighbours within this
        radius is a stray return and is dropped, so that two or three isolated
        points cannot bridge a gap or become the top.

    Returns
    -------
    dict with
      'mask'      : boolean array over `cloud`, True for the points of the refined tree
      'connected' : subset of 'mask' that is connected to the tree's own stem
      'axis'      : traced lower stem plus its straight extension (supported = False)
      'metrics'   : height, x_top, y_top, x_base, y_base, lean_m, traced_stem_top_m,
                    crown_radius_m, n_points, n_connected, n_orphan (clusters taken by
                    position only), lowest_branch_m, candidate_tops, top_chosen_by ('field_height' / 'first_gap' /
                    'search_limit' if no gap was found below the search ceiling)
    """
    stem_radius = (dbh_mm / 2000.0) if np.isfinite(dbh_mm) else 0.15
    use_field_height = expected_height_m is not None and np.isfinite(expected_height_m)
    x_all, y_all, h_all = cloud["x"].to_numpy(), cloud["y"].to_numpy(), cloud["h"].to_numpy()

    traced = trace_stem_axis(x_all, y_all, h_all, seed_xy, **trace_kwargs)
    if not traced["supported"].any():
        raise ValueError("No stem points found at the seed position.")
    traced_top = float(traced["h_high"].iloc[-1])
    ceiling = (expected_height_m + search_above_m) if use_field_height else (traced_top + 2 * search_above_m)
    axis = extend_axis(traced, ceiling)

    # Working set: thinned points around the tree, up to the search ceiling
    work = np.flatnonzero((np.abs(x_all - seed_xy[0]) <= work_radius_m) & (np.abs(y_all - seed_xy[1]) <= work_radius_m)
                          & (h_all >= min_height_m) & (h_all <= ceiling))
    voxel = np.floor(np.column_stack([x_all[work], y_all[work], h_all[work]]) / voxel_m).astype(np.int64)
    _, first, inverse = np.unique(voxel, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    px, py, ph = x_all[work][first], y_all[work][first], h_all[work][first]

    def off_axis(ax: pd.DataFrame) -> np.ndarray:
        ax_x, ax_y = axis_position(ax, ph)
        return np.hypot(px - ax_x, py - ax_y)

    # Sources: points on each tree's own traced stem. Label 0 = the tree to refine.
    label_of_source = np.full(len(px), -1, dtype=np.int64)
    if competitor_seeds is not None:
        for k, comp in enumerate(competitor_seeds.itertuples(index=False), start=1):
            comp_radius = (comp.dbh_mm / 2000.0) if np.isfinite(comp.dbh_mm) else 0.25
            comp_axis = trace_stem_axis(x_all, y_all, h_all, (comp.x, comp.y), search_radius_m=comp_radius + 0.25,
                                        max_gap_m=3.0, max_height_m=ceiling)
            if comp_axis["supported"].any():
                with np.errstate(invalid="ignore"):
                    label_of_source[off_axis(comp_axis) < comp_radius + 0.15] = k
    with np.errstate(invalid="ignore"):
        label_of_source[off_axis(traced) < stem_radius + 0.15] = 0      # the target wins on its own stem

    # Shortest paths through the cloud from all stems at once
    pairs = cKDTree(np.column_stack([px, py, ph])).query_pairs(link_m, output_type="ndarray")
    length = np.sqrt((px[pairs[:, 0]] - px[pairs[:, 1]]) ** 2 + (py[pairs[:, 0]] - py[pairs[:, 1]]) ** 2
                     + (ph[pairs[:, 0]] - ph[pairs[:, 1]]) ** 2)
    graph = coo_matrix((length, (pairs[:, 0], pairs[:, 1])), shape=(len(px), len(px))).tocsr()
    sources = np.flatnonzero(label_of_source >= 0)
    dist, _, nearest_source = dijkstra(graph, directed=False, indices=sources, min_only=True, return_predecessors=True)
    reached = np.isfinite(dist) & (nearest_source >= 0)
    label = np.full(len(px), -1, dtype=np.int64)
    label[reached] = label_of_source[nearest_source[reached]]

    with np.errstate(invalid="ignore"):
        radial = off_axis(axis)
        connected = (label == 0) & (radial <= max_crown_radius_m)
        orphan_from = (crown_base_m - 0.5) if (crown_base_m is not None and np.isfinite(crown_base_m)) else 2.0
        orphan = (label == -1) & (radial <= orphan_radius_m) & (ph > orphan_from)   # connected to no stem, around the axis
    candidate = connected | orphan

    # Drop stray returns
    cand_idx = np.flatnonzero(candidate)
    cand_xyz = np.column_stack([px[cand_idx], py[cand_idx], ph[cand_idx]])
    neighbours = cKDTree(cand_xyz).query_ball_point(cand_xyz, stray_radius_m, return_length=True) - 1
    candidate[cand_idx[neighbours < stray_min_neighbours]] = False

    # Top: a gap in the heights of the candidate points
    heights = np.sort(ph[candidate])
    gap_after = np.flatnonzero(np.diff(heights) > top_gap_m)
    candidate_tops = [float(heights[i]) for i in gap_after] + [float(heights[-1])]
    if use_field_height:
        top_height = min(candidate_tops, key=lambda t: abs(t - expected_height_m))
    else:
        above = [t for t in candidate_tops if t >= traced_top - layer_tolerance(trace_kwargs)]
        top_height = above[0] if above else candidate_tops[-1]
    mine = candidate & (ph <= top_height + 1e-6)

    mask = np.zeros(len(cloud), dtype=bool)
    mask[work] = mine[inverse]
    connected_full = np.zeros(len(cloud), dtype=bool)
    connected_full[work] = (connected & mine)[inverse]

    hx, hy, hh = x_all[mask], y_all[mask], h_all[mask]
    top = int(np.argmax(hh))
    ax_x, ax_y = axis_position(axis, hh)
    upper = hh > 0.5 * hh[top]
    metrics = {
        "height": float(hh[top]), "x_top": float(hx[top]), "y_top": float(hy[top]),
        "x_base": float(axis["x"].iloc[0]), "y_base": float(axis["y"].iloc[0]),
        "lean_m": float(np.hypot(hx[top] - axis["x"].iloc[0], hy[top] - axis["y"].iloc[0])),
        "traced_stem_top_m": traced_top,
        "crown_radius_m": float(np.nanpercentile(np.hypot(hx - ax_x, hy - ax_y)[upper], 95)) if upper.any() else np.nan,
        "n_points": int(mask.sum()),
        "n_connected": int(connected_full.sum()),
        "n_orphan": int(mask.sum() - connected_full.sum()),
        # lowest branch: 2nd percentile of the heights of points clearly off the stem (compare with the field crown base)
        "lowest_branch_m": (float(np.percentile(hh[np.hypot(hx - ax_x, hy - ax_y) > stem_radius + 0.4], 2))
                            if (np.hypot(hx - ax_x, hy - ax_y) > stem_radius + 0.4).any() else np.nan),
        "candidate_tops": [round(t, 2) for t in candidate_tops],
        "top_chosen_by": ("field_height" if use_field_height
                          else "search_limit" if top_height > ceiling - top_gap_m else "first_gap"),
    }
    logger.info("Refined tree: height %.2f m (%s), %d points (%d connected to the stem, %d by position)",
                metrics["height"], metrics["top_chosen_by"], metrics["n_points"], metrics["n_connected"], metrics["n_orphan"])
    return {"mask": mask, "connected": connected_full, "axis": axis, "metrics": metrics}


def extend_top_into_canopy(
    cloud: pd.DataFrame,
    mask: np.ndarray,
    layer_m: float = 0.25,
    core_radius_m: float = 1.0,
    ring_radius_m: float = 2.5,
    min_concentration: float = 3.0,
    min_points: int = 5,
    max_miss_m: float = 0.5,
    max_extension_m: float = 4.0,
    start_band_m: float = 2.0,
    stray_radius_m: float = 0.3,
    stray_min_neighbours: int = 4,
) -> dict[str, object]:
    """Close the gap between a refined small tree and the crown base of the canopy above it.

    A small tree under a tall neighbour often ends where the neighbour's lower
    crown begins; the shortest-path assignment then gives the small tree's top to
    the neighbour, and the first height gap stops the tree too low. Above the
    refined top, the points are followed upward in thin layers inside a cylinder
    (`core_radius_m`) around the centre of the tree's upper crown. A layer belongs
    to the small tree while its points are concentrated in that cylinder: their
    density there is at least `min_concentration` times the density in the ring
    out to `ring_radius_m`. The crown base of the neighbour shows up as an evenly
    spread layer, and the search ends after `max_miss_m` of such layers in a row.
    The centre follows the accepted points, so a leaning top can be followed.
    The field height is not used.

    Returns dict with 'mask' (refined tree plus the added top), 'added' (the added
    points only), 'layers' (one row per layer: h_low, n_core, n_ring,
    concentration, accepted) and 'metrics' (height_before, height, n_added,
    canopy_base_m: lowest layer of the run that ended the search).
    """
    x, y, h = cloud["x"].to_numpy(), cloud["y"].to_numpy(), cloud["h"].to_numpy()
    own_top = float(h[mask].max())
    upper = mask & (h >= own_top - start_band_m)
    centre = np.array([x[upper].mean(), y[upper].mean()])
    near = np.flatnonzero((np.abs(x - centre[0]) <= ring_radius_m + 2.0) & (np.abs(y - centre[1]) <= ring_radius_m + 2.0)
                          & (h >= own_top - layer_m) & (h <= own_top + max_extension_m))

    added = np.zeros(len(cloud), dtype=bool)
    miss, miss_start, rows = 0.0, np.nan, []
    core_area, ring_area = np.pi * core_radius_m ** 2, np.pi * (ring_radius_m ** 2 - core_radius_m ** 2)
    for h_low in np.arange(np.floor(own_top / layer_m) * layer_m, own_top + max_extension_m, layer_m):
        idx = near[(h[near] >= h_low) & (h[near] < h_low + layer_m)]
        r = np.hypot(x[idx] - centre[0], y[idx] - centre[1])
        core, n_ring = r < core_radius_m, int(((r >= core_radius_m) & (r < ring_radius_m)).sum())
        concentration = (core.sum() / core_area) / (n_ring / ring_area) if n_ring else np.inf
        accepted = int(core.sum()) >= min_points and concentration >= min_concentration
        rows.append({"h_low": h_low, "n_core": int(core.sum()), "n_ring": n_ring,
                     "concentration": concentration, "accepted": accepted})
        if accepted:
            added[idx[core & ~mask[idx]]] = True
            centre = 0.5 * centre + 0.5 * np.array([np.median(x[idx[core]]), np.median(y[idx[core]])])
            miss, miss_start = 0.0, np.nan
        else:
            miss_start = h_low if miss == 0 else miss_start
            miss += layer_m
            if miss >= max_miss_m:
                break

    # Drop stray returns among the added points, so that a few isolated points cannot become the top
    cand = np.flatnonzero(added | mask)
    if added.any():
        xyz = np.column_stack([x[cand], y[cand], h[cand]])
        new = added[cand]
        counts = cKDTree(xyz).query_ball_point(xyz[new], stray_radius_m, return_length=True) - 1
        added[cand[new][counts < stray_min_neighbours]] = False
    refined = mask | added
    metrics = {"height_before": own_top, "height": float(h[refined].max()), "n_added": int(added.sum()),
               "canopy_base_m": float(miss_start)}
    logger.info("Top extended from %.2f m to %.2f m (%d points added); canopy above from %.2f m",
                own_top, metrics["height"], metrics["n_added"], metrics["canopy_base_m"])
    return {"mask": refined, "added": added, "layers": pd.DataFrame(rows), "metrics": metrics}


def layer_tolerance(trace_kwargs: dict) -> float:
    """One trace layer: a top just below the last traced layer still counts as above it."""
    return float(trace_kwargs.get("layer_m", 0.5))


def write_refined_tree(cloud: pd.DataFrame, mask: np.ndarray, path: Path | str) -> Path:
    """Save the refined tree as LAZ (x, y, z and height_above_ground), same naming as the hand-segmented trees."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = cloud[mask]
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([np.floor(pts["x"].min()), np.floor(pts["y"].min()), 0.0])
    header.add_extra_dim(laspy.ExtraBytesParams(name="height_above_ground", type=np.float32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts["x"].to_numpy(), pts["y"].to_numpy(), pts["z"].to_numpy(dtype=np.float64)
    las["height_above_ground"] = pts["h"].to_numpy(dtype=np.float32)
    las.write(path)
    return path


# ---------------------------------------------------------------------------
# QA figure
# ---------------------------------------------------------------------------

def plot_refinement(
    cloud: pd.DataFrame,
    mask: np.ndarray,
    axis: pd.DataFrame,
    tree: pd.Series,
    metrics: dict[str, object],
    output_path: Optional[Path | str] = None,
    connected: Optional[np.ndarray] = None,
    half_width_m: float = 5.0,
    slab_half_m: float = 0.6,
):
    """Two vertical slabs through the refined tree (along and across its lean) and a plan view.

    Dark blue: points connected to the tree's own stem. Light blue: clusters taken
    because they lie around the extended axis (pass `connected` to tell them apart).
    Red: the SegmentAnyTree instance that was matched to the field tree. Grey:
    everything else. Black line: traced stem (solid) and its straight extension
    (dotted). Dashed lines: refined LiDAR height, field heights, SAT height.
    """
    import matplotlib.pyplot as plt

    if connected is None:
        connected = mask
    x, y, h = cloud["x"].to_numpy(), cloud["y"].to_numpy(), cloud["h"].to_numpy()
    base = np.array([metrics["x_base"], metrics["y_base"]])
    direction = np.array([metrics["x_top"], metrics["y_top"]]) - base
    direction = direction / np.hypot(*direction) if np.hypot(*direction) > 0.3 else np.array([1.0, 0.0])
    normal = np.array([-direction[1], direction[0]])
    rel = np.column_stack([x - base[0], y - base[1]])
    along, across = rel @ direction, rel @ normal
    has_instance = pd.notna(tree.get("instance"))
    sat = (cloud["instance"].to_numpy() == int(tree["instance"])) if has_instance else np.zeros(len(x), dtype=bool)
    ax_rel = np.column_stack([axis["x"] - base[0], axis["y"] - base[1]])
    ax_h = ((axis["h_low"] + axis["h_high"]) / 2).to_numpy()
    shown_axis = ax_h <= metrics["height"] + 0.25
    field_max = np.nanmax([tree.get("height_vertex", np.nan), tree.get("height_massband", np.nan),
                           tree.get("height_standing", np.nan), 0.0])
    y_max = max(metrics["height"], float(field_max)) + 8

    fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1, 1, 0.95]})
    for ax, u, v, ax_u, ax_v, title in [
        (axes[0], along, across, ax_rel @ direction, ax_rel @ normal, "Slab along the lean"),
        (axes[1], across, along, ax_rel @ normal, ax_rel @ direction, "Slab across the lean"),
    ]:
        v_axis = np.interp(h, ax_h, ax_v)                # the slab follows the axis
        in_slab = (np.abs(v - v_axis) <= slab_half_m) & (np.abs(u) <= half_width_m) & (h <= y_max)
        sat_label = f"SAT instance {int(tree['instance'])}" if has_instance else "SAT instance: none"
        for sel, color, size, label in [(in_slab & ~mask & ~sat, "0.72", 0.5, "other points"),
                                        (in_slab & sat & ~mask, "#d03b3b", 1.2, sat_label),
                                        (in_slab & mask & ~connected, "#6da7ec", 1.2, "refined tree, by position"),
                                        (in_slab & mask & connected, "#184f95", 1.2, "refined tree, connected to stem")]:
            ax.scatter(u[sel], h[sel], s=size, color=color, label=label, rasterized=True)
        for supported, style in [(True, "-"), (False, ":")]:
            seg = np.where((axis["supported"].to_numpy() == supported) & shown_axis, ax_u, np.nan)
            ax.plot(seg, ax_h, color="black", linewidth=1.0, linestyle=style)
        ax.axhline(metrics["height"], color="#184f95", linewidth=1.1, linestyle="--", label=f"refined {metrics['height']:.1f} m")
        for col, color, name in [("height_vertex", "#0b0b0b", "Vertex"), ("height_massband", "#008300", "Maßband"),
                                 ("height_standing", "#eb6834", "standing height")]:
            if pd.notna(tree.get(col)):
                ax.axhline(tree[col], color=color, linewidth=1.0, linestyle="--", label=f"{name} {tree[col]:.1f} m")
        if pd.notna(tree.get("lidar_height")) and tree["lidar_height"] <= y_max:
            ax.axhline(tree["lidar_height"], color="#d03b3b", linewidth=1.0, linestyle="--", label=f"SAT {tree['lidar_height']:.1f} m")
        ax.set_xlim(-half_width_m, half_width_m)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("distance from stem base (m)")
        ax.set_ylabel("height above ground (m)")
        ax.set_title(f"{title} ({2 * slab_half_m:.1f} m wide)", loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8, markerscale=6)

    ax = axes[2]
    low = (np.abs(rel[:, 0]) <= half_width_m) & (np.abs(rel[:, 1]) <= half_width_m) & (h > 1.0) & (h <= metrics["height"] + 0.5)
    ax.scatter(rel[low & ~mask, 0], rel[low & ~mask, 1], s=0.4, color="0.75", rasterized=True)
    sc = ax.scatter(rel[low & mask, 0], rel[low & mask, 1], s=1.5, c=h[low & mask], cmap="viridis", rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="height of the refined tree's points (m)")
    ax.plot(ax_rel[shown_axis, 0], ax_rel[shown_axis, 1], color="black", linewidth=1.0)
    ax.plot(0, 0, "+", color="black", markersize=12, markeredgewidth=2, label="stem base")
    if pd.notna(tree.get("utm_x")):
        ax.plot(tree["utm_x"] - base[0], tree["utm_y"] - base[1], "x", color="#d03b3b", markersize=10, markeredgewidth=2, label="field position")
    ax.set_xlim(-half_width_m, half_width_m)
    ax.set_ylim(-half_width_m, half_width_m)
    ax.set_aspect("equal")
    ax.set_xlabel("east of stem base (m)")
    ax.set_ylabel("north of stem base (m)")
    ax.set_title("Plan view, points between 1 m and the tree top", loc="left", fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"Tree {tree['tree_id']}: stem-seeded refinement  |  refined {metrics['height']:.2f} m, {metrics['n_points']:,} points "
                 f"({metrics['n_connected']:,} connected to the stem)  |  stem traced to {metrics['traced_stem_top_m']:.1f} m, "
                 f"top chosen by {str(metrics['top_chosen_by']).replace('_', ' ')}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig
