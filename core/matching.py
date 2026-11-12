"""Matching of field stems to LiDAR tree instances, detection classes, instances without a field tree.

Pure functions take every threshold as an argument; the stage runners `run_matching(cfg)` and
`run_unmatched(cfg)` read the config and the stage inputs and write the tables.

Tree height never selects or verifies a match: the cost of a pair is the distance between the
stem and the instance base, plus a penalty when the harvest evidence disagrees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

CLASSES = ["detected", "uncertain", "not detected"]
RECORD = {True: "marked for removal 2026", False: "not marked"}
_CONFIDENCE_TO_CLASS = {"high": "detected", "check": "uncertain", "unmatched": "not detected"}


# ---------------------------------------------------------------------------
# The matcher
# ---------------------------------------------------------------------------

def match_stems_to_instances(
    field: pd.DataFrame,
    instances: pd.DataFrame,
    max_distance_m: float,
    removed_threshold: float,
    removal_penalty_m: float,
    high_confidence_distance_m: float,
    height_gate_rel: float,
) -> pd.DataFrame:
    """Assign each field stem to at most one LiDAR instance (global optimum, one-to-one).

    Cost of a pair = distance stem -> instance base, plus `removal_penalty_m` when the harvest
    evidence disagrees (tree recorded as felled but instance still present afterwards, or vice
    versa). Field heights do NOT enter the cost; they only set `height_gate_ok`, which lowers the
    confidence class from 'high' to 'check'.

    field: tree_id, utm_x, utm_y, removed_2026 and optionally height_vertex / height_massband.
    instances: instance, x_base, y_base, x_top, y_top, height, removed_fraction, lean_m, crown_base_m.

    Returns one row per field tree: tree_id, instance, base_dist_m, runner_up_instance,
    runner_up_dist_m, n_candidates, removal_consistent, lidar_height, x_top, y_top, x_base, y_base,
    removed_fraction, lean_m, crown_base_lidar_m, field_height, height_gate_ok, confidence, method.
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

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inventory helpers (only used when the inventory table lacks these columns)
# ---------------------------------------------------------------------------

def add_block(field: pd.DataFrame, plot_column: str = "parzelle") -> pd.DataFrame:
    """Label per inventory sheet: 'Parzellen <first>-<last>' of the plots of each `source_file`."""
    if "block" in field.columns:
        return field
    plots = field[plot_column].astype(int)
    first = plots.groupby(field["source_file"]).transform("min")
    last = plots.groupby(field["source_file"]).transform("max")
    return field.assign(block="Parzellen " + first.astype(str) + "-" + last.astype(str))


def add_dbh_class(field: pd.DataFrame, plausible_mm, edges_mm) -> pd.DataFrame:
    """DBH class from the plausible DBH only; the rest is 'no / implausible DBH'."""
    if "dbh_class" in field.columns:
        return field
    edges = list(edges_mm)
    labels = [f"< {edges[1]:g} mm"] + [f"{a:g}-{b:g} mm" for a, b in zip(edges[1:-2], edges[2:-1])] + [f"> {edges[-2]:g} mm"]
    ok = field["dbh_mm"].between(*plausible_mm)
    cls = pd.cut(field["dbh_mm"].where(ok), edges, labels=labels)
    cls = cls.cat.add_categories("no / implausible DBH").fillna("no / implausible DBH")
    return field.assign(dbh_class=cls)


# ---------------------------------------------------------------------------
# Detection classes and tables
# ---------------------------------------------------------------------------

def detection_classes(field: pd.DataFrame, matches: pd.DataFrame, high_confidence_distance_m: float, max_distance_m: float) -> pd.DataFrame:
    """Join the matching to the inventory and set `detection` and `uncertain_reason`."""
    result = field.merge(matches.drop(columns=["field_height", "height_gate_ok"]), on="tree_id", how="left")
    result["detection"] = result["confidence"].map(_CONFIDENCE_TO_CLASS)
    result["detection"] = pd.Categorical(result["detection"], categories=CLASSES)
    result["uncertain_reason"] = np.select(
        [result["detection"] != "uncertain",
         (result["base_dist_m"] > high_confidence_distance_m) & result["removal_consistent"].eq(False),
         result["base_dist_m"] > high_confidence_distance_m],
        ["", "far and harvest evidence disagrees", f"base {high_confidence_distance_m:g}-{max_distance_m:g} m away"],
        default="harvest evidence disagrees")
    return result


def lidar_state(removed_fraction, has_instance, gone_above: float, standing_below: float, no_instance_label: str) -> np.ndarray:
    """'gone' / 'still standing' / 'unclear' from the share of an instance's points gone after the harvest."""
    removed_fraction = np.asarray(removed_fraction, dtype=float)
    has_instance = np.broadcast_to(np.asarray(has_instance, dtype=bool), removed_fraction.shape)
    return np.select([~has_instance, removed_fraction > gone_above, removed_fraction < standing_below],
                     [no_instance_label, "gone", "still standing"], default="unclear")


def distance_only_table(field: pd.DataFrame, plain: pd.DataFrame, high_confidence_distance_m: float,
                        gone_above: float, standing_below: float) -> pd.DataFrame:
    """Distance-only assignment with the harvest record and the LiDAR state of the instance at the stem."""
    felled = plain[["tree_id", "block", "removed_2026", "instance", "base_dist_m", "removed_fraction"]].merge(
        field[["tree_id", "parzelle", "species", "dbh_mm", "remark"]], on="tree_id")
    felled["parzelle"] = felled["parzelle"].astype(int)
    felled["record"] = felled["removed_2026"].map(RECORD)
    states = [f"no instance within {high_confidence_distance_m} m", "gone", "still standing", "unclear"]
    felled["lidar"] = pd.Categorical(
        lidar_state(felled["removed_fraction"], felled["base_dist_m"] <= high_confidence_distance_m,
                    gone_above, standing_below, states[0]),
        categories=["gone", "unclear", "still standing", states[0]])
    return felled


def instances_in_plots(instances, plots, result: pd.DataFrame, removed_threshold: float):
    """Instances with their base inside a Parzelle and the field tree they are matched to.

    Returns (inst, in_plot): all instances with parzelle, tree_id, detection, in_plot, gone_after_harvest,
    and the subset inside a Parzelle.
    """
    import geopandas as gpd

    inst = gpd.sjoin(instances, plots[["parzelle", "geometry"]], predicate="within", how="left").drop(columns="index_right")
    inst = inst[~inst.index.duplicated()].copy()
    held = result.dropna(subset=["instance"])
    matched_tree = held.set_index(held["instance"].astype(int))
    inst["tree_id"] = inst["instance"].map(matched_tree["tree_id"])
    inst["detection"] = inst["instance"].map(matched_tree["detection"].astype(str))
    inst["in_plot"] = inst["parzelle"].notna()
    inst["gone_after_harvest"] = inst["removed_fraction"] > removed_threshold
    return inst, inst[inst["in_plot"]]


def rate_table(groups: pd.Series, table: pd.DataFrame) -> pd.DataFrame:
    """Counts per detection class and detection rates for a grouping of the field trees."""
    out = pd.crosstab(groups, table["detection"], dropna=False).reindex(columns=CLASSES, fill_value=0)
    out.insert(0, "field_trees", out.sum(axis=1))
    out["detection_rate"] = (out["detected"] / out["field_trees"]).round(3)
    out["incl_uncertain"] = ((out["detected"] + out["uncertain"]) / out["field_trees"]).round(3)
    return out[out["field_trees"] > 0]


def detection_per_parzelle(result: pd.DataFrame, in_plot: pd.DataFrame, corner_rmse: pd.Series) -> pd.DataFrame:
    """Field trees per class, LiDAR instances and their count ratio, per Parzelle.

    `in_plot`: LiDAR instances with their base inside a Parzelle (parzelle, instance, tree_id).
    `corner_rmse` is indexed by the Parzelle as string.
    """
    per_plot = rate_table(result["parzelle"].astype(int).rename("parzelle"), result)
    lidar_counts = in_plot.groupby(in_plot["parzelle"].astype(int)).agg(
        lidar_instances=("instance", "size"), lidar_without_field_tree=("tree_id", lambda s: int(s.isna().sum())))
    per_plot = per_plot.join(lidar_counts).fillna({"lidar_instances": 0, "lidar_without_field_tree": 0})
    per_plot[["lidar_instances", "lidar_without_field_tree"]] = per_plot[["lidar_instances", "lidar_without_field_tree"]].astype(int)
    per_plot["count_ratio"] = (per_plot["lidar_instances"] / per_plot["field_trees"]).round(2)
    per_plot["corner_rmse_m"] = per_plot.index.map(lambda p: round(corner_rmse[str(p)], 2))
    return per_plot


def detection_table(result: pd.DataFrame) -> pd.DataFrame:
    """The table written as field_trees_detection.csv (no confidence / method)."""
    table = result.copy()
    table["instance"] = table["instance"].astype("Int64")
    table["runner_up_instance"] = table["runner_up_instance"].astype("Int64")
    table["removal_consistent"] = table["removal_consistent"].astype("boolean")
    table["detection"] = table["detection"].astype(str)
    if "dbh_class" in table.columns:
        table["dbh_class"] = table["dbh_class"].astype(str)
    return table.drop(columns=["confidence", "method"])


def compute_matching(field, instances, plots, corner_rmse, *, max_distance_m, removed_threshold, removal_penalty_m,
                     high_confidence_distance_m, height_gate_rel, gone_above, standing_below,
                     control_shift_m=None) -> dict:
    """All stage-4 tables from the inventory, the instances and the plot outlines.

    Returns a dict: matches (full matcher output incl. confidence / method), result, plain, felled,
    inst, in_plot, per_plot, table.
    """
    stems = field[["tree_id", "utm_x", "utm_y", "removed_2026"]]      # no height columns: heights play no role

    def match(stem_table, penalty):
        return match_stems_to_instances(stem_table, instances, max_distance_m, removed_threshold, penalty,
                                        high_confidence_distance_m, height_gate_rel)

    plain = field[["tree_id", "block", "removed_2026"]].merge(match(stems, 0.0), on="tree_id")
    matches = match(stems, removal_penalty_m)
    if matches["instance"].dropna().duplicated().any():
        raise ValueError("An instance was assigned to more than one field tree")

    result = detection_classes(field, matches, high_confidence_distance_m, max_distance_m)
    felled = distance_only_table(field, plain, high_confidence_distance_m, gone_above, standing_below)
    inst, in_plot = instances_in_plots(instances, plots, result, removed_threshold)
    per_plot = detection_per_parzelle(result, in_plot, corner_rmse)
    out = {"matches": matches, "result": result, "plain": plain, "felled": felled, "inst": inst,
           "in_plot": in_plot, "per_plot": per_plot, "table": detection_table(result)}
    if control_shift_m is not None:
        # Chance control: all stems shifted by control_shift_m in four directions, matched with the same rules
        control = []
        for dx, dy, label in [(0, 1, "north"), (1, 0, "east"), (0, -1, "south"), (-1, 0, "west")]:
            shifted = stems.assign(utm_x=stems["utm_x"] + dx * control_shift_m, utm_y=stems["utm_y"] + dy * control_shift_m)
            control.append(match(shifted, removal_penalty_m).assign(shift=label))
        out["control"] = pd.concat(control, ignore_index=True)[
            ["tree_id", "shift", "instance", "base_dist_m", "removal_consistent", "confidence"]]
    return out


def write_detection_gpkg(path, table, inst, per_plot, plots, crs) -> None:
    """GeoPackage with the layers field_trees, lidar_instances, stem_to_base, parzellen."""
    import geopandas as gpd
    from shapely.geometry import LineString

    path = Path(path)
    if path.exists():
        path.unlink()
    gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table["utm_x"], table["utm_y"]), crs=crs).to_file(
        path, layer="field_trees", driver="GPKG")
    inst.to_file(path, layer="lidar_instances", driver="GPKG")
    linked = table[table["x_base"].notna()]
    gpd.GeoDataFrame(linked[["tree_id", "instance", "detection", "base_dist_m"]],
                     geometry=[LineString([(r.utm_x, r.utm_y), (r.x_base, r.y_base)]) for r in linked.itertuples()],
                     crs=crs).to_file(path, layer="stem_to_base", driver="GPKG")
    plots.merge(per_plot.reset_index().assign(parzelle=lambda d: d["parzelle"].astype(str)), on="parzelle").to_file(
        path, layer="parzellen", driver="GPKG")


# ---------------------------------------------------------------------------
# Instances without a field tree (section 13, without figures)
# ---------------------------------------------------------------------------

def unmatched_instances(
    inst: pd.DataFrame,
    in_plot: pd.DataFrame,
    result: pd.DataFrame,
    plots,
    stems_at_base: pd.DataFrame,
    *,
    max_distance_m: float,
    high_confidence_distance_m: float,
    min_stem_slices: int,
    edge_m: float,
    figure_count: int,
    figure_seed: int,
) -> pd.DataFrame:
    """Evidence for every instance inside a Parzelle that has no field tree.

    `stems_at_base`: stem search started at the instance base, columns tree_id (= instance id),
    stem_slices, stem_offset_m, stem_instance, stem_instance_share. Instances without a row get NaN.
    """
    from shapely.geometry import Point

    lonely = in_plot[in_plot["tree_id"].isna()].drop(columns=["geometry", "tree_id", "detection"]).copy()
    base_xy = lonely[["x_base", "y_base"]].to_numpy()

    # Nearest field stem, and whether that tree is matched to an instance
    field_xy = result[["utm_x", "utm_y"]].to_numpy()
    dist, idx = cKDTree(field_xy).query(base_xy)
    lonely["nearest_stem_m"] = dist
    lonely["nearest_tree_id"] = result["tree_id"].to_numpy()[idx]
    lonely["nearest_tree_detection"] = result["detection"].astype(str).to_numpy()[idx]
    lonely["nearest_tree_dbh_mm"] = result["dbh_mm"].to_numpy()[idx]
    lonely["field_stems_within_3m"] = cKDTree(field_xy).query_ball_point(base_xy, r=max_distance_m, return_length=True)

    # Nearest instance that has a field tree
    with_tree = inst[inst["tree_id"].notna()]
    dist, idx = cKDTree(with_tree[["x_base", "y_base"]].to_numpy()).query(base_xy)
    lonely["nearest_matched_base_m"] = dist
    lonely["nearest_matched_height"] = with_tree["height"].to_numpy()[idx]
    lonely["nearest_matched_tree_id"] = with_tree["tree_id"].to_numpy()[idx]
    lonely["nearest_matched_top_m"] = cKDTree(with_tree[["x_top", "y_top"]].to_numpy()).query(
        lonely[["x_top", "y_top"]].to_numpy())[0]

    outline = plots.set_index("parzelle").geometry.boundary
    lonely["edge_dist_m"] = [outline[p].distance(Point(x, y)) for p, x, y in zip(lonely["parzelle"], lonely["x_base"], lonely["y_base"])]

    lonely = lonely.merge(
        stems_at_base[["tree_id", "stem_slices", "stem_offset_m", "stem_instance", "stem_instance_share"]]
        .rename(columns={"tree_id": "instance"}), on="instance", how="left")
    lonely["stem_is_own"] = lonely["stem_instance"] == lonely["instance"]
    has_stem = lonely["stem_slices"] >= min_stem_slices
    lonely["reading"] = np.select(
        [has_stem & (lonely["nearest_stem_m"] > high_confidence_distance_m), has_stem,
         lonely["nearest_matched_base_m"] <= max_distance_m],
        ["own stem, not in the inventory", "own stem next to a field stem",
         "no stem, next to a matched tree: probably split off"],
        default="no stem, no matched tree nearby")
    lonely["edge"] = lonely["edge_dist_m"] < edge_m

    rng = np.random.default_rng(figure_seed)
    lonely["figure"] = False
    lonely.loc[rng.choice(lonely.index, size=min(figure_count, len(lonely)), replace=False), "figure"] = True
    return lonely


def points_near_bases(cloud: pd.DataFrame, centres, radius_m: float, h_min_m: float, h_max_m: float,
                      chunk: int = 5_000_000) -> pd.DataFrame:
    """Points within a square window of `radius_m` around any centre and `h_min_m` < h < `h_max_m` above ground."""
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    index = cKDTree(centres)
    keep = np.zeros(len(cloud), dtype=bool)
    x, y, h = cloud["x"].to_numpy(), cloud["y"].to_numpy(), cloud["h"].to_numpy()
    lo, hi = centres.min(axis=0) - radius_m, centres.max(axis=0) + radius_m
    for start in range(0, len(cloud), chunk):
        sl = slice(start, start + chunk)
        near = (x[sl] >= lo[0]) & (x[sl] <= hi[0]) & (y[sl] >= lo[1]) & (y[sl] <= hi[1]) & (h[sl] > h_min_m) & (h[sl] < h_max_m)
        if near.any():
            d, _ = index.query(np.column_stack([x[sl][near], y[sl][near]]), p=np.inf, distance_upper_bound=radius_m + 1e-9)
            near[np.flatnonzero(near)] = np.isfinite(d)
        keep[sl] = near
    return cloud.loc[keep, ["x", "y", "z", "h", "instance"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage inputs and runners
# ---------------------------------------------------------------------------

def load_plots(cfg):
    """Plot outlines and corner RMSE from the corner sheet (reads Excel: call before geopandas reads a file)."""
    from core import coords

    from core.io_field import load_corners

    corners = load_corners(cfg)
    plots = coords.parzelle_polygons(corners, crs=cfg.site.crs)
    corner_rmse = pd.Series({k: v["rmse"] for k, v in coords.fit_parzelle_transforms(corners).items()}, name="corner_rmse_m")
    return plots, corner_rmse


def _read_inputs(cfg):
    import geopandas as gpd

    field = pd.read_csv(cfg.table("field_trees.csv"), dtype={"tree_id": str})
    instances = gpd.read_file(cfg.table("lidar_instances.gpkg"), layer="instance_base")
    return field, instances


def run_matching(cfg) -> None:
    """Stage 4: writes match_distance_only.csv, field_trees_detection.csv, detection_per_parzelle.csv, detection_all_trees.gpkg."""
    plots, corner_rmse = load_plots(cfg)          # Excel first
    field, instances = _read_inputs(cfg)
    field = add_block(field)
    field = add_dbh_class(field, cfg.cleaning.dbh_plausible_mm, cfg.cleaning.get("dbh_class_edges_mm",
                                                                                 [0, 200, 300, 400, 500, 2000]))
    m = cfg.matching
    tables = compute_matching(
        field, instances, plots, corner_rmse,
        max_distance_m=m.max_distance_m, removed_threshold=m.removed_threshold, removal_penalty_m=m.removal_penalty_m,
        high_confidence_distance_m=m.high_confidence_distance_m, height_gate_rel=m.height_gate_rel,
        gone_above=m.gone_above, standing_below=m.standing_below, control_shift_m=m.control_shift_m)

    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    tables["control"].to_csv(cfg.table("chance_control_matches.csv"), index=False)
    tables["felled"].to_csv(cfg.table("match_distance_only.csv"), index=False)
    tables["table"].to_csv(cfg.table("field_trees_detection.csv"), index=False)
    tables["per_plot"].to_csv(cfg.table("detection_per_parzelle.csv"))
    write_detection_gpkg(cfg.table("detection_all_trees.gpkg"), tables["table"], tables["inst"], tables["per_plot"],
                         plots, cfg.site.crs)
    print(f"matching: {tables['result']['detection'].value_counts().to_dict()}")


def run_unmatched(cfg) -> None:
    """Stage 10: writes lidar_instances_without_field_tree.csv (no QA figures)."""
    from core import instances as core_instances

    plots, _ = load_plots(cfg)                    # Excel first
    field, instances = _read_inputs(cfg)
    detection = pd.read_csv(cfg.table("field_trees_detection.csv"), dtype={"tree_id": str})
    m = cfg.matching
    # Rebuild the LiDAR side of stage 4 from the written detection table
    result = detection.copy()
    result["detection"] = pd.Categorical(result["detection"], categories=CLASSES)
    inst, in_plot = instances_in_plots(instances, plots, result, m.removed_threshold)

    lonely_bases = in_plot.loc[in_plot["tree_id"].isna(), ["instance", "x_base", "y_base"]]
    cloud = core_instances.load_before_cloud(cfg)
    near = points_near_bases(cloud, lonely_bases[["x_base", "y_base"]].to_numpy(), 1.5, 1.0, 9.0)
    del cloud
    try:
        from core.dbh import stem_profiles
    except ImportError as exc:
        raise RuntimeError("core.dbh.stem_profiles is needed for the stem search at the instance bases") from exc
    stems_at_base = stem_profiles(near, pd.DataFrame({"tree_id": lonely_bases["instance"], "x": lonely_bases["x_base"],
                                                      "y": lonely_bases["y_base"]}))
    lonely = unmatched_instances(
        inst, in_plot, result, plots, stems_at_base,
        max_distance_m=m.max_distance_m, high_confidence_distance_m=m.high_confidence_distance_m,
        min_stem_slices=cfg.structure.min_stem_slices, edge_m=cfg.structure.edge_m,
        figure_count=m.get("unmatched_figure_count", 10), figure_seed=m.get("unmatched_figure_seed", 155))
    lonely.to_csv(cfg.table("lidar_instances_without_field_tree.csv"), index=False)
    print(f"unmatched: {len(lonely)} instances; {lonely['reading'].value_counts().to_dict()}")
