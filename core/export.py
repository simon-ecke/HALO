"""Export the tables and statistics of one site as the data object of the static dashboard.

`build_data(cfg, tables_dir)` returns a plain dict {meta, stats, trees, parzellen, samples, volume,
identity, ledger, ...}; `write_data_js` writes it as `window.DASHBOARD_DATA = ...;`. Nothing here
computes a statistic: tables are joined, columns are chosen and numbers are rounded. Layout geometry
(plot centres, plot frame angle) is the only derived quantity.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from core.tables import has_table, read_table

# Tables the page can draw. Missing optional ones are listed in meta.missing_tables.
REQUIRED = ["field_trees_detection.csv", "match_identity.csv", "wood_volume_per_tree.csv", "sample_trees.csv"]
OPTIONAL = [
    "match_tiers.csv", "match_causes.csv", "detection_per_parzelle.csv", "lidar_instances_without_field_tree.csv",
    "plot_offsets.csv", "volume_per_parzelle.csv", "volume_per_species.csv", "height_diameter_curves.csv",
    "stand_per_parzelle.csv", "stand_per_parzelle_species.csv", "sampling_value.csv", "detection_model.csv",
    "ledger.csv", "accuracy_by_identity.csv", "height_variants.csv", "sample_tree_points.csv",
]


def _clean(value, nd=None):
    """JSON-safe scalar: NaN/inf -> None, numpy -> python, optional rounding."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return None
        return round(float(value), nd) if nd is not None else float(value)
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    return value


def frame_records(df: pd.DataFrame, rounding: dict | None = None, default_nd: int = 3) -> list[dict]:
    """List of row dicts with NaN -> None and per-column rounding (decimals)."""
    rounding = rounding or {}
    out = []
    cols = list(df.columns)
    for row in df.itertuples(index=False, name=None):
        out.append({c: _clean(v, rounding.get(c, default_nd)) for c, v in zip(cols, row)})
    return out


def frame_columnar(df: pd.DataFrame, rounding: dict | None = None, default_nd: int = 3) -> dict:
    """Compact {cols: [...], rows: [[...], ...]} form of a table (the page turns it into objects)."""
    rounding = rounding or {}
    cols = list(df.columns)
    rows = [[_clean(v, rounding.get(c, default_nd)) for c, v in zip(cols, r)]
            for r in df.itertuples(index=False, name=None)]
    return {"cols": cols, "rows": rows}


def _stats(path: Path) -> list[dict]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for e in entries:
        out.append({
            "id": e["id"], "value": _clean(e.get("value"), 6), "unit": e.get("unit"), "n": _clean(e.get("n")),
            "ci_low": _clean(e.get("ci_low"), 6), "ci_high": _clean(e.get("ci_high"), 6),
            "subset": e.get("subset"), "statement": e.get("statement"),
        })
    return out


def _int_or_str(v):
    """Plot ids are numbers in some sites and text in others; keep ints as ints."""
    try:
        f = float(v)
        return int(f) if f == int(f) else v
    except (TypeError, ValueError):
        return v


def _trees(cfg, td) -> pd.DataFrame:
    ft = read_table(cfg, "field_trees_detection.csv", td)
    keep = ["tree_id", "parzelle", "species", "local_x", "local_y", "dbh_mm", "dbh_class", "detection",
            "removed_2026", "base_dist_m"]
    t = ft[[c for c in keep if c in ft.columns]].copy()
    mi = read_table(cfg, "match_identity.csv", td)
    t = t.merge(mi[["tree_id", "offset_m", "offset_corrected_m", "second_best_margin_m", "identity_score",
                    "identity_class"]], on="tree_id", how="left")
    wv = read_table(cfg, "wood_volume_per_tree.csv", td)
    wcols = ["tree_id", "dbh_uls_mm", "height_uls_m", "height_source", "dbh_quality", "dbh_field_valid",
             "wood_volume_field_m3", "wood_volume_uls_m3"]
    t = t.merge(wv[[c for c in wcols if c in wv.columns]], on="tree_id", how="left")
    if has_table(cfg, "match_causes.csv", td):
        mc = read_table(cfg, "match_causes.csv", td)
        t = t.merge(mc[["tree_id", "cause", "attributed_to"]], on="tree_id", how="left")
    else:
        t["cause"] = None
        t["attributed_to"] = None
    return t


def _plot_geometry(cfg, td) -> dict:
    """Plot centre in map coordinates (m, relative to the south-west-most plot) and the angle of the
    plot frame against the map axes; both only place the plots and the offset arrows on the map."""
    ft = read_table(cfg, "field_trees_detection.csv", td)
    if "utm_x" not in ft.columns:
        return {}
    ox, oy = ft.utm_x.min(), ft.utm_y.min()
    geo = {}
    for p, g in ft.dropna(subset=["utm_x", "utm_y", "local_x", "local_y"]).groupby("parzelle"):
        ang = 0.0
        if len(g) >= 3:
            a = np.c_[g.local_x, g.local_y, np.ones(len(g))]
            cx = np.linalg.lstsq(a, g.utm_x.to_numpy(), rcond=None)[0]
            cy = np.linalg.lstsq(a, g.utm_y.to_numpy(), rcond=None)[0]
            ang = math.degrees(math.atan2(cy[0], cx[0]))  # direction of the local x axis on the map
        geo[_int_or_str(p)] = {"mx": float(g.utm_x.mean() - ox), "my": float(g.utm_y.mean() - oy), "frame_deg": ang}
    return geo


def _parzellen(cfg, td) -> list[dict]:
    geo = _plot_geometry(cfg, td)
    base = None
    if has_table(cfg, "stand_per_parzelle.csv", td):
        base = read_table(cfg, "stand_per_parzelle.csv", td)
    else:
        ft = read_table(cfg, "field_trees_detection.csv", td)
        base = ft.groupby("parzelle").size().rename("n_field_trees").reset_index()
    base = base.copy()
    base["parzelle"] = base["parzelle"].map(_int_or_str)
    if has_table(cfg, "plot_offsets.csv", td):
        po = read_table(cfg, "plot_offsets.csv", td)
        po["parzelle"] = po["parzelle"].map(_int_or_str)
        base = base.merge(po, on="parzelle", how="left")
    if has_table(cfg, "detection_per_parzelle.csv", td):
        dp = read_table(cfg, "detection_per_parzelle.csv", td)[
            ["parzelle", "uncertain", "not detected", "lidar_instances", "lidar_without_field_tree",
             "count_ratio", "corner_rmse_m"]]
        dp["parzelle"] = dp["parzelle"].map(_int_or_str)
        base = base.merge(dp, on="parzelle", how="left")
    if has_table(cfg, "volume_per_parzelle.csv", td):
        vp = read_table(cfg, "volume_per_parzelle.csv", td)
        vp["parzelle"] = vp["parzelle"].map(_int_or_str)
        vcols = ["parzelle", "n_modelled", "volume_field_with_modelled_m3", "volume_field_without_modelled_m3",
                 "volume_uls_m3", "volume_field_removed_m3", "volume_field_remaining_m3", "volume_uls_removed_m3",
                 "volume_uls_remaining_m3", "volume_field_with_modelled_m3_per_ha", "volume_uls_m3_per_ha"]
        base = base.merge(vp[[c for c in vcols if c in vp.columns]], on="parzelle", how="left")
    mirrored = {_int_or_str(x) for x in (cfg.plots.get("y_mirrored", []) or [])}
    recs = frame_records(base, {"area_m2": 0, "detection_rate": 4, "matched_rate": 4, "dx_m": 3, "dy_m": 3,
                                "rotation_deg": 3, "residual_rms_m": 3})
    for r in recs:
        g = geo.get(r["parzelle"], {})
        r["mx"] = _clean(g.get("mx"), 1)
        r["my"] = _clean(g.get("my"), 1)
        r["frame_deg"] = _clean(g.get("frame_deg"), 2)
        r["y_mirrored"] = r["parzelle"] in mirrored
    return recs


def _samples(cfg, td, out_dir: Path | None) -> tuple[list[dict], dict, list[str]]:
    st = read_table(cfg, "sample_trees.csv", td)
    wv = read_table(cfg, "wood_volume_per_tree.csv", td)
    extra = ["tree_id"] + [c for c in ("wood_volume_massband_m3", "wood_volume_vertex_m3", "wood_volume_uls_m3",
                                         "wood_volume_field_m3") if c in wv.columns]
    st = st.merge(wv[extra], on="tree_id", how="left")
    recs = frame_records(st, {"identity_score": 3, "dbh_field_mm": 0, "dbh_uls_mm": 0}, 3)
    figs: list[str] = []
    sf = cfg.get("dashboard", None)
    sf = sf.get("sample_figures", None) if sf is not None else None
    if sf is not None and out_dir is not None:
        src = Path(sf.dir)
        if not src.is_absolute():
            src = cfg.project_root / src
        patterns = [sf.pattern] + list(sf.get("extra_patterns", []) or [])
        for r in recs:
            r["figures"] = []
            for pat in patterns:
                name = pat.format(tree_id=r["tree_id"])
                f = src / name
                if f.exists():
                    figs.append(name)
                    r["figures"].append("img/" + name)
    points: dict = {}
    if has_table(cfg, "sample_tree_points.csv", td):
        pts = read_table(cfg, "sample_tree_points.csv", td)
        for tid, g in pts.groupby("tree_id"):
            points[str(tid)] = frame_columnar(g[["x_rel", "y_rel", "h", "group"]], {"x_rel": 2, "y_rel": 2, "h": 2})
    return recs, points, figs


def _methods(cfg) -> list[dict]:
    """Text of the method documents (site-independent, in docs/) for the Methods tab."""
    out = []
    for name in ("METHODS.md", "METHODS_identity.md"):
        f = cfg.project_root / "docs" / name
        if f.exists():
            out.append({"name": name, "text": f.read_text(encoding="utf-8")})
    return out


def _volume(cfg, td) -> dict:
    out: dict = {"min_trees_per_species": int(cfg.volume.get("min_trees_per_species", 20))}
    if has_table(cfg, "volume_per_species.csv", td):
        out["per_species"] = frame_records(read_table(cfg, "volume_per_species.csv", td), default_nd=3)
    if has_table(cfg, "height_diameter_curves.csv", td):
        out["height_curves"] = frame_records(read_table(cfg, "height_diameter_curves.csv", td), default_nd=5)
    if has_table(cfg, "sampling_value.csv", td):
        out["sampling"] = frame_records(read_table(cfg, "sampling_value.csv", td), default_nd=3)
    return out


def build_data(cfg, tables_dir=None, out_dir: Path | None = None) -> tuple[dict, list[str]]:
    """Return (data object, list of sample figure files to copy)."""
    td = tables_dir
    for name in REQUIRED:
        if not has_table(cfg, name, td):
            raise FileNotFoundError(f"Required table {name} not found (tables dir: {td or cfg.tables_dir}).")
    missing = [n for n in OPTIONAL if not has_table(cfg, n, td)]

    stats_path = cfg.output_dir / "stats.json"
    trees = _trees(cfg, td)
    tree_cols = {
        "tree_id": ("id", 0), "parzelle": ("p", 0), "species": ("sp", 0), "local_x": ("x", 2), "local_y": ("y", 2),
        "dbh_mm": ("dbh", 1), "dbh_class": ("cls", 0), "detection": ("det", 0), "removed_2026": ("rem", 0),
        "base_dist_m": ("bd", 2), "offset_m": ("off", 2), "offset_corrected_m": ("offc", 2),
        "second_best_margin_m": ("mar", 2), "identity_score": ("sc", 3), "identity_class": ("idc", 0),
        "dbh_uls_mm": ("dbhU", 1), "height_uls_m": ("hU", 2), "height_source": ("hsrc", 0),
        "dbh_quality": ("dbhq", 0), "dbh_field_valid": ("valid", 0), "wood_volume_field_m3": ("vF", 4),
        "wood_volume_uls_m3": ("vU", 4), "cause": ("cause", 0), "attributed_to": ("att", 0),
    }
    t = trees[[c for c in tree_cols if c in trees.columns]].copy()
    t["parzelle"] = t["parzelle"].map(_int_or_str)
    t["identity_class"] = t["identity_class"].fillna("unmatched")
    rounding = {c: tree_cols[c][1] for c in t.columns}
    t.columns = [tree_cols[c][0] for c in t.columns]
    rounding = {tree_cols[c][0]: nd for c, nd in rounding.items()}

    samples, points, figs = _samples(cfg, td, out_dir)
    dash = cfg.get("dashboard", None)
    title = (dash.get("title", None) if dash is not None else None) or f"Site {cfg.site.id}: {cfg.site.name}"
    effort = cfg.effort.to_dict() if hasattr(cfg.effort, "to_dict") else dict(cfg.effort or {})

    data = {
        "meta": {
            "site_id": str(cfg.site.id), "site_name": cfg.site.name, "title": title,
            "built": date.today().isoformat(), "has_points": bool(points),
            "effort": effort if effort else None,
            "missing_tables": missing,
            "min_trees_per_species": int(cfg.volume.get("min_trees_per_species", 20)),
            "y_mirrored": [_int_or_str(x) for x in (cfg.plots.get("y_mirrored", []) or [])],
            "explain_height_diff_m": float(cfg.get("analyses").get("explain_height_diff_m", 1.0))
            if cfg.get("analyses", None) is not None else 1.0,
        },
        "stats": _stats(stats_path),
        "trees": frame_columnar(t, rounding),
        "parzellen": _parzellen(cfg, td),
        "samples": samples,
        "sample_points": points,
        "volume": _volume(cfg, td),
        "identity": {},
        "methods": _methods(cfg),
        "ledger": [],
        "detection_model": [],
        "unmatched_instances": [],
        "height_variants": [],
        "stand_species": [],
    }
    if has_table(cfg, "accuracy_by_identity.csv", td):
        data["identity"]["accuracy"] = frame_records(read_table(cfg, "accuracy_by_identity.csv", td), default_nd=4)
    if has_table(cfg, "ledger.csv", td):
        led = read_table(cfg, "ledger.csv", td)
        led["parzelle"] = led["parzelle"].map(_int_or_str)
        data["ledger"] = frame_records(led)
    if has_table(cfg, "detection_model.csv", td):
        data["detection_model"] = frame_records(read_table(cfg, "detection_model.csv", td), default_nd=4)
    if has_table(cfg, "height_variants.csv", td):
        data["height_variants"] = frame_records(read_table(cfg, "height_variants.csv", td), default_nd=3)
    if has_table(cfg, "stand_per_parzelle_species.csv", td):
        sp = read_table(cfg, "stand_per_parzelle_species.csv", td)
        sp["parzelle"] = sp["parzelle"].map(_int_or_str)
        data["stand_species"] = frame_records(sp, default_nd=2)
    if has_table(cfg, "lidar_instances_without_field_tree.csv", td):
        li = read_table(cfg, "lidar_instances_without_field_tree.csv", td)
        keep = [c for c in ("instance", "parzelle", "height", "crown_area_m2", "nearest_stem_m", "reading",
                            "gone_after_harvest", "in_plot", "edge") if c in li.columns]
        li = li[keep].copy()
        li["parzelle"] = li["parzelle"].map(_int_or_str)
        data["unmatched_instances"] = frame_records(li, {"height": 1, "crown_area_m2": 1, "nearest_stem_m": 2})
    return data, figs


def write_data_js(data: dict, path) -> None:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # "</" would end an inline script if the file were ever inlined; keep it safe.
    text = text.replace("</", "<\\/")
    Path(path).write_text("window.DASHBOARD_DATA = " + text + ";\n", encoding="utf-8")
