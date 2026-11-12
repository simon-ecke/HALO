"""Wood volume per tree, computed twice (field diameter and ULS diameter) with a replaceable volume function.

Definitions
- wood_volume_field = f(species, field diameter, ULS height); wood_volume_uls = f(species, ULS diameter, ULS height).
  Both use the same ULS height, so their difference is the diameter difference only.
- Field trees without a matched instance get a modelled height (height-diameter curve fitted on matched trees)
  for the field volume; their ULS volume stays empty.
- A missing or implausible field diameter is not corrected: no field volume, the tree is listed as excluded.

Volume methods are registered in METHODS. A method is a function `evaluate(species, d_mm, h_m, params, unit)`
returning m3. All numbers of the default form-factor method are provisional placeholders.

Pure functions take and return DataFrames; only `run_volume` writes files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from core.stats import agreement
from core.tables import read_table

DEFAULT_SPECIES = "default"          # name of the fallback row in a parameter file
UNIT_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001}


# ---------------------------------------------------------------------------
# Volume methods (plug-in registry)
# ---------------------------------------------------------------------------

def _row_values(species: pd.Series, table: pd.DataFrame, column: str) -> np.ndarray:
    """Per-tree value of `column`, looked up by species with the `default` row as fallback (NaN if neither)."""
    lookup = table[column].to_dict()
    fallback = lookup.get(DEFAULT_SPECIES, np.nan)
    return np.array([lookup.get(s, fallback) for s in species], dtype=float)


def _form_factor_params(df: pd.DataFrame) -> pd.DataFrame:
    if not {"species", "form_factor"} <= set(df.columns):
        raise ValueError("form-factor file needs the columns species and form_factor")
    return df.set_index("species")[["form_factor"]].astype(float)


def _form_factor_eval(species, d_mm, h_m, params, unit="cm"):
    """V = pi/4 * d^2 * h * f with d in metres."""
    f = _row_values(species, params, "form_factor")
    d = np.asarray(d_mm, dtype=float) / 1000.0
    return np.pi / 4.0 * d ** 2 * np.asarray(h_m, dtype=float) * f


def _form_factor_shift(params: pd.DataFrame, delta: float) -> pd.DataFrame:
    out = params.copy()
    out["form_factor"] = out["form_factor"] + delta
    return out


def _parameter_table_params(df: pd.DataFrame) -> pd.DataFrame:
    if not {"species", "a", "b", "c"} <= set(df.columns):
        raise ValueError("parameter file needs the columns species, a, b, c")
    return df.set_index("species")[["a", "b", "c"]].astype(float)


def _parameter_table_eval(species, d_mm, h_m, params, unit="cm"):
    """V = a * d^b * h^c (m3) with d in `unit` (cm, mm or m) and h in metres; a, b, c per species."""
    d = np.asarray(d_mm, dtype=float) / 1000.0 / UNIT_TO_M[unit]
    h = np.asarray(h_m, dtype=float)
    a, b, c = (_row_values(species, params, k) for k in ("a", "b", "c"))
    return a * d ** b * h ** c


@dataclass(frozen=True)
class VolumeMethod:
    name: str
    prepare: Callable[[pd.DataFrame], pd.DataFrame]       # parameter file -> parameter table
    evaluate: Callable                                      # (species, d_mm, h_m, params, unit) -> m3
    shift: Callable | None = None                           # optional: parameters moved by +delta (form-factor sensitivity)


METHODS: dict[str, VolumeMethod] = {
    "form_factor": VolumeMethod("form_factor", _form_factor_params, _form_factor_eval, _form_factor_shift),
    "parameter_table": VolumeMethod("parameter_table", _parameter_table_params, _parameter_table_eval),
}


def method_from_cfg(cfg) -> tuple[VolumeMethod, pd.DataFrame, str]:
    """Method object, its parameter table and the diameter unit, as set in the config's `volume` section."""
    vol = cfg.volume
    name = vol.method
    if name not in METHODS:
        raise ValueError(f"volume.method '{name}' is not registered; known: {sorted(METHODS)}")
    path = Path(str(vol.params_file))
    if not path.is_absolute():
        path = cfg.project_root / path
    params = METHODS[name].prepare(pd.read_csv(path))
    unit = vol.get("parameter_diameter_unit", "cm")
    return METHODS[name], params, unit


# ---------------------------------------------------------------------------
# Height-diameter curve (Naeslund form, fitted by a straight line)
#   h = 1.3 + (d / (a + b d))^2, d in cm  <=>  d / sqrt(h - 1.3) = a + b d
# ---------------------------------------------------------------------------

def _fit_naeslund(d_mm: np.ndarray, h_m: np.ndarray, breast: float) -> tuple[float, float]:
    d = d_mm / 10.0
    y = d / np.sqrt(h_m - breast)
    b, a = np.polyfit(d, y, 1)
    return float(a), float(b)


def _predict_naeslund(d_mm, a: float, b: float, breast: float) -> np.ndarray:
    d = np.asarray(d_mm, dtype=float) / 10.0
    denom = a + b * d
    with np.errstate(divide="ignore", invalid="ignore"):
        h = breast + (d / denom) ** 2
    return np.where(denom > 0, h, np.nan)


def fit_height_curve(d_mm, h_m, breast: float = 1.3, min_height_margin_m: float = 0.5) -> dict:
    """Fit one curve and its leave-one-out error (every tree predicted by a curve fitted without it).

    Returns a, b, n, loo_bias_m, loo_rmse_m, loo_median_m, loo_mad_m (error = predicted - ULS height).
    """
    d = np.asarray(d_mm, dtype=float)
    h = np.asarray(h_m, dtype=float)
    ok = np.isfinite(d) & np.isfinite(h) & (d > 0) & (h > breast + min_height_margin_m)
    d, h = d[ok], h[ok]
    n = len(d)
    if n < 3:
        return {"a": np.nan, "b": np.nan, "n": n, "loo_bias_m": np.nan, "loo_rmse_m": np.nan,
                "loo_median_m": np.nan, "loo_mad_m": np.nan}
    a, b = _fit_naeslund(d, h, breast)
    err = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        ai, bi = _fit_naeslund(d[keep], h[keep], breast)
        err[i] = _predict_naeslund(d[i], ai, bi, breast) - h[i]
    err = err[np.isfinite(err)]
    med = float(np.median(err)) if len(err) else np.nan
    return {"a": a, "b": b, "n": n, "loo_bias_m": float(err.mean()) if len(err) else np.nan,
            "loo_rmse_m": float(np.sqrt((err ** 2).mean())) if len(err) else np.nan,
            "loo_median_m": med, "loo_mad_m": float(1.4826 * np.median(np.abs(err - med))) if len(err) else np.nan}


def fit_curves(species: pd.Series, d_mm: pd.Series, h_m: pd.Series, *, min_trees: int, breast: float) -> pd.DataFrame:
    """One pooled curve plus one per species with at least `min_trees` fitting trees.

    Columns: scope (pooled | species), species, form, n, a, b, loo_*.
    """
    rows = [{"scope": "pooled", "species": "", "form": "naeslund_cm", **fit_height_curve(d_mm, h_m, breast)}]
    usable = d_mm.notna() & h_m.notna()
    for sp, idx in species[usable].groupby(species[usable]).groups.items():
        if len(idx) >= min_trees:
            rows.append({"scope": "species", "species": sp, "form": "naeslund_cm",
                         **fit_height_curve(d_mm[idx], h_m[idx], breast)})
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype(int)
    return out


def model_heights(species: pd.Series, d_mm: pd.Series, curves: pd.DataFrame, breast: float) -> pd.Series:
    """Height from the species curve where there is one, else from the pooled curve."""
    pooled = curves[curves["scope"] == "pooled"].iloc[0]
    out = pd.Series(_predict_naeslund(d_mm.to_numpy(float), pooled["a"], pooled["b"], breast), index=d_mm.index)
    for _, row in curves[curves["scope"] == "species"].iterrows():
        m = species == row["species"]
        if m.any() and np.isfinite(row["a"]):
            pred = _predict_naeslund(d_mm[m].to_numpy(float), row["a"], row["b"], breast)
            out[m] = np.where(np.isfinite(pred), pred, out[m])
    return out


# ---------------------------------------------------------------------------
# The per-tree table
# ---------------------------------------------------------------------------

def compute_volume_table(trees: pd.DataFrame, *, method: VolumeMethod, params: pd.DataFrame,
                         dbh_plausible_mm: Sequence[float], min_trees_per_species: int,
                         breast_height_m: float = 1.3, diameter_unit: str = "cm") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-tree volumes and the height-diameter curves behind the modelled heights.

    `trees` needs: tree_id, parzelle, species, dbh_field_mm, dbh_uls_mm, height_uls_m (NaN = no matched
    instance), and optionally detection, identity_class, removed_2026, height_massband, height_vertex.
    Returns (table, curves).
    """
    t = trees.copy()
    lo, hi = dbh_plausible_mm
    d_field = pd.to_numeric(t["dbh_field_mm"], errors="coerce")
    t["dbh_field_valid"] = d_field.between(lo, hi)
    d_field_ok = d_field.where(t["dbh_field_valid"])
    d_uls = pd.to_numeric(t["dbh_uls_mm"], errors="coerce")
    h_uls = pd.to_numeric(t["height_uls_m"], errors="coerce")
    has_h = h_uls.notna()

    # curves: matched trees with a valid field diameter (the diameter they will be applied to)
    curves = fit_curves(t["species"], d_field_ok.where(has_h), h_uls, min_trees=min_trees_per_species,
                        breast=breast_height_m)
    modelled = model_heights(t["species"], d_field_ok, curves, breast_height_m)
    t["height_used_m"] = h_uls.where(has_h, modelled)
    t["height_source"] = np.where(has_h, "uls", np.where(t["height_used_m"].notna(), "modelled", ""))

    def vol(d, h):
        return pd.Series(method.evaluate(t["species"], d, h, params, diameter_unit), index=t.index)

    t["wood_volume_field_m3"] = vol(d_field_ok, t["height_used_m"])
    t["wood_volume_uls_m3"] = vol(d_uls.where(has_h), h_uls)
    t["volume_method"] = method.name
    for col, name in (("height_massband", "wood_volume_massband_m3"), ("height_vertex", "wood_volume_vertex_m3")):
        if col in t:
            t[name] = vol(d_field_ok, pd.to_numeric(t[col], errors="coerce"))
    t = t.rename(columns={"dbh_field_mm": "dbh_field_mm", "dbh_uls_mm": "dbh_uls_mm"})
    first = ["tree_id", "parzelle", "species", "dbh_field_mm", "dbh_uls_mm", "height_uls_m", "height_used_m",
             "height_source", "wood_volume_field_m3", "wood_volume_uls_m3", "volume_method", "dbh_field_valid",
             "detection", "identity_class", "removed_2026"]
    cols = [c for c in first if c in t] + [c for c in t if c not in first]
    return t[cols], curves


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------

def load_inputs(cfg, tables_dir=None) -> pd.DataFrame:
    """Merge the pipeline tables into the input of compute_volume_table (one row per field tree)."""
    f = read_table(cfg, "field_trees_detection.csv", tables_dir)
    h = read_table(cfg, "height_dbh_all_trees.csv", tables_dir)[["tree_id", "dbh_lidar_mm", "dbh_quality"]]
    tiers = read_table(cfg, "match_tiers.csv", tables_dir)[["tree_id", "instance", "lidar_height"]]
    tiers = tiers.rename(columns={"instance": "instance_tier", "lidar_height": "height_uls_m"})
    t = f[["tree_id", "parzelle", "species", "dbh_mm", "detection", "removed_2026", "height_massband",
           "height_vertex"]].merge(h, on="tree_id", how="left").merge(tiers, on="tree_id", how="left")
    t["height_uls_m"] = t["height_uls_m"].where(t["instance_tier"].notna())   # matched = has an instance
    try:
        ident = read_table(cfg, "match_identity.csv", tables_dir)[["tree_id", "identity_class"]]
        t = t.merge(ident, on="tree_id", how="left")
    except FileNotFoundError:
        t["identity_class"] = ""
    t["removed_2026"] = t["removed_2026"].astype(bool)
    return t.rename(columns={"dbh_mm": "dbh_field_mm", "dbh_lidar_mm": "dbh_uls_mm"}).drop(columns="instance_tier")


def dbh_ratio_check(trees: pd.DataFrame, plausible: Sequence[float]) -> dict:
    """Median of field / ULS diameter on matched trees; about 1 for a diameter, about 3.14 for a circumference."""
    m = trees["height_uls_m"].notna() & trees["dbh_uls_mm"].gt(0) & trees["dbh_field_mm"].between(*plausible)
    ratio = (trees["dbh_field_mm"] / trees["dbh_uls_mm"])[m]
    return {"n": int(len(ratio)), "median": float(ratio.median()) if len(ratio) else np.nan}


def plot_areas_ha(cfg) -> pd.Series:
    """Plot area in ha per Parzelle (convex hull of the surveyed corners)."""
    from core import coords, io_field
    poly = coords.parzelle_polygons(io_field.load_corners(cfg))
    return pd.Series(poly["area_m2"].to_numpy() / 10000.0, index=poly["parzelle"].astype(int).to_numpy())


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _sums(t: pd.DataFrame) -> dict:
    uls_h = t["height_source"] == "uls"
    return {"n_trees": len(t), "n_field_valid": int(t["dbh_field_valid"].sum()),
            "n_uls_height": int(uls_h.sum()), "n_modelled": int((t["height_source"] == "modelled").sum()),
            "n_uls_volume": int(t["wood_volume_uls_m3"].notna().sum()),
            "volume_field_with_modelled_m3": t["wood_volume_field_m3"].sum(),
            "volume_field_without_modelled_m3": t.loc[uls_h, "wood_volume_field_m3"].sum(),
            "volume_uls_m3": t["wood_volume_uls_m3"].sum()}


def summary_per_parzelle(t: pd.DataFrame, areas_ha: pd.Series | None) -> pd.DataFrame:
    rows = []
    for p, g in t.groupby("parzelle"):
        r = {"parzelle": p, **_sums(g)}
        rem = g["removed_2026"].astype(bool)
        r["volume_field_removed_m3"] = g.loc[rem, "wood_volume_field_m3"].sum()
        r["volume_field_remaining_m3"] = g.loc[~rem, "wood_volume_field_m3"].sum()
        r["volume_uls_removed_m3"] = g.loc[rem, "wood_volume_uls_m3"].sum()
        r["volume_uls_remaining_m3"] = g.loc[~rem, "wood_volume_uls_m3"].sum()
        area = float(areas_ha.get(int(p), np.nan)) if areas_ha is not None else np.nan
        r["area_ha"] = area
        for k in ("volume_field_with_modelled_m3", "volume_field_without_modelled_m3", "volume_uls_m3"):
            r[k.replace("_m3", "_m3_per_ha")] = r[k] / area if area > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def summary_per_species(t: pd.DataFrame, area_ha: float | None) -> pd.DataFrame:
    rows = []
    for sp, g in t.groupby("species"):
        r = {"species": sp, **_sums(g)}
        for k in ("volume_field_with_modelled_m3", "volume_field_without_modelled_m3", "volume_uls_m3"):
            r[k.replace("_m3", "_m3_per_ha")] = r[k] / area_ha if area_ha else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics entries
# ---------------------------------------------------------------------------

def _entry(id_, value, unit, n, statement, subset=None, ci=(None, None)):
    def clean(v):
        if v is None:
            return None
        v = float(v)
        return None if np.isnan(v) else v
    return {"id": id_, "value": clean(value), "unit": unit, "n": None if n is None else int(n),
            "ci_low": clean(ci[0]), "ci_high": clean(ci[1]), "subset": subset, "statement": statement}


def _slug(text) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")


def _rel_bias(field: np.ndarray, uls: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Sum(uls - field) / sum(field) with a bootstrap interval over trees."""
    err = uls - field
    value = err.sum() / field.sum()
    idx = np.random.default_rng(seed).integers(0, len(err), size=(n_boot, len(err)))
    boot = err[idx].sum(axis=1) / field[idx].sum(axis=1)
    return float(value), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _class_labels(d_mm: pd.Series, edges: Sequence[float]) -> pd.Series:
    labels = [f"{int(a)}-{int(b)} mm" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(d_mm, bins=list(edges), labels=labels, right=False)


def _list_text(ids, limit=25) -> str:
    ids = [str(i) for i in ids]
    return ", ".join(ids[:limit]) + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


def entries_from_table(t: pd.DataFrame, curves: pd.DataFrame, *, method: VolumeMethod, params: pd.DataFrame,
                       diameter_unit: str, areas_ha: pd.Series | None, ratio: dict, plausible_mm: Sequence[float],
                       class_edges_mm: Sequence[float], min_trees: int, n_boot: int, seed: int,
                       sens: dict) -> list[dict]:
    """Statistics entries (id, subset, value, n, ci_low, ci_high) for the volume comparison. Pure."""
    out: list[dict] = []
    add = out.append
    M = f"The provisional {method.name} volume"
    pre = f"With the provisional {method.name} volume method"
    area_total = float(areas_ha.sum()) if areas_ha is not None and len(areas_ha) else np.nan

    # --- diameter check -------------------------------------------------
    verdict = "so the field value is a diameter, not a circumference" if 0.7 <= ratio["median"] <= 1.4 else "so the field value may be a circumference"
    add(_entry("volume.dbh_check.median_ratio", ratio["median"], "ratio", ratio["n"],
               f"The median of field diameter divided by ULS diameter is {ratio['median']:.3f} on {ratio['n']} matched trees, {verdict} (a circumference would give about 3.14)."))

    # --- excluded field trees -------------------------------------------
    miss = t[t["dbh_field_mm"].isna()]
    out_of_range = t[t["dbh_field_mm"].notna() & ~t["dbh_field_valid"]]
    lo, hi = plausible_mm
    add(_entry("volume.field_excluded", len(miss) + len(out_of_range), "trees", len(t),
               f"{len(miss) + len(out_of_range)} field trees get no field volume and are excluded from the field totals: "
               f"{len(miss)} with no field diameter ({_list_text(miss['tree_id'])}) and {len(out_of_range)} with a field diameter outside {lo:g} to {hi:g} mm "
               f"({_list_text([f'{a} = {b:g} mm' for a, b in zip(out_of_range['tree_id'], out_of_range['dbh_field_mm'])])}); none is corrected (provisional {method.name} volumes)."))
    uls_bad = t[t["dbh_uls_mm"].notna() & ~t["dbh_uls_mm"].between(lo, hi) & t["wood_volume_uls_m3"].notna()]
    add(_entry("volume.uls_dbh_outside_plausible", len(uls_bad), "trees", int(t["wood_volume_uls_m3"].notna().sum()),
               f"{len(uls_bad)} trees have a ULS diameter outside {lo:g} to {hi:g} mm; they stay in the ULS volume totals and are only flagged ({pre.lower()})."))

    # --- totals and per ha ----------------------------------------------
    def total_entries(scope, label, g, per_ha_area):
        uh = g["height_source"] == "uls"
        n_mod = int((g["height_source"] == "modelled").sum())
        vals = [("field_with_modelled", g["wood_volume_field_m3"].sum(), int(g["wood_volume_field_m3"].notna().sum()),
                 f"field diameter and ULS height, including {n_mod} trees whose height is modelled from the height-diameter curve"),
                ("field_without_modelled", g.loc[uh, "wood_volume_field_m3"].sum(), int(g.loc[uh, "wood_volume_field_m3"].notna().sum()),
                 "field diameter and ULS height, only trees with a matched ULS height (no modelled heights)"),
                ("uls", g["wood_volume_uls_m3"].sum(), int(g["wood_volume_uls_m3"].notna().sum()),
                 "ULS diameter and ULS height, only trees with a matched ULS instance and diameter")]
        for key, v, n, desc in vals:
            add(_entry(f"volume.total.{key}.{scope}", v, "m3", n,
                       f"{pre}, the total volume of {label} is {v:.1f} m3 from {n} trees ({desc})."))
            if per_ha_area and per_ha_area > 0:
                add(_entry(f"volume.per_ha.{key}.{scope}", v / per_ha_area, "m3/ha", n,
                           f"{pre}, the volume of {label} is {v / per_ha_area:.1f} m3/ha of the {per_ha_area:.2f} ha surveyed plot area ({desc})."))

    total_entries("stand", "the whole stand", t, area_total)
    for sp in sorted(t["species"].dropna().unique()):
        total_entries(_slug(sp), f"the species {sp}", t[t["species"] == sp], area_total)

    # --- per-tree difference uls - field ---------------------------------
    matched = t[t["height_source"].eq("uls") & t["wood_volume_field_m3"].notna() & t["wood_volume_uls_m3"].notna()].copy()
    matched["class"] = _class_labels(matched["dbh_field_mm"], class_edges_mm)
    subsets = {"all_matched": matched, "detected_only": matched[matched["detection"] == "detected"]}
    who = {"all_matched": "matched trees (detected and uncertain)", "detected_only": "detected trees"}
    same = "both volumes use the same ULS height, so this is the diameter difference only"

    def compare(prefix, label, g, subset):
        n = len(g)
        if n < 2:
            return
        fld, uls = g["wood_volume_field_m3"].to_numpy(), g["wood_volume_uls_m3"].to_numpy()
        a = agreement(pd.Series(fld), pd.Series(uls), n_boot=n_boot, seed=seed)
        rb, rb_lo, rb_hi = _rel_bias(fld, uls, n_boot, seed)
        sign = lambda v: "higher" if v >= 0 else "lower"
        iv = lambda k: f"{a[k + '_ci_low']:+.4f} to {a[k + '_ci_high']:+.4f} m3"
        add(_entry(f"{prefix}.bias", a["bias"], "m3", n, subset=subset, ci=(a["bias_ci_low"], a["bias_ci_high"]),
                   statement=f"{pre}, for {label} the ULS volume is on average {abs(a['bias']):.4f} m3 {sign(a['bias'])} than the field volume per tree (95 % bootstrap interval {iv('bias')}, {n} {who[subset]}); {same}."))
        add(_entry(f"{prefix}.relative_bias", rb, "share", n, subset=subset, ci=(rb_lo, rb_hi),
                   statement=f"{pre}, for {label} the ULS volume is {abs(rb) * 100:.1f} % {sign(rb)} than the field volume (summed over {n} {who[subset]}, 95 % bootstrap interval {rb_lo * 100:+.1f} % to {rb_hi * 100:+.1f} %); {same}."))
        add(_entry(f"{prefix}.rmse", a["rmse"], "m3", n, subset=subset, ci=(a["rmse_ci_low"], a["rmse_ci_high"]),
                   statement=f"{pre}, for {label} the root-mean-square difference of ULS minus field volume is {a['rmse']:.4f} m3 per tree ({n} {who[subset]}); {same}."))
        add(_entry(f"{prefix}.median", a["median"], "m3", n, subset=subset, ci=(a["median_ci_low"], a["median_ci_high"]),
                   statement=f"{pre}, for {label} the median difference of ULS minus field volume is {a['median']:+.4f} m3 per tree (95 % bootstrap interval {iv('median')}, {n} {who[subset]}); {same}."))
        add(_entry(f"{prefix}.mad", a["mad"], "m3", n, subset=subset, ci=(a["mad_ci_low"], a["mad_ci_high"]),
                   statement=f"{pre}, for {label} the median absolute deviation of the ULS minus field volume is {a['mad']:.4f} m3 per tree ({n} {who[subset]}); {same}."))

    for sub, g in subsets.items():
        compare(f"volume.tree_error.{sub}.all", "all species", g, sub)
        for sp, gs in g.groupby("species"):
            if len(gs) >= min_trees:
                compare(f"volume.tree_error.{sub}.species_{_slug(sp)}", f"the species {sp}", gs, sub)
        for cl, gc in g.groupby("class", observed=True):
            if len(gc) >= min_trees:
                compare(f"volume.tree_error.{sub}.class_{_slug(cl)}", f"field diameters of {cl}", gc, sub)

    if "dbh_quality" in matched:
        ok_fit = matched[matched["dbh_quality"] == "ok"]
        compare("volume.tree_error.ok_fit.all", "all species with a good ULS diameter fit only", ok_fit, "all_matched")

    # --- volume-weighted detection ---------------------------------------
    valid = t[t["wood_volume_field_m3"].notna()]
    tot = valid["wood_volume_field_m3"].sum()
    for state in ("detected", "uncertain", "not detected"):
        g = valid[valid["detection"] == state]
        share = g["wood_volume_field_m3"].sum() / tot
        add(_entry(f"volume.detection_share.{_slug(state)}", share, "share", len(g),
                   f"{pre}, {share * 100:.1f} % of the field volume ({g['wood_volume_field_m3'].sum():.1f} of {tot:.1f} m3, modelled heights included) stands in {len(g)} trees that are {state}."))

    # --- stand gap split --------------------------------------------------
    f_all = t["wood_volume_field_m3"].sum()
    u_all = t["wood_volume_uls_m3"].sum()
    both = t["wood_volume_field_m3"].notna() & t["wood_volume_uls_m3"].notna()
    missed = t.loc[t["wood_volume_field_m3"].notna() & t["wood_volume_uls_m3"].isna(), "wood_volume_field_m3"]
    diam = t.loc[both, "wood_volume_uls_m3"].sum() - t.loc[both, "wood_volume_field_m3"].sum()
    extra = t.loc[t["wood_volume_field_m3"].isna() & t["wood_volume_uls_m3"].notna(), "wood_volume_uls_m3"]
    gap = u_all - f_all
    add(_entry("volume.gap.total", gap, "m3", len(t),
               f"{pre}, the ULS stand volume is {abs(gap):.1f} m3 {'above' if gap >= 0 else 'below'} the field stand volume (field total includes modelled heights for unmatched trees)."))
    add(_entry("volume.gap.trees_missed", -missed.sum(), "m3", len(missed),
               f"{pre}, {missed.sum():.1f} m3 of the field volume stands in {len(missed)} trees without a ULS volume (no matched instance or no ULS diameter), which lowers the ULS total by that amount."))
    add(_entry("volume.gap.diameter_difference", diam, "m3", int(both.sum()),
               f"{pre}, on the {int(both.sum())} trees that have both volumes the ULS diameter changes the stand volume by {diam:+.1f} m3 against the field diameter at the same ULS height."))
    add(_entry("volume.gap.uls_only_trees", extra.sum(), "m3", len(extra),
               f"{pre}, {len(extra)} trees without a usable field diameter carry {extra.sum():.1f} m3 of ULS volume that has no field counterpart."))

    # --- sample trees -----------------------------------------------------
    if "wood_volume_massband_m3" in t:
        s = t[t["wood_volume_field_m3"].notna()]
        sets = {"massband": s[s["wood_volume_massband_m3"].notna() & s["wood_volume_uls_m3"].notna()],
                "vertex": s[s["wood_volume_vertex_m3"].notna() & s["wood_volume_uls_m3"].notna()]}
        # with the ULS height at the field diameter
        s = s.assign(wood_volume_uls_height_m3=np.nan)
        for key, g in sets.items():
            if len(g) == 0:
                continue
            uh = pd.Series(method.evaluate(g["species"], g["dbh_field_mm"], g["height_uls_m"], params, diameter_unit), index=g.index)
            ref = g[f"wood_volume_{key}_m3"]
            add(_entry(f"volume.sample.{key}_total", ref.sum(), "m3", len(g),
                       f"{pre}, the {len(g)} sample trees with a {key} height have {ref.sum():.2f} m3 at the {key} height and {uh.sum():.2f} m3 at the ULS height, all at the field diameter."))
            if len(g) >= 2:
                a = agreement(ref, uh, n_boot=n_boot, seed=seed)
                note = "the tape height is the reference" if key == "massband" else "the Vertex height is a separate check, not the reference"
                add(_entry(f"volume.sample.uls_height_vs_{key}.bias", a["bias"], "m3", len(g), ci=(a["bias_ci_low"], a["bias_ci_high"]),
                           statement=f"{pre}, using the ULS height instead of the {key} height changes the volume of a sample tree by {a['bias']:+.3f} m3 on average (95 % bootstrap interval {a['bias_ci_low']:+.3f} to {a['bias_ci_high']:+.3f} m3, {len(g)} trees, field diameter throughout; {note})."))
                add(_entry(f"volume.sample.uls_height_vs_{key}.median", a["median"], "m3", len(g), ci=(a["median_ci_low"], a["median_ci_high"]),
                           statement=f"{pre}, using the ULS height instead of the {key} height changes the volume of a sample tree by a median of {a['median']:+.3f} m3 ({len(g)} trees, field diameter throughout; {note})."))
        both_h = s[s["wood_volume_massband_m3"].notna() & s["wood_volume_vertex_m3"].notna()]
        if len(both_h) >= 2:
            a = agreement(both_h["wood_volume_massband_m3"], both_h["wood_volume_vertex_m3"], n_boot=n_boot, seed=seed)
            add(_entry("volume.sample.vertex_vs_massband.bias", a["bias"], "m3", len(both_h), ci=(a["bias_ci_low"], a["bias_ci_high"]),
                       statement=f"{pre}, the Vertex height instead of the tape height changes the volume of a sample tree by {a['bias']:+.3f} m3 on average ({len(both_h)} trees with both heights, field diameter throughout)."))

    # --- sensitivity -------------------------------------------------------
    def totals(dh=0.0, dd=0.0, dff=0.0):
        p = method.shift(params, dff) if (dff and method.shift) else params
        d_f = t["dbh_field_mm"].where(t["dbh_field_valid"]) + dd
        d_u = t["dbh_uls_mm"].where(t["height_uls_m"].notna()) + dd
        vf = method.evaluate(t["species"], d_f, t["height_used_m"] + dh, p, diameter_unit)
        vu = method.evaluate(t["species"], d_u, t["height_uls_m"] + dh, p, diameter_unit)
        return np.nansum(vf), np.nansum(vu)
    base_f, base_u = totals()
    cases = [("height_plus", "height plus ", "m", dict(dh=sens["height_m"]), sens["height_m"]),
             ("height_minus", "height minus ", "m", dict(dh=-sens["height_m"]), sens["height_m"]),
             ("dbh_plus", "diameter plus ", "mm", dict(dd=sens["dbh_mm"]), sens["dbh_mm"]),
             ("dbh_minus", "diameter minus ", "mm", dict(dd=-sens["dbh_mm"]), sens["dbh_mm"])]
    if method.shift:
        cases += [("form_factor_plus", "form factor plus ", "", dict(dff=sens["form_factor"]), sens["form_factor"]),
                  ("form_factor_minus", "form factor minus ", "", dict(dff=-sens["form_factor"]), sens["form_factor"])]
    for key, label, unit_txt, kw, amount in cases:
        f_t, u_t = totals(**kw)
        what = label + (f"{amount:g}" + (f" {unit_txt}" if unit_txt else ""))
        for who_, new, base in (("field", f_t, base_f), ("uls", u_t, base_u)):
            add(_entry(f"volume.sensitivity.{key}.{who_}", new - base, "m3", len(t),
                       statement=f"{pre}, changing the input by {what} moves the {who_.replace('uls','ULS')} stand volume by {new - base:+.1f} m3 ({(new / base - 1) * 100:+.1f} %)."))

    # --- harvest -----------------------------------------------------------
    rem = t["removed_2026"].astype(bool)
    for who_, col in (("field", "wood_volume_field_m3"), ("uls", "wood_volume_uls_m3")):
        r, k = t.loc[rem, col].sum(), t.loc[~rem, col].sum()
        add(_entry(f"volume.harvest.stand.{who_}_removed", r, "m3", int(rem.sum()),
                   f"{pre}, the {who_.replace('uls','ULS')} volume of the {int(rem.sum())} trees marked removed in 2026 is {r:.1f} m3, against {k:.1f} m3 in the remaining trees."))
        add(_entry(f"volume.harvest.stand.{who_}_removed_share", r / (r + k), "share", len(t),
                   f"{pre}, {r / (r + k) * 100:.1f} % of the {who_.replace('uls','ULS')} volume stands in trees marked removed in 2026."))
    for p, g in t.groupby("parzelle"):
        rm = g["removed_2026"].astype(bool)
        r, k = g.loc[rm, "wood_volume_field_m3"].sum(), g.loc[~rm, "wood_volume_field_m3"].sum()
        if r + k > 0:
            add(_entry(f"volume.harvest.parzelle_{int(p)}.field_removed_share", r / (r + k), "share", len(g),
                       f"{pre}, in Parzelle {int(p)} {r / (r + k) * 100:.1f} % of the field volume ({r:.1f} of {r + k:.1f} m3) stands in trees marked removed in 2026."))

    # --- height curve check ---------------------------------------------------
    for _, c in curves.iterrows():
        if not np.isfinite(c["loo_rmse_m"]):
            continue
        scope = "pooled" if c["scope"] == "pooled" else _slug(c["species"])
        label = "all species pooled" if c["scope"] == "pooled" else f"the species {c['species']}"
        add(_entry(f"volume.height_curve.{scope}.loo_rmse", c["loo_rmse_m"], "m", c["n"],
                   statement=f"The height-diameter curve for {label}, used for the modelled heights in the provisional {method.name} field volumes, misses a left-out matched tree's ULS height by {c['loo_rmse_m']:.2f} m (leave-one-out root-mean-square error, {int(c['n'])} trees)."))
        add(_entry(f"volume.height_curve.{scope}.loo_bias", c["loo_bias_m"], "m", c["n"],
                   statement=f"The height-diameter curve for {label}, used for the modelled heights in the provisional {method.name} field volumes, is on average {abs(c['loo_bias_m']):.2f} m {'above' if c['loo_bias_m'] >= 0 else 'below'} the ULS height of a left-out matched tree (leave-one-out bias, {int(c['n'])} trees)."))
    n_mod = int((t["height_source"] == "modelled").sum())
    add(_entry("volume.modelled_trees", n_mod, "trees", len(t),
               f"{n_mod} field trees have no matched ULS height and enter the field volume with a modelled height; they are reported separately in every provisional {method.name} total."))
    return out


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _prepare(cfg, tables_dir):
    trees = load_inputs(cfg, tables_dir)
    method, params, unit = method_from_cfg(cfg)
    plaus = list(cfg.cleaning.dbh_plausible_mm)
    ratio = dbh_ratio_check(trees, plaus)
    lo, hi = cfg.volume.get("dbh_ratio_range", [0.7, 1.4])
    if not (lo <= ratio["median"] <= hi):
        raise ValueError(f"median field/ULS diameter ratio is {ratio['median']:.3f}, outside {lo} to {hi}: "
                         "dbh_mm may be a circumference. Stopping.")
    table, curves = compute_volume_table(trees, method=method, params=params, dbh_plausible_mm=plaus,
                                         min_trees_per_species=int(cfg.volume.min_trees_per_species),
                                         breast_height_m=float(cfg.dbh.breast_height_m), diameter_unit=unit)
    return table, curves, method, params, unit, ratio, plaus


def _areas(cfg):
    try:
        return plot_areas_ha(cfg)
    except Exception:       # corners unreadable: no per-ha numbers
        return None


def run_volume(cfg, tables_dir=None) -> pd.DataFrame:
    """Write wood_volume_per_tree.csv, height_diameter_curves.csv, volume_per_parzelle.csv, volume_per_species.csv."""
    table, curves, *_ = _prepare(cfg, tables_dir)
    areas = _areas(cfg)
    area_total = float(areas.sum()) if areas is not None else None
    out = Path(cfg.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "wood_volume_per_tree.csv", index=False)
    curves.to_csv(out / "height_diameter_curves.csv", index=False)
    summary_per_parzelle(table, areas).to_csv(out / "volume_per_parzelle.csv", index=False)
    summary_per_species(table, area_total).to_csv(out / "volume_per_species.csv", index=False)
    return table


def compute_entries(cfg, tables_dir=None) -> list[dict]:
    """Statistics entries of the volume comparison."""
    table, curves, method, params, unit, ratio, plaus = _prepare(cfg, tables_dir)
    sens = cfg.volume.get("sensitivity", None)
    sens = sens.to_dict() if sens is not None else {"height_m": 1.0, "dbh_mm": 10.0, "form_factor": 0.02}
    return entries_from_table(
        table, curves, method=method, params=params, diameter_unit=unit, areas_ha=_areas(cfg), ratio=ratio,
        plausible_mm=plaus, class_edges_mm=list(cfg.cleaning.dbh_class_edges_mm),
        min_trees=int(cfg.volume.min_trees_per_species), n_boot=int(cfg.volume.get("bootstrap_resamples", 2000)),
        seed=int(cfg.volume.get("bootstrap_seed", 0)), sens=sens)
