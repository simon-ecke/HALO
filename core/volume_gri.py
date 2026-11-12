"""Wood volume per tree from the Bavarian GRI volume functions (Franz et al. 1973), computed next to core.volume.

The functions are those of v_gri() in the R package ForestElementsR. With d the diameter at breast height in cm,
h the tree height in m and C a 3 x 3 coefficient matrix per species group:

    ln(form_height) = (1, ln d, ln^2 d) . C . (1, ln h, ln^2 h)^T
    V [m3]          = max(0, pi / 40000 * d^2 * form_height)

V is 0 for d < 6.1 cm or h < 1.3 m. The result is the merchantable volume over bark.

Three volumes are reported per tree
- volume_gri_field_m3: field diameter and ULS height (a modelled height for trees without a matched instance, as in
  core.volume);
- volume_gri_uls_m3: ULS diameter and ULS height, matched trees only;
- volume_gri_practice_m3: field diameter and a height from a height-diameter curve fitted on the heights measured
  in the field, i.e. the volume a forest inventory without laser scanning would obtain.

A tree whose species has no GRI code, or whose code has no coefficients, gets no volume and is counted and reported.

The GRI volume is the standing volume over bark (Vfm m. R.). When the exported reduction factors of ForestElementsR
(config/fer_harvest_reduction.csv) are present, the harvest volume under bark (Efm o. R.) is added per tree as
efm_gri_{field,uls,practice}_m3 = volume x (1 - harvest loss) x (1 - bark reduction) of the species, as in
v_red_harvest_ubark().

Pure functions take and return DataFrames; only `run_volume_gri` writes files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from core.stats import agreement
from core.tables import has_table, read_table
from core.volume import (_class_labels, _entry, _list_text, _rel_bias, _slug, dbh_ratio_check, fit_curves,
                         load_inputs, model_heights, plot_areas_ha)

COEFFICIENT_COLUMNS = [f"c{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]   # c{i}{j} = C[i, j]
MIN_DBH_CM = 6.1
MIN_HEIGHT_M = 1.3
PRE = "With the GRI volume functions"


# ---------------------------------------------------------------------------
# The volume function
# ---------------------------------------------------------------------------

def load_coefficients(path) -> pd.DataFrame:
    """Read the coefficient file: species_code, species_name and c11 ... c33 (one row per species group)."""
    df = pd.read_csv(path)
    missing = [c for c in ["species_code", *COEFFICIENT_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"coefficient file {path} lacks the columns {', '.join(missing)}")
    return df


def gri_volume(species_code, d_mm, h_m, coefficients: pd.DataFrame) -> np.ndarray:
    """GRI volume in m3 per tree (vectorised).

    species_code: GRI species code per tree (NaN where the species has none); d_mm: diameter at breast height in mm;
    h_m: height in m; coefficients: the table of `load_coefficients`.
    NaN for a NaN diameter or height and for a code without coefficients; 0 for d < 6.1 cm or h < 1.3 m.
    """
    code, d_mm, h = np.broadcast_arrays(np.asarray(species_code, dtype=float), np.asarray(d_mm, dtype=float),
                                        np.asarray(h_m, dtype=float))
    d = d_mm / 10.0
    table = coefficients.drop_duplicates("species_code").set_index("species_code")
    pos = np.full(code.shape, -1, dtype=int)
    known = np.isfinite(code)
    pos[known] = table.index.get_indexer(code[known].astype(int))
    matrices = table[COEFFICIENT_COLUMNS].to_numpy(dtype=float).reshape(-1, 3, 3)
    c = matrices[np.where(pos >= 0, pos, 0)]

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ld = np.log(np.where(d > 0, d, 1.0))
        lh = np.log(np.where(h > 0, h, 1.0))
        left = np.stack([np.ones_like(ld), ld, ld ** 2], axis=-1)
        right = np.stack([np.ones_like(lh), lh, lh ** 2], axis=-1)
        form_height = np.exp(np.einsum("...i,...ij,...j->...", left, c, right))
        volume = np.maximum(0.0, np.pi / 40000.0 * d ** 2 * form_height)
    volume = np.where((d < MIN_DBH_CM) | (h < MIN_HEIGHT_M), 0.0, volume)
    return np.where(np.isnan(d) | np.isnan(h) | (pos < 0), np.nan, volume)


def load_harvest_factors(path) -> pd.Series:
    """Reduction factor from the standing volume over bark to the harvest volume under bark, per species code.

    Reads the file exported by scripts/export_gri_coefficients.R (species_code, harv_loss, bark_red, factor), with
    factor = (1 - harv_loss) * (1 - bark_red) as in v_red_harvest_ubark() of ForestElementsR.
    """
    df = pd.read_csv(path)
    missing = [c for c in ("species_code", "factor") if c not in df.columns]
    if missing:
        raise ValueError(f"reduction file {path} lacks the columns {', '.join(missing)}")
    return df.drop_duplicates("species_code").set_index("species_code")["factor"].astype(float)


def harvest_volume(species_code, volume, factors: pd.Series) -> np.ndarray:
    """Harvest volume under bark (Efm o. R.) in m3 from the standing volume over bark: volume x factor of the species.

    NaN where the volume is NaN or the code has no factor.
    """
    code = np.asarray(species_code, dtype=float)
    f = np.full(code.shape, np.nan)
    known = np.isfinite(code)
    f[known] = factors.reindex(code[known].astype(int)).to_numpy(dtype=float)
    return np.asarray(volume, dtype=float) * f


def species_to_code(species: pd.Series, species_codes: dict, coefficients: pd.DataFrame) -> pd.Series:
    """GRI code per tree from the field species name; NaN where the species has no code or the code has no coefficients."""
    code = pd.to_numeric(species.map(species_codes), errors="coerce")
    return code.where(code.isin(coefficients["species_code"]))


def standing_height_m(height_sheet: pd.Series, dm_above: float) -> pd.Series:
    """Height taken on the standing tree in m: sheet values above `dm_above` were typed in dm and are divided by 10."""
    h = pd.to_numeric(height_sheet, errors="coerce")
    return h.where(h <= dm_above, h / 10)


# ---------------------------------------------------------------------------
# The per-tree table
# ---------------------------------------------------------------------------

def compute_gri_table(trees: pd.DataFrame, *, coefficients: pd.DataFrame, species_codes: dict,
                      dbh_plausible_mm: Sequence[float], min_trees_per_species: int, standing_height_dm_above: float,
                      breast_height_m: float = 1.3,
                      harvest_factors: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-tree GRI volumes and the height-diameter curves of the field heights.

    With `harvest_factors` (see `load_harvest_factors`) the harvest volumes under bark efm_gri_*_m3 are appended.

    `trees` needs: tree_id, parzelle, species, dbh_field_mm, dbh_uls_mm, height_uls_m (NaN = no matched instance),
    height_standing_sheet, and optionally detection, identity_class, removed_2026, height_massband, height_vertex.
    Returns (table, curves_field); curves_field has the columns of core.volume.fit_curves.
    """
    t = trees.copy()
    lo, hi = dbh_plausible_mm
    d_field = pd.to_numeric(t["dbh_field_mm"], errors="coerce")
    t["dbh_field_valid"] = d_field.between(lo, hi)
    d_field_ok = d_field.where(t["dbh_field_valid"])
    d_uls = pd.to_numeric(t["dbh_uls_mm"], errors="coerce")
    h_uls = pd.to_numeric(t["height_uls_m"], errors="coerce")
    has_h = h_uls.notna()
    t["gri_code"] = species_to_code(t["species"], species_codes, coefficients)

    # height for the field volume: ULS height, else the curve of field diameter x ULS height (as in core.volume)
    curves_uls = fit_curves(t["species"], d_field_ok.where(has_h), h_uls, min_trees=min_trees_per_species,
                            breast=breast_height_m)
    modelled = model_heights(t["species"], d_field_ok, curves_uls, breast_height_m)
    t["height_used_m"] = h_uls.where(has_h, modelled)
    t["height_source"] = np.where(has_h, "uls", np.where(t["height_used_m"].notna(), "modelled", ""))

    # height of a practical inventory: curve of field diameter x field height, applied to every tree
    t["height_field_m"] = standing_height_m(t["height_standing_sheet"], standing_height_dm_above)
    curves_field = fit_curves(t["species"], d_field_ok, t["height_field_m"], min_trees=min_trees_per_species,
                              breast=breast_height_m)
    t["height_practice_m"] = model_heights(t["species"], d_field_ok, curves_field, breast_height_m)

    def vol(d, h):
        return pd.Series(gri_volume(t["gri_code"], d, h, coefficients), index=t.index)

    t["volume_gri_field_m3"] = vol(d_field_ok, t["height_used_m"])
    t["volume_gri_uls_m3"] = vol(d_uls.where(has_h), h_uls)
    t["volume_gri_practice_m3"] = vol(d_field_ok, t["height_practice_m"])
    for col, name in (("height_massband", "volume_gri_massband_m3"), ("height_vertex", "volume_gri_vertex_m3")):
        t[name] = vol(d_field_ok, pd.to_numeric(t[col], errors="coerce")) if col in t else np.nan
    for col in ("height_massband", "height_vertex", "detection", "identity_class", "removed_2026"):
        if col not in t:
            t[col] = np.nan if col.startswith("height") else ("" if col != "removed_2026" else False)

    cols = ["tree_id", "parzelle", "species", "gri_code", "dbh_field_mm", "dbh_uls_mm", "height_uls_m",
            "height_used_m", "height_source", "height_practice_m", "volume_gri_field_m3", "volume_gri_uls_m3",
            "volume_gri_practice_m3", "dbh_field_valid", "detection", "identity_class", "removed_2026",
            "height_massband", "height_vertex", "volume_gri_massband_m3", "volume_gri_vertex_m3", "height_field_m"]
    out = t[cols].copy()
    if harvest_factors is not None:
        for key in ("field", "uls", "practice"):
            out[f"efm_gri_{key}_m3"] = harvest_volume(out["gri_code"], out[f"volume_gri_{key}_m3"], harvest_factors)
    return out, curves_field


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------

def load_gri_inputs(cfg, tables_dir=None) -> pd.DataFrame:
    """The input of core.volume plus the standing-tree height of the field sheet."""
    t = load_inputs(cfg, tables_dir)
    sheet = read_table(cfg, "field_trees_detection.csv", tables_dir, usecols=["tree_id", "height_standing_sheet"])
    return t.merge(sheet, on="tree_id", how="left")


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

_VOLUMES = ("volume_gri_field_m3", "volume_gri_uls_m3", "volume_gri_practice_m3")
_EFM = ("efm_gri_field_m3", "efm_gri_uls_m3", "efm_gri_practice_m3")


def _volume_columns(t: pd.DataFrame) -> list[str]:
    """The volume columns of a per-tree table: the standing volumes and, when present, the harvest volumes."""
    return list(_VOLUMES) + [c for c in _EFM if c in t]


def _sums(t: pd.DataFrame) -> dict:
    out = {"n_trees": len(t), "n_field_valid": int(t["dbh_field_valid"].sum()),
           "n_without_code": int(t["gri_code"].isna().sum()),
           "n_uls_height": int((t["height_source"] == "uls").sum()),
           "n_modelled": int((t["height_source"] == "modelled").sum())}
    out.update({k.replace("_m3", "_sum_m3"): t[k].sum() for k in _volume_columns(t)})
    return out


def summary_per_parzelle(t: pd.DataFrame, areas_ha: pd.Series | None) -> pd.DataFrame:
    """Volume sums per Parzelle and per ha."""
    rows = []
    for p, g in t.groupby("parzelle"):
        r = {"parzelle": p, **_sums(g)}
        area = float(areas_ha.get(int(p), np.nan)) if areas_ha is not None else np.nan
        r["area_ha"] = area
        for k in _volume_columns(g):
            key = k.replace("_m3", "_sum_m3")
            r[key.replace("_m3", "_m3_per_ha")] = r[key] / area if area > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def summary_per_species(t: pd.DataFrame, area_ha: float | None) -> pd.DataFrame:
    """Volume sums per species and per ha of the whole surveyed area."""
    rows = []
    for sp, g in t.groupby("species"):
        r = {"species": sp, **_sums(g)}
        for k in _volume_columns(g):
            key = k.replace("_m3", "_sum_m3")
            r[key.replace("_m3", "_m3_per_ha")] = r[key] / area_ha if area_ha else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics entries
# ---------------------------------------------------------------------------

def entries_from_table(t: pd.DataFrame, curves: pd.DataFrame, *, evaluate: Callable, areas_ha: pd.Series | None,
                       class_edges_mm: Sequence[float], min_trees: int, n_boot: int, seed: int,
                       old_totals: dict | None = None) -> list[dict]:
    """Statistics entries for the GRI volumes. Pure.

    evaluate(species, d_mm, h_m) -> m3 per tree (used for the sample trees at the ULS height).
    old_totals: {"method": name, "field": m3, "uls": m3} of the existing volume method, or None.
    """
    out: list[dict] = []
    add = out.append
    area_total = float(areas_ha.sum()) if areas_ha is not None and len(areas_ha) else np.nan
    f_col, u_col, p_col = "volume_gri_field_m3", "volume_gri_uls_m3", "volume_gri_practice_m3"

    # --- trees without a GRI code ------------------------------------------
    nocode = t[t["gri_code"].isna()]
    names = sorted(nocode["species"].dropna().astype(str).unique())
    add(_entry("volume_gri.species_without_code", len(nocode), "trees", len(t),
               f"{PRE}, {len(nocode)} of {len(t)} field trees have no volume because their species has no GRI code or no "
               f"coefficients" + (f" ({_list_text(names)})." if names else ".")))

    # --- totals and per ha ---------------------------------------------------
    def total_entries(scope, label, g, per_ha_area):
        uh = g["height_source"] == "uls"
        n_mod = int((g["height_source"] == "modelled").sum())
        vals = [("field_with_modelled", g[f_col].sum(), int(g[f_col].notna().sum()),
                 f"field diameter and ULS height, including {n_mod} trees whose height is modelled from the height-diameter curve"),
                ("field_without_modelled", g.loc[uh, f_col].sum(), int(g.loc[uh, f_col].notna().sum()),
                 "field diameter and ULS height, only trees with a matched ULS height (no modelled heights)"),
                ("uls", g[u_col].sum(), int(g[u_col].notna().sum()),
                 "ULS diameter and ULS height, only trees with a matched ULS instance and diameter"),
                ("practice", g[p_col].sum(), int(g[p_col].notna().sum()),
                 "field diameter and a height from the curve of field-measured heights, applied to every tree")]
        for key, v, n, desc in vals:
            add(_entry(f"volume_gri.total.{key}.{scope}", v, "m3", n,
                       f"{PRE}, the total volume of {label} is {v:.1f} m3 from {n} trees ({desc})."))
            if per_ha_area and per_ha_area > 0:
                add(_entry(f"volume_gri.per_ha.{key}.{scope}", v / per_ha_area, "m3/ha", n,
                           f"{PRE}, the volume of {label} is {v / per_ha_area:.1f} m3/ha of the {per_ha_area:.2f} ha surveyed plot area ({desc})."))

    total_entries("stand", "the whole stand", t, area_total)

    # --- harvest volume under bark (Efm o. R.), stand level -------------------------------
    if "efm_gri_field_m3" in t:
        how = ("from the standing volume with the reduction factors of ForestElementsR (v_red_harvest_ubark)")
        for key, col, desc in (
                ("field_with_modelled", "efm_gri_field_m3", "field diameter and ULS height, including trees whose height is modelled"),
                ("uls", "efm_gri_uls_m3", "ULS diameter and ULS height, only trees with a matched ULS instance and diameter"),
                ("practice", "efm_gri_practice_m3", "field diameter and a height from the curve of field-measured heights, applied to every tree")):
            v, n = t[col].sum(), int(t[col].notna().sum())
            add(_entry(f"volume_gri.efm.total.{key}.stand", v, "m3", n,
                       f"{PRE}, the harvest volume under bark (Efm o. R.) of the whole stand is {v:.1f} m3 from {n} trees ({desc}), {how}."))
            if area_total and area_total > 0:
                add(_entry(f"volume_gri.efm.per_ha.{key}.stand", v / area_total, "m3/ha", n,
                           f"{PRE}, the harvest volume under bark (Efm o. R.) is {v / area_total:.1f} m3/ha of the {area_total:.2f} ha surveyed plot area ({desc}), {how}."))
    for sp in sorted(t["species"].dropna().unique()):
        total_entries(_slug(sp), f"the species {sp}", t[t["species"] == sp], area_total)

    # --- per-tree difference ULS - field ---------------------------------------
    matched = t[t["height_source"].eq("uls") & t[f_col].notna() & t[u_col].notna()].copy()
    matched["class"] = _class_labels(matched["dbh_field_mm"], class_edges_mm)
    subsets = {"all_matched": matched, "detected_only": matched[matched["detection"] == "detected"]}
    who = {"all_matched": "matched trees (detected and uncertain)", "detected_only": "detected trees"}
    same = "both volumes use the same ULS height, so this is the diameter difference only"

    def compare(prefix, label, g, subset):
        n = len(g)
        if n < 2:
            return
        fld, uls = g[f_col].to_numpy(), g[u_col].to_numpy()
        a = agreement(pd.Series(fld), pd.Series(uls), n_boot=n_boot, seed=seed)
        rb, rb_lo, rb_hi = _rel_bias(fld, uls, n_boot, seed)
        sign = lambda v: "higher" if v >= 0 else "lower"
        iv = lambda k: f"{a[k + '_ci_low']:+.4f} to {a[k + '_ci_high']:+.4f} m3"
        add(_entry(f"{prefix}.bias", a["bias"], "m3", n, subset=subset, ci=(a["bias_ci_low"], a["bias_ci_high"]),
                   statement=f"{PRE}, for {label} the ULS volume is on average {abs(a['bias']):.4f} m3 {sign(a['bias'])} than the field volume per tree (95 % bootstrap interval {iv('bias')}, {n} {who[subset]}); {same}."))
        add(_entry(f"{prefix}.relative_bias", rb, "share", n, subset=subset, ci=(rb_lo, rb_hi),
                   statement=f"{PRE}, for {label} the ULS volume is {abs(rb) * 100:.1f} % {sign(rb)} than the field volume (summed over {n} {who[subset]}, 95 % bootstrap interval {rb_lo * 100:+.1f} % to {rb_hi * 100:+.1f} %); {same}."))
        add(_entry(f"{prefix}.rmse", a["rmse"], "m3", n, subset=subset, ci=(a["rmse_ci_low"], a["rmse_ci_high"]),
                   statement=f"{PRE}, for {label} the root-mean-square difference of ULS minus field volume is {a['rmse']:.4f} m3 per tree ({n} {who[subset]}); {same}."))
        add(_entry(f"{prefix}.median", a["median"], "m3", n, subset=subset, ci=(a["median_ci_low"], a["median_ci_high"]),
                   statement=f"{PRE}, for {label} the median difference of ULS minus field volume is {a['median']:+.4f} m3 per tree (95 % bootstrap interval {iv('median')}, {n} {who[subset]}); {same}."))
        add(_entry(f"{prefix}.mad", a["mad"], "m3", n, subset=subset, ci=(a["mad_ci_low"], a["mad_ci_high"]),
                   statement=f"{PRE}, for {label} the median absolute deviation of the ULS minus field volume is {a['mad']:.4f} m3 per tree ({n} {who[subset]}); {same}."))

    for sub, g in subsets.items():
        compare(f"volume_gri.tree_error.{sub}.all", "all species", g, sub)
        for sp, gs in g.groupby("species"):
            if len(gs) >= min_trees:
                compare(f"volume_gri.tree_error.{sub}.species_{_slug(sp)}", f"the species {sp}", gs, sub)
        for cl, gc in g.groupby("class", observed=True):
            if len(gc) >= min_trees:
                compare(f"volume_gri.tree_error.{sub}.class_{_slug(cl)}", f"field diameters of {cl}", gc, sub)

    # --- practice against field volume -------------------------------------------
    both_pf = t[t[f_col].notna() & t[p_col].notna()]
    if len(both_pf) >= 2:
        fld, prc = both_pf[f_col].to_numpy(), both_pf[p_col].to_numpy()
        a = agreement(pd.Series(fld), pd.Series(prc), n_boot=n_boot, seed=seed)
        rb, rb_lo, rb_hi = _rel_bias(fld, prc, n_boot, seed)
        n = len(both_pf)
        sign = "higher" if a["bias"] >= 0 else "lower"
        what = "the height of the field-height curve instead of the ULS height (or the modelled ULS-based height), field diameter throughout"
        add(_entry("volume_gri.practice_vs_field.bias", a["bias"], "m3", n, subset="all",
                   ci=(a["bias_ci_low"], a["bias_ci_high"]),
                   statement=f"{PRE}, the practice volume is on average {abs(a['bias']):.4f} m3 {sign} than the field volume per tree (95 % bootstrap interval {a['bias_ci_low']:+.4f} to {a['bias_ci_high']:+.4f} m3, {n} trees); the practice volume uses {what}."))
        add(_entry("volume_gri.practice_vs_field.relative_bias", rb, "share", n, subset="all", ci=(rb_lo, rb_hi),
                   statement=f"{PRE}, the practice volume is {abs(rb) * 100:.1f} % {'higher' if rb >= 0 else 'lower'} than the field volume (summed over {n} trees, 95 % bootstrap interval {rb_lo * 100:+.1f} % to {rb_hi * 100:+.1f} %)."))
        add(_entry("volume_gri.practice_vs_field.rmse", a["rmse"], "m3", n, subset="all",
                   ci=(a["rmse_ci_low"], a["rmse_ci_high"]),
                   statement=f"{PRE}, the root-mean-square difference of practice minus field volume is {a['rmse']:.4f} m3 per tree ({n} trees)."))

    # --- stand gap split ----------------------------------------------------------
    f_all, u_all = t[f_col].sum(), t[u_col].sum()
    both = t[f_col].notna() & t[u_col].notna()
    missed = t.loc[t[f_col].notna() & t[u_col].isna(), f_col]
    diam = t.loc[both, u_col].sum() - t.loc[both, f_col].sum()
    extra = t.loc[t[f_col].isna() & t[u_col].notna(), u_col]
    gap = u_all - f_all
    add(_entry("volume_gri.gap.total", gap, "m3", len(t),
               f"{PRE}, the ULS stand volume is {abs(gap):.1f} m3 {'above' if gap >= 0 else 'below'} the field stand volume (field total includes modelled heights for unmatched trees)."))
    add(_entry("volume_gri.gap.trees_missed", -missed.sum(), "m3", len(missed),
               f"{PRE}, {missed.sum():.1f} m3 of the field volume stands in {len(missed)} trees without a ULS volume (no matched instance or no ULS diameter), which lowers the ULS total by that amount."))
    add(_entry("volume_gri.gap.diameter_difference", diam, "m3", int(both.sum()),
               f"{PRE}, on the {int(both.sum())} trees that have both volumes the ULS diameter changes the stand volume by {diam:+.1f} m3 against the field diameter at the same ULS height."))
    add(_entry("volume_gri.gap.uls_only_trees", extra.sum(), "m3", len(extra),
               f"{PRE}, {len(extra)} trees without a usable field diameter carry {extra.sum():.1f} m3 of ULS volume that has no field counterpart."))

    # --- sample trees (the tape height is the reference, Vertex is a separate check) --------
    s = t[t[f_col].notna()]
    sets = {"massband": s[s["volume_gri_massband_m3"].notna() & s[u_col].notna()],
            "vertex": s[s["volume_gri_vertex_m3"].notna() & s[u_col].notna()]}
    for key, g in sets.items():
        if len(g) == 0:
            continue
        uh = pd.Series(evaluate(g["species"], g["dbh_field_mm"], g["height_uls_m"]), index=g.index)
        ref = g[f"volume_gri_{key}_m3"]
        add(_entry(f"volume_gri.sample.{key}_total", ref.sum(), "m3", len(g),
                   f"{PRE}, the {len(g)} sample trees with a {key} height have {ref.sum():.2f} m3 at the {key} height and {uh.sum():.2f} m3 at the ULS height, all at the field diameter."))
        if len(g) >= 2:
            a = agreement(ref, uh, n_boot=n_boot, seed=seed)
            note = "the tape height is the reference" if key == "massband" else "the Vertex height is a separate check, not the reference"
            add(_entry(f"volume_gri.sample.uls_height_vs_{key}.bias", a["bias"], "m3", len(g),
                       ci=(a["bias_ci_low"], a["bias_ci_high"]),
                       statement=f"{PRE}, using the ULS height instead of the {key} height changes the volume of a sample tree by {a['bias']:+.3f} m3 on average (95 % bootstrap interval {a['bias_ci_low']:+.3f} to {a['bias_ci_high']:+.3f} m3, {len(g)} trees, field diameter throughout; {note})."))
            add(_entry(f"volume_gri.sample.uls_height_vs_{key}.median", a["median"], "m3", len(g),
                       ci=(a["median_ci_low"], a["median_ci_high"]),
                       statement=f"{PRE}, using the ULS height instead of the {key} height changes the volume of a sample tree by a median of {a['median']:+.3f} m3 ({len(g)} trees, field diameter throughout; {note})."))
    both_h = s[s["volume_gri_massband_m3"].notna() & s["volume_gri_vertex_m3"].notna()]
    if len(both_h) >= 2:
        a = agreement(both_h["volume_gri_massband_m3"], both_h["volume_gri_vertex_m3"], n_boot=n_boot, seed=seed)
        add(_entry("volume_gri.sample.vertex_vs_massband.bias", a["bias"], "m3", len(both_h),
                   ci=(a["bias_ci_low"], a["bias_ci_high"]),
                   statement=f"{PRE}, the Vertex height instead of the tape height changes the volume of a sample tree by {a['bias']:+.3f} m3 on average ({len(both_h)} trees with both heights, field diameter throughout)."))

    # --- harvest -----------------------------------------------------------------
    rem = t["removed_2026"].astype(bool)
    for who_, col in (("field", f_col), ("uls", u_col)):
        r, k = t.loc[rem, col].sum(), t.loc[~rem, col].sum()
        label = who_.replace("uls", "ULS")
        add(_entry(f"volume_gri.harvest.stand.{who_}_removed", r, "m3", int(rem.sum()),
                   f"{PRE}, the {label} volume of the {int(rem.sum())} trees marked removed in 2026 is {r:.1f} m3, against {k:.1f} m3 in the remaining trees."))
        if r + k > 0:
            add(_entry(f"volume_gri.harvest.stand.{who_}_removed_share", r / (r + k), "share", len(t),
                       f"{PRE}, {r / (r + k) * 100:.1f} % of the {label} volume stands in trees marked removed in 2026."))

    # --- height curve of the field heights --------------------------------------------
    pooled_n = int(curves.loc[curves["scope"] == "pooled", "n"].iloc[0])
    add(_entry("volume_gri.height_field_trees", pooled_n, "trees", len(t),
               f"{pooled_n} trees with a valid field diameter and a field-measured height enter the height-diameter curves behind the practice volume."))
    for _, c in curves.iterrows():
        if not np.isfinite(c["loo_rmse_m"]):
            continue
        scope = "pooled" if c["scope"] == "pooled" else _slug(c["species"])
        label = "all species pooled" if c["scope"] == "pooled" else f"the species {c['species']}"
        add(_entry(f"volume_gri.height_curve_field.{scope}.loo_rmse", c["loo_rmse_m"], "m", c["n"],
                   statement=f"The height-diameter curve for {label}, fitted on field diameter and field-measured height, misses a left-out tree's measured height by {c['loo_rmse_m']:.2f} m (leave-one-out root-mean-square error, {int(c['n'])} trees)."))
        add(_entry(f"volume_gri.height_curve_field.{scope}.loo_bias", c["loo_bias_m"], "m", c["n"],
                   statement=f"The height-diameter curve for {label}, fitted on field diameter and field-measured height, is on average {abs(c['loo_bias_m']):.2f} m {'above' if c['loo_bias_m'] >= 0 else 'below'} the measured height of a left-out tree (leave-one-out bias, {int(c['n'])} trees)."))

    # --- against the existing volume method -------------------------------------------
    if old_totals is not None:
        for who_, new in (("field", t[f_col].sum()), ("uls", t[u_col].sum())):
            old = old_totals[who_]
            if not np.isfinite(old) or old == 0:
                continue
            add(_entry(f"volume_gri.vs_old.{who_}", new - old, "m3", len(t),
                       f"{PRE}, the {who_.replace('uls', 'ULS')} stand volume is {new:.1f} m3, against {old:.1f} m3 with the {old_totals['method']} volume method ({(new / old - 1) * 100:+.1f} %, a difference of {new - old:+.1f} m3)."))
    return out


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _settings(cfg) -> dict:
    return dict(cfg.raw.get("volume_gri") or {})


def _setup(cfg):
    """(coefficients, species_codes, problem). `problem` is a one-line text when the calculation cannot run."""
    s = _settings(cfg)
    path = Path(str(s.get("params_file", "config/volume_gri_coefficients.csv")))
    if not path.is_absolute():
        path = cfg.project_root / path
    codes = dict(s.get("species_codes") or {})
    problems = []
    if not path.is_file():
        problems.append(f"the coefficient file {path} is missing (`Rscript scripts/export_gri_coefficients.R` creates it)")
    if not codes:
        problems.append("volume_gri.species_codes is empty (set the GRI code of each field species in the config)")
    if problems:
        return None, codes, "volume_gri skipped: " + " and ".join(problems) + "."
    return load_coefficients(path), codes, None


def _harvest_factors(cfg) -> tuple[pd.Series | None, Path]:
    """(factors, path) of the exported reduction factors, which lie next to the coefficient file; None if missing."""
    path = Path(str(_settings(cfg).get("params_file", "config/volume_gri_coefficients.csv")))
    if not path.is_absolute():
        path = cfg.project_root / path
    path = path.parent / "fer_harvest_reduction.csv"
    return (load_harvest_factors(path) if path.is_file() else None), path


def _prepare(cfg, tables_dir, coefficients, codes):
    trees = load_gri_inputs(cfg, tables_dir)
    plaus = list(cfg.cleaning.dbh_plausible_mm)
    ratio = dbh_ratio_check(trees, plaus)
    lo, hi = cfg.volume.get("dbh_ratio_range", [0.7, 1.4])
    if not (lo <= ratio["median"] <= hi):
        raise ValueError(f"median field/ULS diameter ratio is {ratio['median']:.3f}, outside {lo} to {hi}: "
                         "dbh_mm may be a circumference. Stopping.")
    table, curves = compute_gri_table(
        trees, coefficients=coefficients, species_codes=codes, dbh_plausible_mm=plaus,
        min_trees_per_species=int(_settings(cfg).get("min_trees_per_species", 20)),
        standing_height_dm_above=float(cfg.cleaning.standing_height_dm_above),
        breast_height_m=float(cfg.dbh.breast_height_m), harvest_factors=_harvest_factors(cfg)[0])
    return table, curves


def _areas(cfg):
    try:
        return plot_areas_ha(cfg)
    except Exception:       # corners unreadable: no per-ha numbers
        return None


def run_volume_gri(cfg, tables_dir=None) -> pd.DataFrame | None:
    """Write wood_volume_gri_per_tree.csv, height_diameter_curves_field.csv, volume_gri_per_parzelle.csv and
    volume_gri_per_species.csv. Without coefficient file or species codes it prints one line and returns None."""
    coefficients, codes, problem = _setup(cfg)
    if problem:
        print(problem)
        return None
    factors, red_path = _harvest_factors(cfg)
    if factors is None:
        print(f"volume_gri: harvest volume (Efm o. R.) skipped, the reduction file {red_path} is missing "
              "(`Rscript scripts/export_gri_coefficients.R` creates it).")
    table, curves = _prepare(cfg, tables_dir, coefficients, codes)
    areas = _areas(cfg)
    area_total = float(areas.sum()) if areas is not None else None
    out = Path(cfg.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "wood_volume_gri_per_tree.csv", index=False)
    curves.to_csv(out / "height_diameter_curves_field.csv", index=False)
    summary_per_parzelle(table, areas).to_csv(out / "volume_gri_per_parzelle.csv", index=False)
    summary_per_species(table, area_total).to_csv(out / "volume_gri_per_species.csv", index=False)
    return table


def compute_entries(cfg, tables_dir=None) -> list[dict]:
    """Statistics entries of the GRI volumes; empty when the coefficient file or the species codes are missing."""
    coefficients, codes, problem = _setup(cfg)
    if problem:
        return []
    table, curves = _prepare(cfg, tables_dir, coefficients, codes)
    old = None
    if has_table(cfg, "wood_volume_per_tree.csv", tables_dir):
        o = read_table(cfg, "wood_volume_per_tree.csv", tables_dir)
        method = o["volume_method"].dropna().iloc[0] if "volume_method" in o and o["volume_method"].notna().any() else "form_factor"
        old = {"method": method, "field": o["wood_volume_field_m3"].sum(), "uls": o["wood_volume_uls_m3"].sum()}
    s = _settings(cfg)
    code_of = lambda sp: species_to_code(pd.Series(sp), codes, coefficients).to_numpy()
    evaluate = lambda sp, d, h: gri_volume(code_of(sp), d, h, coefficients)
    return entries_from_table(
        table, curves, evaluate=evaluate, areas_ha=_areas(cfg), class_edges_mm=list(cfg.cleaning.dbh_class_edges_mm),
        min_trees=int(s.get("min_trees_per_species", 20)), n_boot=int(s.get("bootstrap_resamples", 2000)),
        seed=int(s.get("bootstrap_seed", 0)), old_totals=old)
