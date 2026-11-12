"""Volume increment of the stand by three methods, following the R package ForestElementsR.

All volumes are GRI volumes (core.volume_gri), i.e. standing volume over bark (Vfm m. R.); increments are m3 per ha
and year. The harvest volume under bark (Efm o. R.) follows with the volume-weighted reduction factor of the trees
(v_red_harvest_ubark). The package data (yield tables, growth functions, reduction factors) are exported once by
scripts/export_gri_coefficients.R; the calculation is done here.

1. Yield table (stand level, depends on age). Per Parzelle: main species = species with the largest basal area;
   d_q = quadratic mean of the valid field diameters; h_q = height of the d_q tree from the Naeslund curve of field
   diameter x ULS height; site index = site_index(age, h_q) of the yield table assigned to the main species;
   increment = pai of the table at (age, site index), multiplied by the stocking level (observed basal area over table
   basal area) where that is below 1, as in growth_est_ytable().
2. Tree growth (per tree, depends on age). Diameter and height after `horizon_years` from d_age_gnfi3() and
   h_age_gnfi3() (growth functions of the third national forest inventory, species coding tum_wwk_short);
   increment = (GRI volume after - GRI volume now) / horizon. All standing trees grow; there is no mortality or removal.
3. Measured (no age). The classical periodic increment of two surveys (2020 and the current one):
       increment = V_alive(t2) + V_removed(between t1 and t2) - V_alive(t1).
   Trees alive at both dates (survivors) are measured twice; trees removed in an earlier thinning (flag in the sheet) or
   without a current diameter are alive at t1 and removed in between. The height at each date comes from the
   height-diameter curve of that survey (2020 diameter x 2020 height; current diameter x current standing height).
   Neither a removal year nor a diameter at removal is recorded, so a removed tree enters with its 2020 volume plus an
   assumed growth (`removed_growth_share` of the period at the relative volume growth of the survivors of its
   Parzelle; 0 = removed at the 2020 survey). No ingrowth is recorded. Survivors with a diameter change outside
   `dbh_change_plausible_mm` are recording errors and are left out at both dates. The increment per ha and year divides
   by the growing seasons between the surveys (`growing_seasons` per survey year). The variant `static_curve` keeps the
   survivors only and one current curve for both dates.

The interpolation rules of ForestElementsR are ported exactly: linear inter- and extrapolation through the two
nearest tabulated points; an age outside the table is clamped to its coverage. Where the package warns (age or site
index outside the table) the result carries a flag instead.

Pure functions take and return DataFrames or arrays; only `run_increment` writes files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from core import volume_gri
from core.tables import has_table, read_table
from core.volume import _entry, _list_text, _slug, fit_curves, model_heights

SI_VARIABLE = "h_q_m"
PAI, MAI, BA = "pai_m3_ha_yr", "mai_m3_ha_yr", "ba_m2_ha"
PER_TREE_COLUMNS = ["tree_id", "parzelle", "species", "status", "dbh_2020_mm", "dbh_mm", "dbh_change_mm",
                    "height_2020_curve_m", "height_now_curve_m", "v_2020_m3", "v_now_m3", "survey_year"]
LEDGER_COLUMNS = ["parzelle", "survey_year", "growing_seasons", "area_ha", "n_survivors", "n_removed_earlier", "n_missing",
                  "n_excluded", "v_alive_t1_m3", "v_alive_t2_m3", "v_removed_2020_m3", "v_removed_m3", "survivor_growth_m3",
                  "survivor_rel_growth", "increment_m3", "pai_measured_m3_ha_yr", "pai_static_curve_m3_ha_yr",
                  "dbh_increment_mm_yr", "efm_factor", "pai_measured_efm_m3_ha_yr"]
REMOVED_STATUSES = ("removed_earlier", "missing")
EXCLUDED_STATUSES = ("no_2020_dbh", "excluded_implausible_dbh", "excluded_dbh_change", "no_volume")
HEIGHT_2020_BOUNDS_M = (2.0, 60.0)     # 2020 heights outside are recording errors and do not enter the curve
BY_AGE_COLUMNS = ["parzelle", "age", "main_species", "gri_code", "yield_table", "d_q_cm", "h_q_m", "ba_m2_ha",
                  "site_index", "si_extrapolated", "age_outside_table", "pai_table_m3_ha_yr", "mai_table_m3_ha_yr",
                  "stocking_level", "pai_yield_table_m3_ha_yr", "pai_tree_growth_m3_ha_yr", "n_trees_tree_growth",
                  "efm_factor", "pai_yield_table_efm_m3_ha_yr", "pai_tree_growth_efm_m3_ha_yr"]


# ---------------------------------------------------------------------------
# Yield tables (ports of site_index(), ytable_lookup(), stocking_level())
# ---------------------------------------------------------------------------

@dataclass
class YieldTable:
    """One yield table: values[variable] has one row per age and one column per site index (NaN = not tabulated)."""
    code: int
    name: str
    site_indexes: np.ndarray
    ages: np.ndarray
    values: dict
    site_index_variables: tuple = (SI_VARIABLE,)


def load_yield_tables(path) -> dict[int, YieldTable]:
    """Read config/fer_yield_tables.json: yield table per species code (coding tum_wwk_short)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for code, t in raw["tables"].items():
        out[int(code)] = YieldTable(
            code=int(code), name=t["table_name"], site_indexes=np.asarray(t["site_indexes"], dtype=float),
            ages=np.asarray(t["age"], dtype=float),
            values={k: np.asarray(v, dtype=float) for k, v in t["values"].items()},
            site_index_variables=tuple(t.get("site_index_variable", [SI_VARIABLE])))
    return out


def _two_nearest(distance: np.ndarray) -> np.ndarray:
    """Indices of the two smallest distances (ties by position, NaN left out), as which.minn(rank(dist), n = 2)."""
    order = np.argsort(distance, kind="stable")[:2]
    return order[~np.isnan(distance[order])]


def _line(x, x_pair, y_pair) -> float:
    """Straight line through two points evaluated at x; NaN when fewer than two points are available."""
    if len(x_pair) < 2 or np.isnan(y_pair).any():
        return np.nan
    return float((x - x_pair[0]) * (y_pair[1] - y_pair[0]) / (x_pair[1] - x_pair[0]) + y_pair[0])


def age_outside_table(table: YieldTable, age: float) -> bool:
    """True where the age lies outside the coverage of the table (the package then uses the nearest tabulated age)."""
    return bool(age < table.ages.min() or age > table.ages.max())


def si_outside_table(table: YieldTable, si: float) -> bool:
    """True where the site index lies outside the table, so that the lookup extrapolates."""
    return bool(si < table.site_indexes.min() or si > table.site_indexes.max())


def age_slice(table: YieldTable, variable: str, age: float) -> np.ndarray:
    """Value of `variable` at `age` for every site index (ytable_age_slice()): linear through the two nearest ages."""
    age = min(max(age, table.ages.min()), table.ages.max())
    mat = table.values[variable]
    out = np.full(mat.shape[1], np.nan)
    for j in range(mat.shape[1]):
        ok = ~np.isnan(mat[:, j])
        ages, y = table.ages[ok], mat[ok, j]
        i = _two_nearest(np.abs(age - ages))
        out[j] = _line(age, ages[i], y[i])
    return out


def ytable_lookup(table: YieldTable, age: float, si: float, variable: str) -> float:
    """Table value at (age, site index), interpolated (or extrapolated) through the two nearest site indexes."""
    if variable not in table.values:
        raise KeyError(f"variable {variable} is not contained in the yield table {table.name}")
    sl = age_slice(table, variable, age)
    i = _two_nearest(np.abs(si - table.site_indexes))
    return _line(si, table.site_indexes[i], sl[i])


def site_index(table: YieldTable, age: float, size: float, variable: str = SI_VARIABLE) -> float:
    """Site index from the size (here h_q in m) at `age`: inverse lookup through the two nearest tabulated sizes."""
    if variable not in table.site_index_variables:
        raise ValueError(f"{variable} is not a site index variable of the yield table {table.name}")
    sl = age_slice(table, variable, age)
    i = _two_nearest(np.abs(size - sl))
    return _line(size, sl[i], table.site_indexes[i])


def stocking_level(table: YieldTable, ba: float, age: float, si: float) -> float:
    """Observed basal area over the table basal area at (age, site index); a negative value is set to 0."""
    level = ba / ytable_lookup(table, age, si, BA)
    return 0.0 if level < 0 else float(level)


# ---------------------------------------------------------------------------
# Growth functions of the third national forest inventory
# ---------------------------------------------------------------------------

def load_growth_parameters(path) -> pd.DataFrame:
    """Read config/fer_gnfi3_parameters.csv, indexed by species code (d_alpha, d_beta, d_gamma, h_alpha, ...)."""
    df = pd.read_csv(path)
    need = ["species_code"] + [f"{p}_{c}" for p in "dh" for c in ("alpha", "beta", "gamma")]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"growth parameter file {path} lacks the columns {', '.join(missing)}")
    return df.drop_duplicates("species_code").set_index("species_code")


def _grow(kind: str, species_code, age, known, age_known, params: pd.DataFrame) -> np.ndarray:
    code, age, known, age_known = np.broadcast_arrays(np.asarray(species_code, dtype=float), np.asarray(age, dtype=float),
                                                      np.asarray(known, dtype=float), np.asarray(age_known, dtype=float))
    p = pd.DataFrame(index=np.where(np.isfinite(code), code, -1).astype(int).ravel())
    for c in ("alpha", "beta", "gamma"):
        p[c] = params[f"{kind}_{c}"].reindex(p.index).to_numpy(dtype=float)
    alpha, beta, gamma = (p[c].to_numpy().reshape(code.shape) for c in ("alpha", "beta", "gamma"))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return gamma * (known / gamma) ** np.exp(-beta / alpha * (age ** alpha - age_known ** alpha))


def d_age_gnfi3(species_code, age, dbh_cm_known, age_known, params: pd.DataFrame) -> np.ndarray:
    """Diameter in cm at `age` from the diameter known at `age_known` (d_age_gnfi3(); codes of tum_wwk_short).

    x(t) = gamma * (x0 / gamma) ^ exp(-beta / alpha * (t ^ alpha - t0 ^ alpha)); NaN for a code without parameters.
    """
    return _grow("d", species_code, age, dbh_cm_known, age_known, params)


def h_age_gnfi3(species_code, age, h_m_known, age_known, params: pd.DataFrame) -> np.ndarray:
    """Height in m at `age` from the height known at `age_known` (h_age_gnfi3(); same form as `d_age_gnfi3`)."""
    return _grow("h", species_code, age, h_m_known, age_known, params)


# ---------------------------------------------------------------------------
# Stand structure per Parzelle
# ---------------------------------------------------------------------------

def basal_area_m2(d_mm) -> np.ndarray:
    """Basal area of a tree in m2 from the diameter in mm."""
    return np.pi / 4.0 * (np.asarray(d_mm, dtype=float) / 1000.0) ** 2


def height_curves(trees: pd.DataFrame, *, min_trees: int, breast: float) -> pd.DataFrame:
    """Height-diameter curves of field diameter x ULS height, exactly as the field volume of core.volume_gri."""
    d = pd.to_numeric(trees["dbh_field_mm"], errors="coerce").where(trees["dbh_field_valid"].astype(bool))
    h = pd.to_numeric(trees["height_uls_m"], errors="coerce")
    return fit_curves(trees["species"], d.where(h.notna()), h, min_trees=min_trees, breast=breast)


def group_structure(g: pd.DataFrame, area_ha: float, curves: pd.DataFrame, breast: float) -> dict:
    """Stand structure of one group of trees: main species, d_q, h_q and basal area per ha.

    Only trees with a valid field diameter count; the main species has the largest basal area; h_q is the height of the
    d_q tree on the curve of the main species (pooled curve where it has none).
    """
    v = g[g["dbh_field_valid"].astype(bool)]
    d = pd.to_numeric(v["dbh_field_mm"], errors="coerce")
    out = {"area_ha": area_ha, "n_valid": int(len(v)), "main_species": "", "gri_code": np.nan, "d_q_cm": np.nan,
           "h_q_m": np.nan, "ba_m2_ha": np.nan}
    if len(v) == 0:
        return out
    ba = pd.Series(basal_area_m2(d), index=v.index)
    main = ba.groupby(v["species"]).sum().idxmax()
    codes = v.loc[v["species"] == main, "gri_code"].dropna()
    d_q_mm = float(np.sqrt((d ** 2).mean()))
    h_q = model_heights(pd.Series([main]), pd.Series([d_q_mm]), curves, breast).iloc[0]
    out.update(main_species=main, gri_code=float(codes.iloc[0]) if len(codes) else np.nan, d_q_cm=d_q_mm / 10.0,
               h_q_m=float(h_q), ba_m2_ha=float(ba.sum() / area_ha) if area_ha > 0 else np.nan)
    return out


def stand_structure(trees: pd.DataFrame, areas_ha: pd.Series, curves: pd.DataFrame, breast: float) -> pd.DataFrame:
    """One row per Parzelle (index parzelle) with `group_structure`, plus the row 'stand' for all trees together."""
    rows = {}
    for p, g in trees.groupby("parzelle"):
        rows[int(p)] = group_structure(g, float(areas_ha.get(int(p), np.nan)), curves, breast)
    area_total = float(sum(r["area_ha"] for r in rows.values() if np.isfinite(r["area_ha"])))
    rows["stand"] = group_structure(trees, area_total, curves, breast)
    return pd.DataFrame.from_dict(rows, orient="index")


def volume_weighted_factor(code, volume, factors: pd.Series | None) -> float:
    """Volume-weighted reduction factor from standing volume to Efm o. R.; NaN without factors or volume."""
    if factors is None:
        return np.nan
    v = np.asarray(volume, dtype=float)
    f = volume_gri.harvest_volume(code, np.ones_like(v), factors)
    ok = np.isfinite(v) & np.isfinite(f) & (v > 0)
    return float((v[ok] * f[ok]).sum() / v[ok].sum()) if ok.any() else np.nan


# ---------------------------------------------------------------------------
# Method 1 and 2
# ---------------------------------------------------------------------------

def yield_table_values(structure: Mapping, age: float, tables: Mapping[int, YieldTable]) -> dict:
    """Yield-table increment of one stand structure at `age`: site index, table pai and mai, stocking level, adjusted pai."""
    out = {"yield_table": "", "site_index": np.nan, "si_extrapolated": False, "age_outside_table": False,
           "pai_table_m3_ha_yr": np.nan, "mai_table_m3_ha_yr": np.nan, "stocking_level": np.nan,
           "pai_yield_table_m3_ha_yr": np.nan}
    code = structure["gri_code"]
    table = tables.get(int(code)) if np.isfinite(code) else None
    if table is None or not np.isfinite(structure["h_q_m"]):
        return out
    out["yield_table"] = table.name
    si = site_index(table, age, structure["h_q_m"], SI_VARIABLE)
    out["age_outside_table"] = age_outside_table(table, age)
    if not np.isfinite(si):
        return out
    pai, mai = ytable_lookup(table, age, si, PAI), ytable_lookup(table, age, si, MAI)
    level = stocking_level(table, structure["ba_m2_ha"], age, si)
    out.update(site_index=si, si_extrapolated=si_outside_table(table, si), pai_table_m3_ha_yr=pai,
               mai_table_m3_ha_yr=mai, stocking_level=level,
               pai_yield_table_m3_ha_yr=pai * min(level, 1.0))
    return out


def tree_growth_increment(trees: pd.DataFrame, age: float, horizon: float, coefficients: pd.DataFrame,
                          growth: pd.DataFrame) -> pd.Series:
    """Volume increment per tree in m3/yr over `horizon` years from the growth functions; NaN for an unusable tree.

    trees needs gri_code, dbh_field_mm, dbh_field_valid and height_used_m. All trees survive the horizon.
    """
    d_mm = pd.to_numeric(trees["dbh_field_mm"], errors="coerce").where(trees["dbh_field_valid"].astype(bool))
    h = pd.to_numeric(trees["height_used_m"], errors="coerce")
    code = trees["gri_code"].to_numpy(dtype=float)
    d2 = d_age_gnfi3(code, age + horizon, d_mm.to_numpy(dtype=float) / 10.0, age, growth) * 10.0
    h2 = h_age_gnfi3(code, age + horizon, h.to_numpy(dtype=float), age, growth)
    v0 = volume_gri.gri_volume(code, d_mm.to_numpy(dtype=float), h.to_numpy(dtype=float), coefficients)
    v1 = volume_gri.gri_volume(code, d2, h2, coefficients)
    return pd.Series((v1 - v0) / horizon, index=trees.index)


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    """Area-weighted mean over the entries that have a value (the others are left out)."""
    ok = values.notna() & weights.notna()
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() and weights[ok].sum() > 0 else np.nan


def evaluate_age(structure: pd.DataFrame, trees: pd.DataFrame, age_of: Callable[[int], float], *,
                 tables: Mapping[int, YieldTable], coefficients: pd.DataFrame, growth: pd.DataFrame, horizon: float,
                 factors: pd.Series | None = None, per_tree_cache: dict | None = None) -> pd.DataFrame:
    """Methods 1 and 2 for every Parzelle at the age `age_of(parzelle)`, plus the area-weighted row 'stand'.

    The columns are those of increment_by_age.csv except the `age` of the stand row, which is the common age when all
    Parzellen share one age and NaN otherwise. In the stand row the means are area-weighted over the Parzellen that
    have a value; d_q, main species and basal area come from all trees together.
    """
    cache = per_tree_cache if per_tree_cache is not None else {}
    parz = [p for p in structure.index if p != "stand"]
    rows = []
    for p in parz:
        s = structure.loc[p]
        age = float(age_of(p))
        if age not in cache:
            cache[age] = tree_growth_increment(trees, age, horizon, coefficients, growth)
        g = trees["parzelle"].astype(int) == p
        inc = cache[age][g]
        area = float(s["area_ha"])
        factor = volume_weighted_factor(trees.loc[g, "gri_code"], trees.loc[g, "volume_gri_field_m3"], factors)
        pai_tree = float(inc.sum() / area) if inc.notna().any() and area > 0 else np.nan
        r = {"parzelle": p, "age": age, "main_species": s["main_species"], "gri_code": s["gri_code"],
             "d_q_cm": s["d_q_cm"], "h_q_m": s["h_q_m"], "ba_m2_ha": s["ba_m2_ha"], **yield_table_values(s, age, tables),
             "pai_tree_growth_m3_ha_yr": pai_tree, "n_trees_tree_growth": int(inc.notna().sum()), "efm_factor": factor}
        r["pai_yield_table_efm_m3_ha_yr"] = r["pai_yield_table_m3_ha_yr"] * factor
        r["pai_tree_growth_efm_m3_ha_yr"] = pai_tree * factor
        rows.append(r)
    df = pd.DataFrame(rows)
    ages = df["age"].unique()
    w = structure.loc[parz, "area_ha"].reset_index(drop=True)
    st = structure.loc["stand"]
    stand = {"parzelle": "stand", "age": float(ages[0]) if len(ages) == 1 else np.nan,
             "main_species": st["main_species"], "gri_code": st["gri_code"], "d_q_cm": st["d_q_cm"], "ba_m2_ha": st["ba_m2_ha"],
             "yield_table": "; ".join(sorted(set(df["yield_table"]) - {""})),
             "si_extrapolated": bool(df["si_extrapolated"].any()), "age_outside_table": bool(df["age_outside_table"].any()),
             "n_trees_tree_growth": int(df["n_trees_tree_growth"].sum())}
    for c in ("h_q_m", "site_index", "pai_table_m3_ha_yr", "mai_table_m3_ha_yr", "stocking_level", "pai_yield_table_m3_ha_yr",
              "pai_tree_growth_m3_ha_yr", "efm_factor", "pai_yield_table_efm_m3_ha_yr", "pai_tree_growth_efm_m3_ha_yr"):
        stand[c] = _wmean(df[c], w)
    out = pd.concat([df, pd.DataFrame([stand])], ignore_index=True)
    return out[BY_AGE_COLUMNS]


def by_age_table(structure: pd.DataFrame, trees: pd.DataFrame, ages, **kw) -> pd.DataFrame:
    """Rows per Parzelle and 'stand' for every age of `ages` (all Parzellen at the same age)."""
    cache: dict = {}
    parts = [evaluate_age(structure, trees, lambda p, a=a: a, per_tree_cache=cache, **kw) for a in ages]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Method 3: measured increment from two surveys (periodic increment)
# ---------------------------------------------------------------------------

def clean_height_2020(height_sheet: pd.Series, dm_above: float, bounds_m: Sequence[float]) -> pd.Series:
    """Height of the 2020 survey in m: values above `dm_above` were typed in dm (as for the current standing heights);
    heights outside `bounds_m` after that are recording errors and become NaN."""
    h = volume_gri.standing_height_m(height_sheet, dm_above)
    return h.where(h.between(*bounds_m))


def curves_2020(all_2020: pd.DataFrame, *, dbh_plausible_mm: Sequence[float], dm_above: float, bounds_m: Sequence[float],
                min_trees: int, breast: float) -> tuple[pd.DataFrame, int]:
    """Height-diameter curves of the 2020 survey (2020 diameter x 2020 height) and the number of dropped heights.

    All trees of the survey enter, also those removed since. Returns (curves as in core.volume.fit_curves, n heights
    that were measured but are outside `bounds_m`).
    """
    d = pd.to_numeric(all_2020["dbh_2020_mm"], errors="coerce")
    d = d.where(d.between(*dbh_plausible_mm))
    raw = volume_gri.standing_height_m(all_2020["height_2020_m"], dm_above)
    h = clean_height_2020(all_2020["height_2020_m"], dm_above, bounds_m)
    n_dropped = int((raw.notna() & h.isna()).sum())
    return fit_curves(all_2020["species"], d.where(h.notna()), h, min_trees=min_trees, breast=breast), n_dropped


def tree_table(all_2020: pd.DataFrame, now: pd.DataFrame, curve_2020: pd.DataFrame, curve_now: pd.DataFrame,
               coefficients: pd.DataFrame, species_codes: Mapping, *, dbh_plausible_mm: Sequence[float],
               dbh_change_plausible_mm: Sequence[float], survey_years: Mapping[int, int], breast: float) -> pd.DataFrame:
    """One row per tree of the 2020 survey: status, diameters, curve heights and GRI volumes at both dates.

    all_2020: tree_id, parzelle, species, dbh_2020_mm, removed_earlier (core.io_field.load_survey_2020).
    now: tree_id, dbh_field_mm (the current inventory; trees felled in the latest harvest were measured before felling
    and are part of it). Status, in this order:
      no_2020_dbh              no diameter in 2020
      excluded_implausible_dbh a diameter outside `dbh_plausible_mm` (2020, or current for a standing tree)
      removed_earlier          flagged as removed in an earlier thinning
      missing                  2020 diameter, no current diameter, no flag
      excluded_dbh_change      the diameter change lies outside `dbh_change_plausible_mm` (recording error)
      no_volume                no GRI volume at one of the dates (species without code)
      survivor                 everything else
    The height at each date is that of the height-diameter curve of that date. v_static_2020_m3 is the volume at the
    2020 diameter with the current curve, for the comparison variant with one static curve.
    """
    t = all_2020.reset_index(drop=True).copy()
    lo, hi = dbh_plausible_mm
    d_old = pd.to_numeric(t["dbh_2020_mm"], errors="coerce")
    d_now = pd.to_numeric(t["tree_id"].map(now.set_index("tree_id")["dbh_field_mm"]), errors="coerce")
    change = d_now - d_old
    status = np.select(
        [d_old.isna(), ~d_old.between(lo, hi), t["removed_earlier"].astype(bool), d_now.isna(), ~d_now.between(lo, hi),
         ~change.between(*dbh_change_plausible_mm)],
        ["no_2020_dbh", "excluded_implausible_dbh", "removed_earlier", "missing", "excluded_implausible_dbh",
         "excluded_dbh_change"], default="survivor")
    t["status"] = status
    code = volume_gri.species_to_code(t["species"], species_codes, coefficients).to_numpy(dtype=float)

    d_old_ok = d_old.where(d_old.between(lo, hi))
    has_now = t["status"].isin(["survivor", "excluded_dbh_change"])
    d_now_ok = d_now.where(has_now)
    h_old = model_heights(t["species"], d_old_ok, curve_2020, breast)
    h_now = model_heights(t["species"], d_now_ok, curve_now, breast)
    h_static = model_heights(t["species"], d_old_ok, curve_now, breast)
    t["v_2020_m3"] = volume_gri.gri_volume(code, d_old_ok, h_old, coefficients)
    t["v_now_m3"] = volume_gri.gri_volume(code, d_now_ok, h_now, coefficients)
    t["v_static_2020_m3"] = volume_gri.gri_volume(code, d_old_ok, h_static, coefficients)
    no_volume = (t["status"] == "survivor") & (t["v_2020_m3"].isna() | t["v_now_m3"].isna())
    t.loc[no_volume, "status"] = "no_volume"

    t["dbh_mm"] = d_now
    t["dbh_change_mm"] = change.where(d_old_ok.notna() & ~t["removed_earlier"].astype(bool))
    t["height_2020_curve_m"] = h_old.where(d_old_ok.notna())
    t["height_now_curve_m"] = h_now.where(has_now)
    t["survey_year"] = t["parzelle"].astype(int).map(survey_years)
    t["gri_code"] = code
    return t[[*PER_TREE_COLUMNS, "v_static_2020_m3", "gri_code"]]


def _seasons_of(years: pd.Series, seasons: Mapping[int, float]) -> pd.Series:
    """Growing seasons of the survey year of each tree (NaN for a year without an entry)."""
    return pd.to_numeric(years.map({int(k): float(v) for k, v in seasons.items()}), errors="coerce")


def _row(g: pd.DataFrame, area_ha: float, seasons: float, share: float, factors: pd.Series | None) -> dict:
    """Ledger of one Parzelle (columns of increment_measured.csv); g is its rows of `tree_table`."""
    surv = g[g["status"] == "survivor"]
    gone = g[g["status"].isin(REMOVED_STATUSES)]
    v_t1_surv, v_t2 = surv["v_2020_m3"].sum(), surv["v_now_m3"].sum()
    growth = float((surv["v_now_m3"] - surv["v_2020_m3"]).sum())
    rel = growth / v_t1_surv if v_t1_surv > 0 else np.nan
    removed_2020 = float(gone["v_2020_m3"].sum())
    removed = removed_2020 * (1.0 + share * rel) if removed_2020 > 0 and np.isfinite(rel) else removed_2020
    v_t1 = v_t1_surv + removed_2020
    increment = v_t2 + removed - v_t1
    static_growth = float((surv["v_now_m3"] - surv["v_static_2020_m3"]).sum())
    per_ha = lambda v: v / seasons / area_ha if seasons > 0 and area_ha > 0 else np.nan
    return {"survey_year": g["survey_year"].iloc[0], "growing_seasons": seasons, "area_ha": area_ha,
            "n_survivors": len(surv), "n_removed_earlier": int((g["status"] == "removed_earlier").sum()),
            "n_missing": int((g["status"] == "missing").sum()),
            "n_excluded": int(g["status"].isin(EXCLUDED_STATUSES).sum()),
            "v_alive_t1_m3": v_t1, "v_alive_t2_m3": v_t2, "v_removed_2020_m3": removed_2020, "v_removed_m3": removed,
            "survivor_growth_m3": growth, "survivor_rel_growth": rel, "increment_m3": increment,
            "pai_measured_m3_ha_yr": per_ha(increment), "pai_static_curve_m3_ha_yr": per_ha(static_growth),
            "dbh_increment_mm_yr": float(((surv["dbh_mm"] - surv["dbh_2020_mm"]) / seasons).mean()) if len(surv) else np.nan,
            "efm_factor": volume_weighted_factor(surv["gri_code"], surv["v_now_m3"], factors)}


def ledger_table(per_tree: pd.DataFrame, areas_ha: pd.Series, seasons: Mapping[int, float], share: float,
                 factors: pd.Series | None = None) -> pd.DataFrame:
    """Periodic increment per Parzelle and for the 'stand' (columns of increment_measured.csv).

    increment = V_alive(t2) + V_removed - V_alive(t1) over the trees of `tree_table`. Survivors are alive at both
    dates; removed_earlier and missing trees are alive at t1 with their 2020 volume and removed in between, with
    `share` x relative volume growth of the survivors of their Parzelle added to their volume. With share 0 the
    increment is the volume growth of the survivors. Everything is in m3; pai is per ha and growing season.

    The columns allow pai to be recomputed for other seasons and shares without this code:
    pai = (survivor_growth_m3 + share x v_removed_2020_m3 x survivor_rel_growth) / seasons / area_ha.
    The stand row sums the Parzellen; its survivor_rel_growth is weighted by v_removed_2020_m3 so that the formula
    holds there too, and its growing_seasons is empty where the Parzellen differ (pai is then the sum of the
    Parzelle values per season over the total area). per_tree is the table of `tree_table` (with gri_code).
    """
    sec = _seasons_of(per_tree["survey_year"], seasons)
    rows = {}
    for p, g in per_tree.groupby("parzelle"):
        area = float(areas_ha.get(int(p), np.nan))
        rows[int(p)] = _row(g, area, float(sec[g.index].iloc[0]), share, factors)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "parzelle"
    ok = df["area_ha"] > 0
    d = df[ok]
    total_area = float(d["area_ha"].sum())
    v_rem = d["v_removed_2020_m3"]
    v_surv_t1 = (d["v_alive_t1_m3"] - v_rem).sum()
    rel_w = ((d["survivor_rel_growth"].fillna(0.0) * v_rem).sum() / v_rem.sum() if v_rem.sum() > 0
             else d["survivor_growth_m3"].sum() / v_surv_t1 if v_surv_t1 > 0 else np.nan)
    surv = per_tree["status"] == "survivor"
    stand = {c: d[c].sum() for c in ("n_survivors", "n_removed_earlier", "n_missing", "n_excluded", "v_alive_t1_m3",
                                     "v_alive_t2_m3", "v_removed_2020_m3", "v_removed_m3", "survivor_growth_m3", "increment_m3")}
    s = d["growing_seasons"].dropna().unique()
    stand.update(survey_year=np.nan, growing_seasons=float(s[0]) if len(s) == 1 else np.nan, area_ha=total_area,
                 survivor_rel_growth=rel_w)
    for col in ("pai_measured_m3_ha_yr", "pai_static_curve_m3_ha_yr"):
        stand[col] = float((d[col] * d["area_ha"]).sum() / total_area) if total_area > 0 else np.nan
    keep = surv & per_tree["parzelle"].astype(int).isin(d.index)
    stand["dbh_increment_mm_yr"] = float(((per_tree.loc[keep, "dbh_mm"] - per_tree.loc[keep, "dbh_2020_mm"]) / sec[keep]).mean())
    stand["efm_factor"] = _wmean(d["efm_factor"], d["area_ha"])
    df = pd.concat([df.reset_index(), pd.DataFrame([{"parzelle": "stand", **stand}])], ignore_index=True)
    df["pai_measured_efm_m3_ha_yr"] = df["pai_measured_m3_ha_yr"] * df["efm_factor"]
    for c in ("n_survivors", "n_removed_earlier", "n_missing", "n_excluded"):
        df[c] = df[c].astype(int)
    return df[LEDGER_COLUMNS]


def bootstrap_pai(per_tree: pd.DataFrame, areas_ha: pd.Series, seasons: Mapping[int, float], share: float, n_boot: int,
                  seed: int) -> tuple[float, float]:
    """95 % interval of the stand increment per ha and year, resampling the survivors with replacement within each Parzelle.

    The removed trees and the area are fixed; the survivors' volume growth, and with it the growth assumed for the
    removed trees, varies between resamples.
    """
    rng = np.random.default_rng(seed)
    sec = _seasons_of(per_tree["survey_year"], seasons)
    total = np.zeros(n_boot)
    area_total, n_used = 0.0, 0
    for p, g in per_tree.groupby("parzelle"):
        area = float(areas_ha.get(int(p), np.nan))
        surv = g[g["status"] == "survivor"]
        if not area > 0:
            continue
        area_total += area
        n_used += len(surv)
        if len(surv) == 0:
            continue
        v_old, v_now = surv["v_2020_m3"].to_numpy(float), surv["v_now_m3"].to_numpy(float)
        removed = float(g.loc[g["status"].isin(REMOVED_STATUSES), "v_2020_m3"].sum())
        idx = rng.integers(0, len(surv), size=(n_boot, len(surv)))
        growth = (v_now[idx] - v_old[idx]).sum(axis=1)
        rel = growth / v_old[idx].sum(axis=1)
        total += (growth + share * removed * rel) / float(sec[g.index].iloc[0])
    if n_used < 2:
        return np.nan, np.nan
    boot = total / area_total
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Statistics entries
# ---------------------------------------------------------------------------

def _seasons_text(ledger: pd.DataFrame) -> str:
    p = ledger[ledger["parzelle"].astype(str) != "stand"].dropna(subset=["survey_year", "growing_seasons"])
    pairs = sorted({(int(y), float(s)) for y, s in zip(p["survey_year"], p["growing_seasons"])})
    return ", ".join(f"{s:g} growing seasons between 2020 and {y}" for y, s in pairs)


def measured_entries(per_tree: pd.DataFrame, ledger: pd.DataFrame, curves: pd.DataFrame, *, areas_ha: pd.Series,
                     seasons: Mapping[int, float], share: float, n_dropped_heights: int, n_boot: int, seed: int) -> list[dict]:
    """Statistics entries of method 3 (increment from two surveys). Pure."""
    s = ledger[ledger["parzelle"].astype(str) == "stand"]
    if len(s) == 0:
        return []
    s = s.iloc[0]
    n = int(s["n_survivors"])
    lo, hi = bootstrap_pai(per_tree, areas_ha, seasons, share, n_boot, seed)
    area = float(s["area_ha"])
    basis = (f"the periodic increment of the two surveys, V_alive(t2) + V_removed - V_alive(t1), with GRI volumes over bark "
             f"(Vfm m. R.) and one height-diameter curve per survey ({_seasons_text(ledger)}; trees removed earlier are assumed "
             f"to have grown {share * 100:g} % of the period; neither a removal year nor a diameter at removal is recorded, "
             f"and no ingrowth is recorded)")
    frame = (f"{n} surviving trees, {int(s['n_removed_earlier'])} trees removed earlier and {int(s['n_missing'])} trees "
             f"without a current diameter or flag, {int(s['n_excluded'])} trees excluded, {area:.2f} ha")
    out = [_entry("increment.measured.stand.pai", s["pai_measured_m3_ha_yr"], "m3/ha/yr", n, ci=(lo, hi),
                  statement=(f"With {basis}, the volume increment of the stand is {s['pai_measured_m3_ha_yr']:.2f} m3/ha/yr "
                             f"standing volume over bark (95 % bootstrap interval {lo:.2f} to {hi:.2f}, surviving trees "
                             f"resampled within each Parzelle; {frame}).")),
           _entry("increment.measured.stand.pai_static_curve", s["pai_static_curve_m3_ha_yr"], "m3/ha/yr", n,
                  statement=(f"With the volume growth of the {n} surviving trees alone, at one static height-diameter curve of "
                             f"the current survey for both dates (so that height growth is captured only along that curve), "
                             f"the volume increment of the stand is {s['pai_static_curve_m3_ha_yr']:.2f} m3/ha/yr standing "
                             f"volume over bark; {_seasons_text(ledger)}; removed and missing trees do not enter this variant.")),
           _entry("increment.measured.stand.dbh", s["dbh_increment_mm_yr"], "mm/yr", n,
                  statement=(f"For the {n} surviving trees the mean diameter increment is {s['dbh_increment_mm_yr']:.2f} mm "
                             f"per year ({_seasons_text(ledger)}).")),
           _entry("increment.measured.trees", n, "trees", n,
                  statement=(f"{n} trees have a plausible diameter in 2020 and at the current survey and a plausible diameter "
                             f"change and enter the survivors' volume growth of the periodic increment.")),
           ]
    for key, label, col in (("v_alive_t1", "alive at the 2020 survey, V_alive(t1)", "v_alive_t1_m3"),
                            ("v_alive_t2", "alive at the current survey, V_alive(t2) (the survivors)", "v_alive_t2_m3"),
                            ("v_removed", "removed between the surveys, V_removed (their 2020 volume plus the assumed growth)",
                             "v_removed_m3")):
        v = s[col] / area if area > 0 else np.nan
        out.append(_entry(f"increment.measured.stand.{key}_per_ha", v, "m3/ha", n,
                          statement=(f"With {basis}, the volume {label}, is {v:.1f} m3/ha of the {area:.2f} ha surveyed "
                                     f"plot area.")))
    if np.isfinite(s["pai_measured_efm_m3_ha_yr"]):
        out.append(_entry("increment.measured.stand.pai_efm", s["pai_measured_efm_m3_ha_yr"], "m3/ha/yr", n,
                          statement=(f"With {basis}, the increment of the stand is {s['pai_measured_efm_m3_ha_yr']:.2f} "
                                     f"m3/ha/yr as harvest volume under bark (Efm o. R.), from the standing volume with the "
                                     f"volume-weighted reduction factor of ForestElementsR (v_red_harvest_ubark, "
                                     f"{s['efm_factor']:.3f}).")))

    rem = per_tree[per_tree["status"] == "removed_earlier"]
    v_rem = float(rem["v_2020_m3"].sum())
    share_vol = v_rem / s["v_alive_t1_m3"] if s["v_alive_t1_m3"] > 0 else np.nan
    out.append(_entry("increment.measured.removed_earlier", len(rem), "trees", len(rem),
                      statement=(f"{len(rem)} trees of the 2020 survey are flagged as removed in an earlier thinning; they hold "
                                 f"{share_vol * 100:.1f} % of the stand volume of 2020 ({v_rem:.1f} m3 at the 2020 diameter and "
                                 f"the 2020 curve height). They count as alive at the 2020 survey and as removed in between; "
                                 f"{int(s['n_missing'])} further trees without a current diameter and flag are treated the same "
                                 f"way. Trees removed earlier are assumed to have grown {share * 100:g} % of the period; "
                                 f"neither a removal year nor a diameter at removal is recorded.")))
    exc = per_tree[per_tree["status"].isin(EXCLUDED_STATUSES)]
    ids = _list_text(exc["tree_id"])
    out.append(_entry("increment.measured.excluded", len(exc), "trees", len(exc),
                      statement=(f"{len(exc)} trees are left out of the periodic increment at both dates: a diameter outside the "
                                 f"plausible range or a diameter change outside the plausible range count as recording "
                                 f"errors, and trees without a diameter in 2020 or without a volume have no value"
                                 + (f" (tree ids {ids})." if len(exc) else "."))))
    for _, c in curves.iterrows():
        if not np.isfinite(c["loo_rmse_m"]):
            continue
        scope = "pooled" if c["scope"] == "pooled" else _slug(c["species"])
        label = "all species pooled" if c["scope"] == "pooled" else f"the species {c['species']}"
        extra = (f" {n_dropped_heights} measured heights outside the plausible range were left out." if c["scope"] == "pooled" else "")
        out.append(_entry(f"increment.measured.height_curve_2020.{scope}.loo_rmse", c["loo_rmse_m"], "m", c["n"],
                          statement=(f"The height-diameter curve of the 2020 survey for {label}, fitted on the 2020 diameter and "
                                     f"the 2020 height, misses a left-out tree's measured height by {c['loo_rmse_m']:.2f} m "
                                     f"(leave-one-out root-mean-square error, {int(c['n'])} trees).{extra}")))
    return out


def age_entries(stand_row: pd.Series, structure: pd.DataFrame, *, age: float, year: int, horizon: float,
                overrides: Mapping[int, float]) -> list[dict]:
    """Statistics entries of methods 1 and 2 at the given stand age (row 'stand' of `evaluate_age`). Pure."""
    n_valid = int(structure.loc["stand", "n_valid"])
    n_t = int(stand_row["n_trees_tree_growth"])
    where = f"a stand age of {age:g} years in {year}" + (f" (Parzellen {', '.join(str(k) for k in sorted(overrides))} with their own age)" if overrides else "")
    yt = f"With the yield table of ForestElementsR at {where}"
    tg = f"With the tree growth functions of the third national forest inventory at {where}"
    flag = " The site index lies outside the yield table, so the table values are extrapolated." if stand_row["si_extrapolated"] else ""
    out = []

    def add(id_, value, unit, n, text, ci=(None, None)):
        if np.isfinite(value):
            out.append(_entry(id_, value, unit, n, text, ci=ci))

    table = stand_row["yield_table"] or "the yield tables of the main species"
    base = f"{n_valid} trees, main species {stand_row['main_species']}, {table}"
    add("increment.yield_table.stand.pai", stand_row["pai_yield_table_m3_ha_yr"], "m3/ha/yr", n_valid,
        f"{yt}, the volume increment of the stand is {stand_row['pai_yield_table_m3_ha_yr']:.2f} m3/ha/yr standing volume over bark "
        f"(Vfm m. R.): the table increment multiplied by the stocking level where that is below 1 ({base}).{flag}")
    add("increment.yield_table.stand.pai_table", stand_row["pai_table_m3_ha_yr"], "m3/ha/yr", n_valid,
        f"{yt}, the table increment at the site index of the stand is {stand_row['pai_table_m3_ha_yr']:.2f} m3/ha/yr, before the "
        f"stocking level is applied ({base}).{flag}")
    add("increment.yield_table.stand.mai", stand_row["mai_table_m3_ha_yr"], "m3/ha/yr", n_valid,
        f"{yt}, the mean annual increment of the table at the site index of the stand is {stand_row['mai_table_m3_ha_yr']:.2f} m3/ha/yr ({base}).{flag}")
    add("increment.yield_table.stand.site_index", stand_row["site_index"], "class", n_valid,
        f"{yt}, the site index of the stand is {stand_row['site_index']:.2f} (area-weighted over the Parzellen; {base}).{flag}")
    add("increment.yield_table.stand.stocking_level", stand_row["stocking_level"], "share", n_valid,
        f"{yt}, the stocking level (basal area of the stand over the table basal area) is {stand_row['stocking_level']:.2f} ({base}).")
    add("increment.yield_table.stand.pai_efm", stand_row["pai_yield_table_efm_m3_ha_yr"], "m3/ha/yr", n_valid,
        f"{yt}, the increment of the stand is {stand_row['pai_yield_table_efm_m3_ha_yr']:.2f} m3/ha/yr as harvest volume under bark "
        f"(Efm o. R.), from the standing volume with the volume-weighted reduction factor of ForestElementsR (v_red_harvest_ubark).")
    add("increment.tree_growth.stand.pai", stand_row["pai_tree_growth_m3_ha_yr"], "m3/ha/yr", n_t,
        f"{tg}, the volume increment of the stand over {horizon:g} years is {stand_row['pai_tree_growth_m3_ha_yr']:.2f} m3/ha/yr standing "
        f"volume over bark (Vfm m. R.), from {n_t} trees that all survive the period.")
    add("increment.tree_growth.stand.pai_efm", stand_row["pai_tree_growth_efm_m3_ha_yr"], "m3/ha/yr", n_t,
        f"{tg}, the increment of the stand is {stand_row['pai_tree_growth_efm_m3_ha_yr']:.2f} m3/ha/yr as harvest volume under bark "
        f"(Efm o. R.), from the standing volume with the volume-weighted reduction factor of ForestElementsR (v_red_harvest_ubark).")
    return out


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _settings(cfg) -> dict:
    return dict(cfg.raw.get("increment") or {})


def _params_dir(cfg) -> Path:
    path = Path(str(_settings(cfg).get("params_dir", "config")))
    return path if path.is_absolute() else cfg.project_root / path


def load_parameters(params_dir) -> tuple[dict | None, str | None]:
    """The exported package data: ({tables, growth, factors}, None), or (None, one-line problem) when files are missing.

    The reduction factors are optional (without them the Efm columns stay empty).
    """
    d = Path(params_dir)
    need = [d / "fer_yield_tables.json", d / "fer_gnfi3_parameters.csv"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        return None, (f"increment skipped: the exported file(s) {', '.join(missing)} are missing "
                      "(`Rscript scripts/export_gri_coefficients.R` creates them).")
    red = d / "fer_harvest_reduction.csv"
    return {"tables": load_yield_tables(need[0]), "growth": load_growth_parameters(need[1]),
            "factors": volume_gri.load_harvest_factors(red) if red.is_file() else None}, None


def _prepare(cfg, tables_dir) -> tuple[dict | None, str | None]:
    """Everything the three methods need, or (None, one-line problem)."""
    params, problem = load_parameters(_params_dir(cfg))
    if problem:
        return None, problem
    coefficients, species_codes, problem = volume_gri._setup(cfg)
    if problem:
        return None, "increment skipped: " + problem.replace("volume_gri skipped: ", "")
    if not has_table(cfg, "wood_volume_gri_per_tree.csv", tables_dir):
        return None, "increment skipped: wood_volume_gri_per_tree.csv is missing (run the volume_gri stage first)."
    areas = volume_gri._areas(cfg)
    if areas is None:
        return None, "increment skipped: the plot areas cannot be computed from the corner file."
    trees = read_table(cfg, "wood_volume_gri_per_tree.csv", tables_dir)
    trees["parzelle"] = trees["parzelle"].astype(int)
    breast = float(cfg.dbh.breast_height_m)
    min_trees = int(dict(cfg.raw.get("volume_gri") or {}).get("min_trees_per_species", 20))
    curves = height_curves(trees, min_trees=min_trees, breast=breast)
    detection = read_table(cfg, "field_trees_detection.csv", tables_dir, usecols=["tree_id", "dbh_year"])
    years = trees[["tree_id", "parzelle"]].merge(detection, on="tree_id", how="left").groupby("parzelle")["dbh_year"].first()
    survey_2020 = read_table(cfg, "field_trees_2020.csv", tables_dir) if has_table(cfg, "field_trees_2020.csv", tables_dir) else None
    return {**params, "coefficients": coefficients, "species_codes": species_codes, "areas": areas, "trees": trees,
            "curves": curves, "breast": breast, "min_trees": min_trees, "survey_2020": survey_2020,
            "survey_years": {int(k): int(v) for k, v in years.dropna().items()},
            "structure": stand_structure(trees, areas, curves, breast)}, None


def _ages(cfg) -> tuple[list[int], dict[int, float]]:
    s = _settings(cfg)
    lo, hi = s.get("age_grid", [15, 160])
    overrides = {int(k): float(v) for k, v in (s.get("age_per_parzelle") or {}).items()}
    return list(range(int(lo), int(hi) + 1)), overrides


def _growing_seasons(cfg, survey_years: Mapping[int, int], base_year: int = 2020) -> dict[int, float]:
    """Growing seasons per survey year: increment.growing_seasons, else the calendar years since the 2020 survey."""
    given = {int(k): float(v) for k, v in (_settings(cfg).get("growing_seasons") or {}).items()}
    return {int(y): given.get(int(y), float(y - base_year)) for y in set(survey_years.values())}


def _measured(cfg, ctx):
    """(per-tree table, ledger, curves of 2020, number of dropped 2020 heights, seasons, share) of the periodic increment."""
    s = _settings(cfg)
    plaus = list(cfg.cleaning.dbh_plausible_mm)
    dm_above = float(cfg.cleaning.standing_height_dm_above)
    survey = ctx["survey_2020"]
    curve_2020, n_dropped = curves_2020(survey, dbh_plausible_mm=plaus, dm_above=dm_above, bounds_m=HEIGHT_2020_BOUNDS_M,
                                        min_trees=ctx["min_trees"], breast=ctx["breast"])
    now = ctx["trees"]
    d_now = pd.to_numeric(now["dbh_field_mm"], errors="coerce")
    curve_now = fit_curves(now["species"], d_now.where(d_now.between(*plaus)), now["height_field_m"],
                           min_trees=ctx["min_trees"], breast=ctx["breast"])
    per_tree = tree_table(survey, now, curve_2020, curve_now, ctx["coefficients"], ctx["species_codes"],
                          dbh_plausible_mm=plaus, dbh_change_plausible_mm=list(s.get("dbh_change_plausible_mm", [-10, 150])),
                          survey_years=ctx["survey_years"], breast=ctx["breast"])
    seasons = _growing_seasons(cfg, ctx["survey_years"])
    share = float(s.get("removed_growth_share", 0.0))
    ledger = ledger_table(per_tree, ctx["areas"], seasons, share, ctx["factors"])
    return per_tree, ledger, curve_2020, n_dropped, seasons, share


def _kwargs(cfg, ctx) -> dict:
    return dict(tables=ctx["tables"], coefficients=ctx["coefficients"], growth=ctx["growth"],
                horizon=float(_settings(cfg).get("horizon_years", 5)), factors=ctx["factors"])


def run_increment(cfg, tables_dir=None) -> pd.DataFrame | None:
    """Write increment_by_age.csv, increment_measured.csv, increment_per_tree.csv and height_diameter_curves_2020.csv.

    Returns the by-age table. Without the exported files or the GRI tables it prints one line and returns None; without
    field_trees_2020.csv (written by the inventory stage) the measured increment is skipped.
    """
    ctx, problem = _prepare(cfg, tables_dir)
    if problem:
        print(problem)
        return None
    ages, _ = _ages(cfg)
    by_age = by_age_table(ctx["structure"], ctx["trees"], ages, **_kwargs(cfg, ctx))
    out = Path(cfg.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_age.to_csv(out / "increment_by_age.csv", index=False)
    if ctx["survey_2020"] is None:
        print("increment: the measured increment is skipped, field_trees_2020.csv is missing (run the inventory stage first).")
        return by_age
    per_tree, ledger, curves, *_ = _measured(cfg, ctx)
    ledger.to_csv(out / "increment_measured.csv", index=False)
    per_tree[PER_TREE_COLUMNS].to_csv(out / "increment_per_tree.csv", index=False)
    curves.to_csv(out / "height_diameter_curves_2020.csv", index=False)
    return by_age


def compute_entries(cfg, tables_dir=None) -> list[dict]:
    """Statistics entries of the increment; empty without the exported files or the GRI tables.

    The measured increment is always reported; the yield-table and tree-growth values need increment.stand_age_years.
    """
    ctx, problem = _prepare(cfg, tables_dir)
    if problem:
        return []
    s = _settings(cfg)
    v = dict(cfg.raw.get("volume_gri") or {})
    out = []
    if ctx["survey_2020"] is not None:
        per_tree, ledger, curves, n_dropped, seasons, share = _measured(cfg, ctx)
        out = measured_entries(per_tree, ledger, curves, areas_ha=ctx["areas"], seasons=seasons, share=share,
                               n_dropped_heights=n_dropped, n_boot=int(v.get("bootstrap_resamples", 2000)),
                               seed=int(v.get("bootstrap_seed", 0)))
    age = s.get("stand_age_years")
    if age is not None:
        _, overrides = _ages(cfg)
        table = evaluate_age(ctx["structure"], ctx["trees"], lambda p: overrides.get(p, float(age)), **_kwargs(cfg, ctx))
        out += age_entries(table[table["parzelle"].astype(str) == "stand"].iloc[0], ctx["structure"], age=float(age),
                           year=int(s.get("age_reference_year", 2026)), horizon=float(s.get("horizon_years", 5)),
                           overrides=overrides)
    return out
