"""Identity of field-to-LiDAR matches: is the matched instance the same tree?

Evidence is geometric only (base offset, uniqueness, neighbourhood consistency,
stem evidence at the field position). Tree height and DBH never enter the score;
they are written as flags only. See docs/METHODS_identity.md.

All functions are pure (data in, DataFrame out). Only `run` writes files.
Thresholds are keyword arguments; defaults here are generic, site values come
from `cfg.identity` / `cfg.matching`.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ("confirmed", "probable", "ambiguous", "rejected", "unmatched")
FEATURES = ["offset_corrected_m", "nbr_ratio"]

OUT_COLUMNS = ["tree_id", "instance", "offset_m", "offset_corrected_m",
               "second_best_margin_m", "mutual_best", "neighbourhood_residual_m",
               "stem_points_at_bh", "identity_score", "identity_class", "reasons",
               "dbh_flag", "height_flag"]
OFFSET_COLUMNS = ["parzelle", "dx_m", "dy_m", "rotation_deg", "n_pairs",
                  "residual_rms_m", "best_orientation"]


# ---------------------------------------------------------------------------
# Robust rigid fit per plot
# ---------------------------------------------------------------------------

def fit_rigid_robust(src, dst, spread_floor_m=0.05, tukey_c=4.685, n_iter=20):
    """Fit dst ~ R(src - c_src) + c_dst + nothing else (rotation + translation).

    Iteratively reweighted (Tukey biweight on the residual length, scale = MAD),
    so a few wrong pairs do not pull the fit. Returns dict with the rotation
    angle (rad), the two centres, the residual vectors and the weights.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    w = np.ones(len(src))
    theta, c_src, c_dst = 0.0, src.mean(axis=0), dst.mean(axis=0)
    for _ in range(n_iter):
        sw = w.sum()
        if sw <= 0:
            break
        c_src = (w[:, None] * src).sum(axis=0) / sw
        c_dst = (w[:, None] * dst).sum(axis=0) / sw
        a, b = src - c_src, dst - c_dst
        theta = np.arctan2((w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum(),
                           (w * (a * b).sum(axis=1)).sum())
        res = dst - _apply(src, theta, c_src, c_dst)
        r = np.hypot(res[:, 0], res[:, 1])
        s = max(1.4826 * np.median(r), spread_floor_m)
        u = r / (tukey_c * s)
        w_new = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        if np.allclose(w_new, w, atol=1e-6):
            w = w_new
            break
        w = w_new
    res = dst - _apply(src, theta, c_src, c_dst)
    return {"theta": theta, "c_src": c_src, "c_dst": c_dst, "residual": res, "weights": w}


def _apply(xy, theta, c_src, c_dst):
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (np.asarray(xy, float) - c_src) @ rot.T + c_dst


def plot_transforms(pairs, min_pairs=5, spread_floor_m=0.05):
    """Fit one robust rigid transform per plot.

    pairs: DataFrame with parzelle, utm_x, utm_y (field stem), bx, by (matched base).
    Returns (offsets DataFrame, dict parzelle -> (theta, c_src, c_dst) or None).
    Plots with fewer than `min_pairs` pairs get no transform (raw offsets are used).
    """
    rows, tf = [], {}
    for p, g in pairs.groupby("parzelle", sort=True):
        n = len(g)
        if n < min_pairs:
            tf[p] = None
            rows.append({"parzelle": p, "dx_m": np.nan, "dy_m": np.nan, "rotation_deg": np.nan,
                         "n_pairs": n, "residual_rms_m": np.nan})
            continue
        fit = fit_rigid_robust(g[["utm_x", "utm_y"]], g[["bx", "by"]], spread_floor_m)
        tf[p] = (fit["theta"], fit["c_src"], fit["c_dst"])
        d = fit["c_dst"] - fit["c_src"]
        rows.append({"parzelle": p, "dx_m": d[0], "dy_m": d[1],
                     "rotation_deg": np.degrees(fit["theta"]), "n_pairs": n,
                     "residual_rms_m": float(np.sqrt((fit["residual"] ** 2).sum(axis=1).mean()))})
    return pd.DataFrame(rows, columns=OFFSET_COLUMNS[:-1]), tf


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def compute_evidence(stems, instances, *, min_pairs=5, neighbours_k=5, spread_floor_m=0.05,
                     reference=None):
    """Geometric evidence for each assigned match.

    stems: tree_id, parzelle, utm_x, utm_y, instance (NaN = unmatched).
    instances: instance, x_base, y_base (all candidates).
    reference: optional (transforms, spread) from the real matches. The chance control is
    judged against the real plot fit and the real plot spread, not against a fit to
    its own false pairs (which would absorb part of the error).
    Returns (evidence DataFrame indexed like stems, plot offsets DataFrame, reference).
    """
    st = stems.reset_index(drop=True).copy()
    inst = instances.drop_duplicates("instance").set_index("instance")[["x_base", "y_base"]]
    has = st["instance"].notna() & st["instance"].isin(inst.index)
    st["bx"] = np.nan
    st["by"] = np.nan
    st.loc[has, "bx"] = inst.loc[st.loc[has, "instance"].astype(int), "x_base"].to_numpy()
    st.loc[has, "by"] = inst.loc[st.loc[has, "instance"].astype(int), "y_base"].to_numpy()

    offsets, tf = plot_transforms(st[has], min_pairs, spread_floor_m)
    if reference is not None:
        tf = reference[0]

    # plot-corrected stem positions (all stems, matched or not)
    cx, cy = st["utm_x"].to_numpy(float).copy(), st["utm_y"].to_numpy(float).copy()
    for p, idx in st.groupby("parzelle").groups.items():
        t = tf.get(p)
        if t is not None:
            new = _apply(st.loc[idx, ["utm_x", "utm_y"]], *t)
            cx[idx], cy[idx] = new[:, 0], new[:, 1]
    st["cx"], st["cy"] = cx, cy

    ev = pd.DataFrame({"tree_id": st["tree_id"], "parzelle": st["parzelle"],
                       "instance": st["instance"], "matched": has})
    ev["offset_m"] = np.hypot(st["bx"] - st["utm_x"], st["by"] - st["utm_y"])
    rx, ry = st["bx"] - st["cx"], st["by"] - st["cy"]
    ev["offset_corrected_m"] = np.hypot(rx, ry)
    ev["plot_corrected"] = st["parzelle"].map(lambda p: tf.get(p) is not None)

    # uniqueness: second-best candidate instance and mutual nearest
    ev["second_best_margin_m"] = np.nan
    ev["second_gap_m"] = np.nan
    ev["mutual_best"] = False
    ibase = instances.drop_duplicates("instance").reset_index(drop=True)
    itree = cKDTree(ibase[["x_base", "y_base"]].to_numpy(float))
    stree = cKDTree(st[["cx", "cy"]].to_numpy(float))
    ipos = {v: i for i, v in enumerate(ibase["instance"])}
    hid = np.flatnonzero(has.to_numpy())
    if len(hid):
        kq = min(3, len(ibase))
        _, nn = itree.query(st.loc[hid, ["cx", "cy"]].to_numpy(float), k=kq)
        nn = nn.reshape(len(hid), -1)
        for a, i in enumerate(hid):
            j = ipos[int(st.at[i, "instance"])]
            d1 = ev.at[i, "offset_corrected_m"]
            others = [c for c in nn[a] if c != j]
            if others:
                o = others[0]
                d2 = float(np.hypot(ibase.at[o, "x_base"] - st.at[i, "cx"],
                                    ibase.at[o, "y_base"] - st.at[i, "cy"]))
                ev.at[i, "second_best_margin_m"] = d2 - d1
            # mutual: nearest field stem (corrected) of the matched instance is this tree
            _, ns = stree.query([ibase.at[j, "x_base"], ibase.at[j, "y_base"]], k=1)
            ev.at[i, "mutual_best"] = bool(ns == i)

    # neighbourhood: residual vector vs median residual vector of nearby matched trees
    ev["neighbourhood_residual_m"] = np.nan
    for p, idx in st[has].groupby("parzelle").groups.items():
        idx = np.asarray(list(idx))
        if len(idx) < 3:
            continue
        pos = st.loc[idx, ["cx", "cy"]].to_numpy(float)
        res = np.c_[rx[idx], ry[idx]]
        k = min(neighbours_k + 1, len(idx))
        _, nb = cKDTree(pos).query(pos, k=k)
        for a, i in enumerate(idx):
            others = [b for b in nb[a] if b != a][:neighbours_k]
            med = np.median(res[others], axis=0)
            ev.at[i, "neighbourhood_residual_m"] = float(np.hypot(*(res[a] - med)))
    if reference is not None:
        spread = reference[1]
    else:
        spread = ev.groupby("parzelle")["neighbourhood_residual_m"].median()
    ev["plot_spread_m"] = ev["parzelle"].map(spread).fillna(spread_floor_m).clip(lower=spread_floor_m)
    return ev, offsets, (tf, spread)


def _features(ev, offset_cap_m, ratio_cap=20.0):
    """Model inputs: offset after plot correction and the neighbourhood residual relative to
    the plot spread. Uniqueness (mutual nearest, margin) acts as a rule in `classify`; as
    model inputs they got sign-inconsistent weights because they are correlated with the offset.
    The offset is capped at the maximum match distance (the control cannot show larger values),
    so gross outliers among the real matches cannot bend the fit."""
    x = pd.DataFrame(index=ev.index)
    x["offset_corrected_m"] = ev["offset_corrected_m"].clip(upper=offset_cap_m)
    nbr = ev["neighbourhood_residual_m"].fillna(ev["offset_corrected_m"])
    x["nbr_ratio"] = (nbr / ev["plot_spread_m"]).clip(0, ratio_cap)
    return x[FEATURES]


# ---------------------------------------------------------------------------
# Chance control and calibration
# ---------------------------------------------------------------------------

def make_control(stems, instances, *, max_distance_m, shift_m=5.0, n_directions=8):
    """Shift every field stem by `shift_m` in several directions and match it to the
    nearest instance within `max_distance_m`. These matches are false by construction."""
    inst = instances.drop_duplicates("instance").reset_index(drop=True)
    tree = cKDTree(inst[["x_base", "y_base"]].to_numpy(float))
    parts = []
    for k in range(n_directions):
        ang = 2 * np.pi * k / n_directions
        s = stems.reset_index(drop=True).copy()
        s["utm_x"] = s["utm_x"] + shift_m * np.cos(ang)
        s["utm_y"] = s["utm_y"] + shift_m * np.sin(ang)
        d, j = tree.query(s[["utm_x", "utm_y"]].to_numpy(float), k=1)
        s["instance"] = np.where(d <= max_distance_m, inst["instance"].to_numpy()[np.minimum(j, len(inst) - 1)], np.nan)
        s["direction_deg"] = np.degrees(ang)
        parts.append(s)
    return parts


def fit_model(real_ev, control_evs, *, offset_cap_m, C=1.0):
    """Logistic regression: real matches (1) vs control matches (0), balanced classes.

    The real set contains some false matches, so the fitted probability is conservative.
    """
    xr = _features(real_ev[real_ev["matched"]], offset_cap_m)
    xc = pd.concat([_features(c[c["matched"]], offset_cap_m) for c in control_evs])
    X = pd.concat([xr, xc])
    y = np.r_[np.ones(len(xr)), np.zeros(len(xc))]
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, class_weight="balanced", max_iter=1000))
    model.fit(X.to_numpy(), y)
    return model


def score_evidence(ev, model, offset_cap_m):
    p = np.full(len(ev), np.nan)
    m = ev["matched"].to_numpy()
    if m.any():
        p[m] = model.predict_proba(_features(ev[m], offset_cap_m).to_numpy())[:, 1]
    return p


# ---------------------------------------------------------------------------
# Classes, reasons, flags
# ---------------------------------------------------------------------------

def classify(score, matched, mutual, margin, stem_points, *, confirmed_min=0.9, probable_min=0.7,
             ambiguous_min=0.3, min_stem_points=None, require_mutual=True, min_margin_m=0.5):
    """Class per tree from the score. Rules beyond the thresholds are returned as notes."""
    cls, notes = [], []
    for s, m, mu, mg, sp in zip(score, matched, mutual, margin, stem_points):
        note = ""
        if not m:
            cls.append("unmatched")
            notes.append(note)
            continue
        c = ("confirmed" if s >= confirmed_min else "probable" if s >= probable_min
             else "ambiguous" if s >= ambiguous_min else "rejected")
        if require_mutual and not mu and c in ("confirmed", "probable"):
            c, note = "ambiguous", "instance is nearer to another field tree"
        elif np.isfinite(mg) and mg < min_margin_m and c in ("confirmed", "probable"):
            c, note = "ambiguous", f"second candidate within {min_margin_m:g} m of the match"
        if (min_stem_points is not None and c == "confirmed" and np.isfinite(sp)
                and sp < min_stem_points):
            c, note = "probable", f"only {int(sp)} stem points at the field position"
        cls.append(c)
        notes.append(note)
    return cls, notes


def make_reasons(ev, notes):
    out = []
    for (_, r), note in zip(ev.iterrows(), notes):
        if not r["matched"]:
            out.append("no instance assigned")
            continue
        parts = []
        if r["plot_corrected"]:
            parts.append(f"offset {r['offset_corrected_m']:.1f} m after plot correction")
        else:
            parts.append(f"offset {r['offset_m']:.1f} m (plot too small to correct)")
        mg = r["second_best_margin_m"]
        if np.isfinite(mg):
            parts.append(f"second candidate {abs(mg):.1f} m {'further' if mg >= 0 else 'CLOSER'}")
        else:
            parts.append("no second candidate")
        if not r["mutual_best"]:
            parts.append("not mutual nearest")
        if np.isfinite(r["neighbourhood_residual_m"]) and r["neighbourhood_residual_m"] > 3 * r["plot_spread_m"]:
            parts.append(f"differs {r['neighbourhood_residual_m']:.1f} m from neighbours")
        if note:
            parts.append(note)
        out.append("; ".join(parts))
    return out


def make_flags(field, *, height_flag_rel=0.25, dbh_flag_rel=0.25):
    """height_flag / dbh_flag: 'ok', 'differs' or 'no_reference'. Flags only, never scored.

    Height reference is the tape measure column (height_massband) only.
    """
    n = len(field)
    hf = pd.Series("no_reference", index=field.index)
    if {"height_massband", "lidar_height"} <= set(field.columns):
        h, l = field["height_massband"].astype(float), field["lidar_height"].astype(float)
        ok = h.notna() & l.notna()
        hf[ok] = np.where((l[ok] - h[ok]).abs() <= height_flag_rel * h[ok], "ok", "differs")
    df = pd.Series("no_reference", index=field.index)
    if {"dbh_mm", "dbh_lidar_mm"} <= set(field.columns):
        d, l = field["dbh_mm"].astype(float), field["dbh_lidar_mm"].astype(float)
        ok = d.notna() & l.notna()
        if "dbh_quality" in field.columns:
            ok &= field["dbh_quality"].eq("ok")
        df[ok] = np.where((l[ok] - d[ok]).abs() <= dbh_flag_rel * d[ok], "ok", "differs")
    return df.to_numpy(), hf.to_numpy()


# ---------------------------------------------------------------------------
# Whole analysis (pure)
# ---------------------------------------------------------------------------

def identity_table(field, instances, *, max_distance_m, control_shift_m=5.0,
                   control_directions=8, min_pairs=5, neighbours_k=5, spread_floor_m=0.05,
                   min_margin_m=0.5, confirmed_min=0.9, probable_min=0.7, ambiguous_min=0.3,
                   min_stem_points=None, require_mutual=True, height_flag_rel=0.25,
                   dbh_flag_rel=0.25, orientation=None):
    """Score every match of `field` (columns: tree_id, parzelle, utm_x, utm_y, instance and
    optional stem_points_at_bh, dbh_mm, dbh_lidar_mm, dbh_quality, height_massband, lidar_height).

    Returns (identity DataFrame, plot offsets DataFrame, control DataFrame with the same
    evidence and scores of the false-by-construction matches, fitted model).
    """
    kw = dict(min_pairs=min_pairs, neighbours_k=neighbours_k, spread_floor_m=spread_floor_m)
    ev, offsets, ref = compute_evidence(field, instances, **kw)
    ctrl_stems = make_control(field, instances, max_distance_m=max_distance_m,
                              shift_m=control_shift_m, n_directions=control_directions)
    ctrl_evs = [compute_evidence(s, instances, reference=ref, **kw)[0] for s in ctrl_stems]
    model = fit_model(ev, ctrl_evs, offset_cap_m=max_distance_m)

    sp = (field["stem_points_at_bh"].astype(float).to_numpy() if "stem_points_at_bh" in field
          else np.full(len(field), np.nan))
    ckw = dict(confirmed_min=confirmed_min, probable_min=probable_min, ambiguous_min=ambiguous_min,
               min_stem_points=min_stem_points, require_mutual=require_mutual,
               min_margin_m=min_margin_m)

    ev["identity_score"] = score_evidence(ev, model, max_distance_m)
    ev["identity_class"], notes = classify(ev["identity_score"], ev["matched"], ev["mutual_best"],
                                           ev["second_best_margin_m"], sp, **ckw)
    ev["reasons"] = make_reasons(ev, notes)
    ev["stem_points_at_bh"] = sp
    ev["dbh_flag"], ev["height_flag"] = make_flags(field.reset_index(drop=True),
                                                   height_flag_rel=height_flag_rel,
                                                   dbh_flag_rel=dbh_flag_rel)
    un = ~ev["matched"]
    for c in ("offset_m", "offset_corrected_m", "second_best_margin_m", "neighbourhood_residual_m"):
        ev.loc[un, c] = np.nan
    ev.loc[un, "mutual_best"] = False

    ctrl = pd.concat([_control_scored(e, model, max_distance_m, ckw) for e in ctrl_evs], ignore_index=True)

    if orientation is not None:
        offsets["best_orientation"] = offsets["parzelle"].map(orientation)
    else:
        offsets["best_orientation"] = np.nan
    return ev[OUT_COLUMNS].copy(), offsets[OFFSET_COLUMNS], ctrl, model


def _control_scored(e, model, offset_cap_m, ckw):
    e = e.copy()
    e["identity_score"] = score_evidence(e, model, offset_cap_m)
    # stem evidence is not available at shifted positions: no stem rule in the control
    ck = dict(ckw, min_stem_points=None)
    e["identity_class"], _ = classify(e["identity_score"], e["matched"], e["mutual_best"],
                                      e["second_best_margin_m"], np.full(len(e), np.nan), **ck)
    return e[e["matched"]]


def quality_report(identity, control, verified_ids):
    """False-accept rate of the control and recall on verified matches."""
    n_c = len(control)
    far = float((control["identity_class"] == "confirmed").mean()) if n_c else np.nan
    far_p = float(control["identity_class"].isin(["confirmed", "probable"]).mean()) if n_c else np.nan
    v = identity[identity["tree_id"].isin(verified_ids)]
    return {"n_control": n_c, "false_accept_confirmed": far, "false_accept_confirmed_or_probable": far_p,
            "n_verified": len(v),
            "recall_confirmed": float((v["identity_class"] == "confirmed").mean()) if len(v) else np.nan,
            "recall_confirmed_or_probable":
                float(v["identity_class"].isin(["confirmed", "probable"]).mean()) if len(v) else np.nan}


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _opt(section, key, default):
    v = getattr(section, key, None) if not isinstance(section, dict) else section.get(key)
    return default if v is None else v


def run(cfg, tables_dir=None):
    """Read the input tables, write match_identity.csv and plot_offsets.csv to cfg.tables_dir."""
    import geopandas as gpd  # imported late: Excel readers must run first elsewhere

    src = str(tables_dir) if tables_dir is not None else str(cfg.tables_dir)
    rd = lambda n: pd.read_csv(os.path.join(src, n))

    det = rd("field_trees_detection.csv")
    tiers = rd("match_tiers.csv")
    inst = gpd.read_file(os.path.join(src, "lidar_instances.gpkg"))
    inst = pd.DataFrame(inst.drop(columns="geometry"))[["instance", "x_base", "y_base"]]

    # final match = tier table; only tiers that carry an instance id count as matched
    t = tiers.set_index("tree_id")
    inst_id = det["tree_id"].map(t["instance"])
    method = det["tree_id"].map(t["method"])
    field = det[["tree_id", "parzelle", "utm_x", "utm_y", "dbh_mm", "height_massband",
                 "has_height"]].copy()
    field["instance"] = inst_id.where(method.ne("none") & method.notna())
    field["lidar_height"] = det["tree_id"].map(t["lidar_height"])

    hd = rd("height_dbh_all_trees.csv").set_index("tree_id")
    field["dbh_lidar_mm"] = field["tree_id"].map(hd["dbh_lidar_mm"])
    field["dbh_quality"] = field["tree_id"].map(hd["dbh_quality"])
    # stem evidence at the field position at breast height: points in the stem zone
    # within the stem radius around the field stem (column_structure_per_tree)
    cs = rd("column_structure_per_tree.csv").set_index("tree_id")
    field["stem_points_at_bh"] = field["tree_id"].map(cs["stem_zone_points"])

    orient = None
    oc_path = os.path.join(src, "orientation_check.csv")
    if os.path.exists(oc_path):
        oc = pd.read_csv(oc_path)
        orient = dict(zip(oc["parzelle"], oc["best"]))

    idn = cfg.identity
    ident, offsets, ctrl, model = identity_table(
        field, inst,
        max_distance_m=cfg.matching.max_distance_m,
        control_shift_m=_opt(cfg.matching, "control_shift_m", 5.0),
        control_directions=_opt(idn, "control_directions", 8),
        min_pairs=_opt(idn, "min_pairs", 5),
        neighbours_k=_opt(idn, "neighbours_k", 5),
        spread_floor_m=_opt(idn, "spread_floor_m", 0.05),
        min_margin_m=_opt(idn, "min_margin_m", 0.5),
        confirmed_min=_opt(idn, "confirmed_min", 0.9),
        probable_min=_opt(idn, "probable_min", 0.7),
        ambiguous_min=_opt(idn, "ambiguous_min", 0.3),
        min_stem_points=_opt(idn, "min_stem_points", 20),
        require_mutual=_opt(idn, "require_mutual", True),
        height_flag_rel=_opt(idn, "height_flag_rel", 0.25),
        dbh_flag_rel=_opt(idn, "dbh_flag_rel", 0.25),
        orientation=orient)

    # verified = sample trees with a tape height + hand-checked matches
    verified = set(field.loc[field["has_height"].fillna(False).astype(bool), "tree_id"])
    verified |= set(tiers.loc[tiers["method"].astype(str).str.contains("override|manual"), "tree_id"])
    ov = cfg.paths.get("match_overrides") if hasattr(cfg.paths, "get") else None
    for cand in (ov, os.path.join(src, "match_overrides.csv")):
        if cand and os.path.exists(str(cand)):
            verified |= set(pd.read_csv(cand)["tree_id"])
    rep = quality_report(ident, ctrl, verified)

    out = str(cfg.tables_dir)
    os.makedirs(out, exist_ok=True)
    ident.to_csv(os.path.join(out, "match_identity.csv"), index=False)
    offsets.to_csv(os.path.join(out, "plot_offsets.csv"), index=False)
    print("class counts:", ident["identity_class"].value_counts().to_dict())
    print("quality:", rep)
    return ident, offsets, rep
