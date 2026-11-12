"""Synthetic tests for core.identity (no project data)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import identity as I  # noqa: E402

MAXD = 3.0


def stand(n=70, size=45.0, min_dist=3.2, seed=0):
    """Random stand with a minimum spacing; returns instance table (true base positions)."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(0, size, 2)
        if all(np.hypot(*(p - q)) >= min_dist for q in pts):
            pts.append(p)
    pts = np.array(pts) + [500000.0, 5400000.0]
    return pd.DataFrame({"instance": np.arange(100, 100 + n), "x_base": pts[:, 0], "y_base": pts[:, 1]})


def field_from(inst, dx=0.5, dy=0.3, rot_deg=0.6, noise=0.08, seed=1):
    """Field stems = true positions moved by a rigid offset plus noise."""
    rng = np.random.default_rng(seed)
    xy = inst[["x_base", "y_base"]].to_numpy()
    c = xy.mean(axis=0)
    t = np.radians(rot_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    st = (xy - c) @ rot.T + c + [dx, dy] + rng.normal(0, noise, xy.shape)
    return pd.DataFrame({"tree_id": [f"1_{i}" for i in range(len(inst))], "parzelle": 1,
                         "utm_x": st[:, 0], "utm_y": st[:, 1], "instance": inst["instance"].to_numpy(),
                         "stem_points_at_bh": 200.0})


def analyse(field, inst, **kw):
    return I.identity_table(field, inst, max_distance_m=MAXD, **kw)


def test_shifted_rotated_stand_is_recovered_and_confirmed():
    inst = stand()
    f = field_from(inst)
    ident, offsets, ctrl, _ = analyse(f, inst)
    o = offsets.iloc[0]
    assert abs(o["rotation_deg"] + 0.6) < 0.25
    assert abs(o["dx_m"] + 0.5) < 0.12 and abs(o["dy_m"] + 0.3) < 0.12
    assert o["residual_rms_m"] < 0.2
    assert (ident["identity_class"] == "confirmed").mean() >= 0.8
    assert ident["offset_corrected_m"].median() < ident["offset_m"].median()
    # control matches are false by construction and must almost never be confirmed
    assert (ctrl["identity_class"] == "confirmed").mean() < 0.05


def test_swapped_neighbours_are_flagged():
    inst = stand()
    f = field_from(inst)
    tree = cKDTree(inst[["x_base", "y_base"]].to_numpy())
    used, pairs = set(), []
    for i in range(len(inst)):
        _, j = tree.query(inst.loc[i, ["x_base", "y_base"]].to_numpy(float), k=2)
        j = int(j[1])
        if i not in used and j not in used:
            pairs.append((i, j))
            used |= {i, j}
        if len(pairs) == 5:
            break
    g = f.copy()
    for i, j in pairs:
        g.loc[i, "instance"], g.loc[j, "instance"] = f.loc[j, "instance"], f.loc[i, "instance"]
    ident, _, _, _ = analyse(g, inst)
    swapped = [k for p in pairs for k in p]
    cls = ident.loc[swapped, "identity_class"]
    assert cls.isin(["ambiguous", "rejected"]).all(), cls.tolist()
    # the swapped pairs are false "real" matches, so the score of the others is lower (conservative)
    assert ident.drop(index=swapped)["identity_class"].isin(["confirmed", "probable"]).mean() > 0.9


def test_mirrored_plot_has_large_residual_and_few_confirmed():
    inst = stand()
    good = field_from(inst)
    mir = good.copy()
    mir["utm_y"] = 2 * mir["utm_y"].mean() - mir["utm_y"]
    tree = cKDTree(inst[["x_base", "y_base"]].to_numpy())
    d, j = tree.query(mir[["utm_x", "utm_y"]].to_numpy(), k=1)
    mir["instance"] = np.where(d <= MAXD, inst["instance"].to_numpy()[j], np.nan)
    ident_g, off_g, _, _ = analyse(good, inst)
    ident_m, off_m, _, _ = analyse(mir, inst)
    assert off_m["residual_rms_m"].iloc[0] > 3 * off_g["residual_rms_m"].iloc[0]
    assert (ident_m["identity_class"] == "confirmed").mean() < 0.2


def test_height_and_dbh_do_not_change_score_or_class():
    inst = stand()
    f = field_from(inst)
    rng = np.random.default_rng(5)
    base = f.assign(dbh_mm=400.0, dbh_lidar_mm=400.0, dbh_quality="ok",
                    height_massband=25.0, lidar_height=25.0)
    other = f.assign(dbh_mm=rng.uniform(80, 900, len(f)), dbh_lidar_mm=rng.uniform(80, 900, len(f)),
                     dbh_quality="ok", height_massband=rng.uniform(5, 40, len(f)),
                     lidar_height=rng.uniform(5, 40, len(f)))
    a, _, _, _ = analyse(base, inst)
    b, _, _, _ = analyse(other, inst)
    pd.testing.assert_series_equal(a["identity_score"], b["identity_score"])
    pd.testing.assert_series_equal(a["identity_class"], b["identity_class"])
    assert (a["height_flag"] == "ok").all() and (b["height_flag"] == "differs").any()
    assert (a["dbh_flag"] == "ok").all() and (b["dbh_flag"] == "differs").any()


def test_unmatched_trees_get_unmatched_class():
    inst = stand()
    f = field_from(inst)
    f.loc[:4, "instance"] = np.nan
    ident, _, _, _ = analyse(f, inst)
    assert (ident.loc[:4, "identity_class"] == "unmatched").all()
    assert ident.loc[:4, "identity_score"].isna().all()
