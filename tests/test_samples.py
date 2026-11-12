"""Synthetic test for core.samples.sample_points."""
import numpy as np
import pandas as pd

from core.samples import sample_points


def make_cloud():
    rng = np.random.default_rng(1)
    n = 3000
    return pd.DataFrame({"x": rng.uniform(-10, 10, n), "y": rng.uniform(-10, 10, n), "h": rng.uniform(0, 30, n),
                         "instance": rng.choice([0, 5, 7], n)})


def test_radius_groups_and_thinning():
    cloud = make_cloud()
    trees = pd.DataFrame({"tree_id": ["a", "b"], "x": [0.0, 5.0], "y": [0.0, 5.0], "instance": [5, np.nan]})
    out = sample_points(cloud, trees, radius_m=4.0, max_points=100, seed=0)
    assert set(out["tree_id"]) == {"a", "b"}
    assert (np.hypot(out["x_rel"], out["y_rel"]) <= 4.0 + 1e-9).all()
    assert out.groupby("tree_id").size().max() <= 100
    assert set(out["group"]) <= {"matched instance", "other instance", "no instance"}
    b = out[out["tree_id"] == "b"]
    assert "matched instance" not in set(b["group"])          # no matched instance for b
    again = sample_points(cloud, trees, radius_m=4.0, max_points=100, seed=0)
    pd.testing.assert_frame_equal(out, again)                 # fixed seed


def test_group_assignment_exact():
    cloud = pd.DataFrame({"x": [0.0, 1.0, 2.0, 50.0], "y": [0.0, 0.0, 0.0, 0.0], "h": [1.0, 2.0, 3.0, 4.0],
                          "instance": [5, 7, 0, 5]})
    trees = pd.DataFrame({"tree_id": ["a"], "x": [0.0], "y": [0.0], "instance": [5]})
    out = sample_points(cloud, trees, radius_m=4.0, max_points=10)
    assert out["group"].tolist() == ["matched instance", "other instance", "no instance"]
    assert out["x_rel"].tolist() == [0.0, 1.0, 2.0]
