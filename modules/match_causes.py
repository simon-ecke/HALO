"""Why a field tree is 'uncertain' or 'not detected': one cause per tree, from the evidence in the tables.

The cause is decided by rules on what the point cloud shows at the field position
(is there a stem, which instance do its points belong to, is there canopy) and on
the harvest record. It is a reading of the evidence, not a measurement: check the
QA figures of a cause before relying on its count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import notebook_common as nc

# cause -> who or what it is attributed to
CAUSES = {
    'tree found, harvest record disagrees': 'field record or timing',
    'tree found, instance partly gone': 'SegmentAnyTree',
    'merged with a neighbour': 'SegmentAnyTree',
    'own instance, base far from the stem': 'SegmentAnyTree',
    'stem visible, not segmented': 'SegmentAnyTree',
    'stem not visible in the LiDAR': 'LiDAR coverage',
    'no tree at the position on the flight day': 'field record or timing',
    'instance 1.5-3 m away, not confirmed by a stem': 'unclear',
}
ATTRIBUTIONS = ['SegmentAnyTree', 'field record or timing', 'LiDAR coverage', 'unclear']

CAUSE_EXPLANATION = {
    'tree found, harvest record disagrees':
        'Instance base within 1.5 m, and the instance is clearly gone or clearly standing after the harvest, '
        'but the record says the opposite. The tree itself is matched.',
    'tree found, instance partly gone':
        'Instance base within 1.5 m, but 20-80 % of the instance is gone after the harvest: the instance holds '
        'points of a felled and of a standing tree.',
    'merged with a neighbour':
        "A stem stands at the field position, but its points belong to the instance of another field tree: "
        'two stems, one instance.',
    'own instance, base far from the stem':
        'A stem stands at the field position and belongs to an instance, but the base of that instance was '
        'placed more than 1.5 m away.',
    'stem visible, not segmented':
        'A stem stands at the field position, but its points belong to no tree instance.',
    'stem not visible in the LiDAR':
        'Canopy at the position, but no stem circle in any of the eight slices from 1.5 to 8.5 m.',
    'no tree at the position on the flight day':
        f'The canopy at the position is more than {nc.NO_CANOPY_DEFICIT_M:.0f} m below the height a tree of this '
        'diameter should have: the tree did not stand on the flight day, or the position is wrong.',
    'instance 1.5-3 m away, not confirmed by a stem':
        'An instance was assigned at 1.5-3 m. No stem was found at the field position, or the stem belongs to no '
        'instance, so nothing confirms or contradicts the assignment.',
}


def classify_match_cause(profile: pd.DataFrame) -> pd.DataFrame:
    """One cause per uncertain or not detected field tree.

    Parameters
    ----------
    profile : one row per field tree with detection, base_dist_m, removed_fraction,
              removed_2026, canopy_deficit_m and stem_owner (tables/tree_profile_all.csv).

    Returns
    -------
    DataFrame with tree_id, detection, cause, attributed_to and stem_confirms_instance
    (True: the stem found at the field position lies in the assigned instance; False: it
    lies elsewhere; NA: no stem found, or no instance assigned). Detected trees get an
    empty cause.
    """
    owner = profile['stem_owner'].astype(str)
    own = owner == 'own instance'
    other_tree = owner == 'instance of another field tree'
    other_instance = owner == 'instance without a field tree'
    unsegmented = owner == 'not in any instance'
    no_stem = owner == 'no stem at the position'

    far = profile['base_dist_m'] > nc.HIGH_CONFIDENCE_DISTANCE_M
    clear_state = (profile['removed_fraction'] > nc.GONE_ABOVE) | (profile['removed_fraction'] < nc.STANDING_BELOW)
    not_detected = profile['detection'] == 'not detected'
    uncertain = profile['detection'] == 'uncertain'

    cause = pd.Series('', index=profile.index)
    # Not detected: no free instance base within 3 m
    nd_rules = [
        (profile['canopy_deficit_m'] > nc.NO_CANOPY_DEFICIT_M, 'no tree at the position on the flight day'),
        (other_tree, 'merged with a neighbour'),
        (other_instance, 'own instance, base far from the stem'),
        (unsegmented, 'stem visible, not segmented'),
        (no_stem, 'stem not visible in the LiDAR'),
    ]
    # Uncertain: an instance is assigned, but it is 1.5-3 m away or the harvest evidence disagrees
    unc_rules = [
        (other_tree, 'merged with a neighbour'),
        (far & (own | other_instance), 'own instance, base far from the stem'),
        (far, 'instance 1.5-3 m away, not confirmed by a stem'),
        (clear_state, 'tree found, harvest record disagrees'),
        (pd.Series(True, index=profile.index), 'tree found, instance partly gone'),
    ]
    for selected, rules in [(not_detected, nd_rules), (uncertain, unc_rules)]:
        open_rows = selected.copy()
        for condition, name in rules:
            hit = open_rows & condition
            cause[hit] = name
            open_rows &= ~hit
        cause[open_rows] = 'unclear'

    confirms = pd.Series(pd.NA, index=profile.index, dtype='boolean')
    has_instance = profile['instance'].notna() if 'instance' in profile.columns else profile['base_dist_m'].notna()
    confirms[has_instance & own] = True
    confirms[has_instance & (other_tree | other_instance | unsegmented)] = False

    return pd.DataFrame({
        'tree_id': profile['tree_id'], 'detection': profile['detection'].astype(str), 'cause': cause,
        'attributed_to': cause.map(CAUSES).fillna(cause.where(cause == 'unclear', '')),
        'stem_confirms_instance': confirms,
    })


def stand_conditions(trees: pd.DataFrame, instances: pd.DataFrame, radius_m: float = nc.COMPETITION_RADIUS_M,
                     structure_radius_m: float = 10.0) -> pd.DataFrame:
    """Neighbourhood of every field tree, from the stem map and the LiDAR instance table.

    Parameters
    ----------
    trees : tree_id, utm_x, utm_y, dbh_mm (field values).
    instances : x_base, y_base, height of the LiDAR instances.

    Returns
    -------
    DataFrame with tree_id and
    - larger_share_5m: share of the field stems within `radius_m` with a larger DBH
      (1 = the smallest tree of its group, NaN = no neighbour or no DBH)
    - instances_10m: LiDAR instances with their base within `structure_radius_m`
    - height_sd_10m, height_range_10m: spread of their heights (vertical structure)
    - low_share_10m: share of them below two thirds of the tallest (understorey share)
    """
    xy = trees[['utm_x', 'utm_y']].to_numpy(dtype=float)
    dbh = trees['dbh_mm'].where(trees['dbh_mm'].between(*nc.DBH_PLAUSIBLE_MM)).to_numpy(dtype=float)
    larger_share = np.full(len(trees), np.nan)
    for i, neighbours in enumerate(cKDTree(xy).query_ball_point(xy, radius_m)):
        neighbours = [j for j in neighbours if j != i and np.isfinite(dbh[j])]
        if neighbours and np.isfinite(dbh[i]):
            larger_share[i] = float(np.mean(dbh[neighbours] > dbh[i]))

    base_xy = instances[['x_base', 'y_base']].to_numpy(dtype=float)
    heights = instances['height'].to_numpy(dtype=float)
    rows = []
    for neighbours in cKDTree(base_xy).query_ball_point(xy, structure_radius_m):
        h = heights[neighbours]
        if len(h) >= 2:
            rows.append((len(h), float(h.std(ddof=1)), float(h.max() - h.min()), float(np.mean(h < 2 / 3 * h.max()))))
        else:
            rows.append((len(h), np.nan, np.nan, np.nan))
    out = pd.DataFrame(rows, columns=['instances_10m', 'height_sd_10m', 'height_range_10m', 'low_share_10m'])
    out.insert(0, 'larger_share_5m', larger_share)
    out.insert(0, 'tree_id', trees['tree_id'].to_numpy())
    return out
