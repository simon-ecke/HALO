# Match identity (core/identity.py)
Question per match: is the LiDAR instance the same tree as the field tree? Output: `match_identity.csv`, `plot_offsets.csv`.

## Evidence (geometry only)
1. Offset. `offset_m` = distance field stem to instance base. A robust rigid fit per Parzelle (rotation + translation, Tukey weights, scale from the median) maps stems onto their instances; `offset_corrected_m` is the distance left. `plot_offsets.csv`: `dx_m, dy_m` = shift applied to the stems at the plot's stem centroid, `rotation_deg`, `residual_rms_m` = rms over all pairs. Plots with fewer than `min_pairs` pairs get no fit (raw offset used).
2. Uniqueness. `second_best_margin_m` = distance (corrected stem) to the nearest other instance minus the distance to the matched one; negative = a competitor is closer. `mutual_best` = the matched instance's nearest corrected field stem is this tree.
3. Neighbourhood. `neighbourhood_residual_m` = how far the tree's fit residual differs from the median residual of its `neighbours_k` nearest matched trees; judged against the plot's median of this value. Swapped or wrong matches disagree with their surroundings.
4. Stem evidence. `stem_points_at_bh` = `stem_zone_points` (column_structure_per_tree): LiDAR points in the breast-height zone within the stem radius of the field stem.

Height and DBH never enter the score. `height_flag` (tape height vs LiDAR height; tape is the only reference) and `dbh_flag` (field vs LiDAR DBH, fit quality ok only) are `ok`, `differs` or `no_reference`, for later filtering only.

## Calibration
Chance control: all field stems are shifted by `control_shift_m` in `control_directions` directions and matched to the nearest instance within the maximum match distance; these matches are false by construction. They pass through the same evidence code, judged against the real plot fit and spread. A logistic regression (scikit-learn, standardised inputs, balanced classes) separates real (1) from control (0) matches using `offset_corrected_m` (capped at the match distance) and the neighbourhood residual divided by the plot spread. `identity_score` is its probability. Uniqueness and stem evidence are rules, not inputs: mutual and margin got sign-inconsistent weights (correlated with the offset), and stem points cannot be computed at shifted positions without the point clouds. The real set contains some false matches, so the score is conservative.

## Classes (thresholds are arguments)
- confirmed: score >= `confirmed_min`, mutual nearest, margin >= `min_margin_m`, and at least `min_stem_points` stem points (fewer: probable).
- probable: score >= `probable_min`, same uniqueness rule (otherwise ambiguous).
- ambiguous: score >= `ambiguous_min`, or a competitor is as near as the match.
- rejected: below `ambiguous_min`. unmatched: no instance assigned (includes tiers without an instance id, e.g. column-profile matches).

## Quality (printed by `run`)
False-accept rate = share of control matches classed confirmed. Recall = share of verified matches (sample trees with tape height, plus hand overrides) classed confirmed or probable.

## Limits
- The control tests position evidence only; a wrong match at a plausible position (a thinned-out neighbour 0.5 m away) is caught only through margin and neighbourhood.
- The plot fit is made on the matches it then judges, so a systematic error shared by a whole plot is absorbed. A large `residual_rms_m` (mirrored y, wrong corners) is the warning.
- Control matches are nearest-instance matches, unlike real mismatches from the global assignment; the false-accept rate is an estimate, not a guarantee.
- Thresholds are conventions; set them in `cfg.identity`.
