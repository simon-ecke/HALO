# Methods

## Wood volume (core/volume.py)

Provisional. The volume function is a plug-in; the default form factors in `config/volume_form_factors.csv`
are approximate textbook values, not taken from a cited source. Every volume
number is only as good as that function.

Definitions
- `wood_volume_field` = f(species, field diameter `dbh_mm`, ULS height). `wood_volume_uls` = f(species, ULS
  diameter `dbh_lidar_mm`, ULS height). Species always comes from the field data. Both volumes use the same
  ULS height, so their difference is the diameter difference only.
- Default method `form_factor`: V = (pi/4) d^2 h f, d in m. Method `parameter_table`: V = a d^b h^c with a, b, c
  per species from a CSV (`volume.method`, `volume.params_file`, `volume.parameter_diameter_unit`); a `default`
  row is the fallback for species not in the file. New methods are added to `METHODS` in `core/volume.py`.
- Field heights (tape, Vertex; 33 sample trees) are not used for the general volume. For the sample trees the
  table also holds the volume with the tape height and with the Vertex height, each at the field diameter.
- Field trees without a matched instance get `height_source = modelled`: a Naeslund curve
  h = 1.3 + (d/(a+bd))^2 fitted on ULS heights of matched trees (per species from `volume.min_trees_per_species`
  trees, else one pooled curve). Their ULS volume stays empty. All totals are given with and without them.
- A missing field diameter or one outside `cleaning.dbh_plausible_mm` is not corrected: `dbh_field_valid = False`,
  no field volume, listed in `volume.field_excluded`. Before anything else the run checks that the median of
  field / ULS diameter on matched trees is near 1 (diameter, not circumference) and stops otherwise.

What the comparison tests
- It tests what the ULS diameter measurement means in m3: per-tree bias, relative bias, RMSE, median and MAD of
  ULS minus field volume (subsets `all_matched` and `detected_only`, bootstrap CI, species and diameter classes
  only with at least `volume.min_trees_per_species` trees), volume-weighted detection, and the stand gap split
  into trees ULS missed, diameter difference and ULS-only trees (the three parts sum to the gap).
- It does not test ULS height at stand level: the same ULS height enters both volumes and cancels in the
  difference. Height is checked only on the sample trees (tape and Vertex against ULS height, same diameter).
- It does not test the volume function itself, and the field diameter is the reference only in the sense of
  "what the surveyor measured"; ULS fits of quality `weak` are kept, with a separate `ok_fit` result.
- Area-based values (m3/ha) use the convex hull of the surveyed plot corners.

## Further analyses (core/analyses.py)

Tape (Maßband) is the height reference; Vertex and the mean are separate. Bootstrap: seed 0, 2000 resamples (`analyses.bootstrap`). Every statement names its n. No height-based re-matching anywhere.
- Error budget: on trees with tape, Vertex and ULS height, ULS-minus-tape against Vertex-minus-tape (the terrestrial noise floor): RMSE of each, paired-bootstrap interval of their difference, variance ratio, and the share of ULS-tape differences inside +/- 1.96 x the RMS of the Vertex-tape differences. "ULS is better/worse" is said only when the interval of the RMSE difference excludes 0.
- Identity subsets: height and diameter accuracy for `all_matched`, `detected_only`, `confirmed_only` (identity_class confirmed) and `confirmed_or_probable` (`accuracy_by_identity.csv`, ids `identity.*`). Improvement is claimed only if the RMSE intervals do not overlap (conservative).
- Sampling value (`sampling_value.csv`): stand mean ULS height of the matched trees from k trees drawn without replacement (k in `analyses.sampling.k`, 2000 draws) against the census mean. The census has no sampling error but is not free of measurement error. The actual 33 sample trees are felled trees chosen by the crew, not a random draw.
- Stand description (`stand_per_parzelle.csv`, `stand_per_parzelle_species.csv`): per Parzelle, ULS heights of all matched trees; detection rate (detected / field trees), diameter bias ULS minus field on ok circle fits; top height = mean ULS height of the thickest trees by field diameter, `analyses.top_height.trees_per_ha` (100) per hectare scaled to the plot area (corner polygon, at least one tree).
- Detection model (`detection_model.csv`): logistic regression of detected (1) against uncertain or not detected (0) on field diameter, Hegyi competition and relative diameter (columns in `analyses.detection_model`), each standardised; coefficients are log-odds per SD with bootstrap intervals; trees with a missing predictor are left out of the fit only. Diameter and relative diameter are correlated (r about 0.9), so their separate effects are poorly determined; rates per diameter class (Wilson interval) are the robust view.
- Ledger (`ledger.csv`): one row per issue (missing or implausible diameter, no instance, identity ambiguous or rejected, harvest record against LiDAR state, mirrored plots, sample trees with a non-routine remark, a height difference above `analyses.explain_height_diff_m`). A sample tree recorded as lying whose instance still stands after the harvest (removed fraction below `matching.standing_below`) is flagged by the data, no id is coded.
- Sample trees (`sample_trees.csv`): three heights, differences, identity, tier, cause, remark. `explanation` is filled when the difference to the tape (Vertex if no tape) exceeds the threshold, built from cause, identity class, tier and remark only.
- Height variants (`height_variants.csv`, only with `--height-variants`): highest point, 99th percentile of the point heights, highest canopy-model cell holding a point of the instance, against the tape, with leave-one-out RMSE range and paired RMSE change against the highest point. One streaming pass over the clipped cloud (about 35 s). No variant is adopted unless the interval of the change is below 0; with 21 tape trees this is not expected.
