# Methods

## Wood volume, form-factor method (core/volume.py)

This method is kept for comparison; the GRI method below is the reference calculation.

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

## Wood volume, GRI method (core/volume_gri.py)

- Function: Bavarian GRI volume functions (Franz et al. 1973) as implemented in `v_gri()` of the R package ForestElementsR (Biber and Toraño Caicoya 2025). Result: merchantable wood volume over bark (Derbholz mit Rinde) in m3. With d = DBH in cm, h = height in m and a 3 x 3 coefficient matrix C per species group: ln(form height) = (1, ln d, ln^2 d) C (1, ln h, ln^2 h)^T; V = pi/40000 * d^2 * form height; V = 0 for d < 6.1 cm or h < 1.3 m.
- Units: the GRI volume is the standing merchantable volume over bark, Vorratsfestmeter (Vfm m. R.).
- Harvest volume: the harvest volume under bark, Erntefestmeter (Efm o. R.), is the standing volume times the species-specific reduction factor of ForestElementsR (`v_red_harvest_ubark`). Columns `efm_gri_*_m3` per tree; statistics `volume_gri.efm.*`.
- Species: each field species is assigned a GRI species group (coding `tum_wwk_short`, 10 groups) in `volume_gri.species_codes`. A species without a code gets no volume and is reported.
- Three volumes per tree:
  - field: field DBH and ULS height. Trees without a matched instance use a height from the height-diameter curve fitted on ULS heights, as in the form-factor method.
  - ULS: ULS DBH and ULS height.
  - practice: field DBH and a height from a Naeslund height-diameter curve fitted on the field-measured standing heights, applied to all trees including the measured ones. This is the volume an inventory without ULS would compute.
  - Field vs ULS isolates the diameter; practice vs field isolates the height source.
- Implementation: Python port. The coefficients are exported once from the R package with `scripts/export_gri_coefficients.R`. A test compares the port with `v_gri()` on a reference grid (`tests/data/v_gri_reference.csv`). The coefficient file is package data under GPL (>= 3) and is not distributed with this repository.
- Outputs: `wood_volume_gri_per_tree.csv`, `height_diameter_curves_field.csv`, `volume_gri_per_parzelle.csv`, `volume_gri_per_species.csv`; statistics with ids `volume_gri.*`.
- Dashboard: the tab "Wood volume (GRI)" recomputes the volumes in the browser when the species assignment is changed. Confidence intervals are shown for the configured assignment only.
- References:
  - Franz, F., Bachler, B., Deckelmann, E., Kennel, R., Schmidt, A., Wotschikowski, U. (1973): Bayerische Waldinventur 1970/71, Inventurabschnitt I: Großrauminventur, Aufnahme- und Auswertungsverfahren. Forstliche Forschungsberichte München 11.
  - Biber, P., Toraño Caicoya, A. (2025): ForestElementsR: Data Structures and Functions for Working with Forest Data. R package version 2.2.0, https://CRAN.R-project.org/package=ForestElementsR

## Increment (core/increment.py)

- Purpose and units: the periodic annual volume increment of the stand, in m3 standing volume over bark (Vfm m. R.) per ha and year, and as harvest volume under bark (Efm o. R.). Reported per Parzelle and for the stand (area-weighted). Three methods, following ForestElementsR.
- Stand age: not part of the inventory. It is set as `increment.stand_age_years` (approximate age at `increment.age_reference_year`); `increment.age_per_parzelle` overrides it for single Parzellen. The age-dependent methods are tabulated for every age in `increment.age_grid`, so that any assumed age in that range can be read from `increment_by_age.csv`. Without a stand age only the measured increment is reported.
- Yield table (stand level): main species = the species with the largest basal area. d_q is the quadratic mean of the valid field diameters; h_q is the height of the d_q tree on the height-diameter curve (field diameter x ULS height). The site index is obtained from age and h_q (`site_index`) using the yield table that ForestElementsR assigns to the main species. The current increment is read from the table (`ytable_lookup`) and multiplied by the stocking level (observed basal area over table basal area, `stocking_level`) where that is below 1. A site index outside the table is extrapolated and flagged; an age outside the table's coverage is replaced by the nearest tabulated age and flagged.
- Tree growth (per tree): diameter and height are predicted `increment.horizon_years` ahead (default 5) with the growth functions derived from the third German National Forest Inventory (`d_age_gnfi3`, `h_age_gnfi3`). The increment of a tree is the annual difference of its GRI volume over that horizon; the values are summed per ha. All standing trees are assumed to survive; there is no mortality and no removal.
- Measured increment (classical periodic increment from two surveys): increment(t1, t2) = V alive(t2) + V removed(between t1 and t2) − V alive(t1), with t1 the 2020 survey and t2 the current survey; periodic annual increment = increment / growing seasons / plot area (m3 per ha and year).
  - Volumes by GRI from the diameter of each survey and a height from the height-diameter curve of that survey: the curve for t1 is fitted on the 2020 diameters and the heights measured in 2020 (sheet column named in `increment.height_2020_column`); the curve for t2 is fitted on the current diameters and the current standing heights. The shift of the height curve between the surveys therefore enters the increment. The variant with one curve for both dates is reported as a comparison (`pai_static_curve`).
  - Tree classes: survivors (diameter at both surveys); trees removed earlier (flagged in the sheet; a 2020 diameter only); missing trees (2020 diameter, no current diameter, no flag); trees whose diameter change lies outside `increment.dbh_change_plausible_mm` are excluded at both dates as recording errors and counted. Trees felled in the current harvest were measured before felling and count as alive at t2.
  - Not recorded in the inventory, and therefore set as parameters: the number of growing seasons between the surveys (`increment.growing_seasons`, per survey year) and the time of removal of the flagged trees. Neither a removal year nor a diameter at removal exists; the removed stand enters with its 2020 volume plus an assumed growth, `increment.removed_growth_share` × 2020 volume × relative volume growth of the surviving trees of the same Parzelle (0 = removed at the 2020 survey, 1 = grew like the survivors for the whole period). Ingrowth is not recorded.
  - Reference area: the surveyed plot area (convex hull of the corner points), gaps included.
  - Uncertainty: bootstrap interval over the surviving trees for the configured parameters; the dashboard recalculates the increment for other growing seasons and removal assumptions without an interval.
- What the methods show: the predicted values depend strongly on the assumed age. The measured value is the reference against which they are judged. None of the three methods uses ULS data from two dates. The measured value has two main uncertainties of its own, the two facts that are not recorded in the inventory (growing seasons and time of removal of the flagged trees).
- Implementation: Python ports of the ForestElementsR functions (`site_index`, `ytable_lookup`, `stocking_level`, `d_age_gnfi3`, `h_age_gnfi3`, `v_red_harvest_ubark`), validated against reference values from the R functions (`tests/data/fer_*_reference.csv`). The yield tables and growth functions are documented in ForestElementsR. The exported yield tables, growth-function parameters and reduction factors are package data under GPL (>= 3) and are not distributed with this repository.
- Outputs: `increment_by_age.csv` (per age, per Parzelle and stand: yield table and tree growth values); `increment_measured.csv` (ledger per Parzelle and stand: `v_alive_t1_m3`, `v_alive_t2_m3`, `v_removed_m3`, `increment_m3`, `pai_measured_m3_ha_yr`, with the tree counts and the inputs to recompute the increment); `increment_per_tree.csv` (one row per 2020 tree with its `status`: survivor, removed_earlier, missing, excluded_* or no_volume); `height_diameter_curves_2020.csv`; and `field_trees_2020.csv`, the table of all trees of the 2020 survey written by the inventory stage (`core/io_field.py`). Statistics with ids `increment.*`.
- References: ForestElementsR (Biber and Toraño Caicoya 2025), as cited under the GRI volume method.

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
