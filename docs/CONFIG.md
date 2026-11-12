# Site configuration

One YAML file per site in `config/` (example: `config/site_155.yaml`), checked against
`config/schema.json`. Relative paths are resolved against the project root; `${HALO_DATA_ROOT}` in a
path is replaced by the input data root (environment variable `HALO_DATA_ROOT` or the file
`data_root.local` in the project root). Load it with
`core.config.load_config(path)`; check it with `python -m core.validate config\site_<id>.yaml`.

## Top-level keys

- `site`: `id`, `name`, `crs` (a string pyproj can parse, e.g. `EPSG:25832`).
- `paths`: `output_dir`, `field_tables` (inventory sheets), `sample_heights_table`, `plot_corners`,
  `sat_before` and optional `sat_after` (`path` + `instance_dim`, the tree-id field of the LAZ),
  optional `ortho_before`, `match_overrides`, `manual_segments_dir`.
- `field_columns`: maps sheet columns to standard names so that no loader contains a column name.
  - `layouts`: one entry per inventory sheet: `file` (file name of a `paths.field_tables` entry),
    `sheet`, `dbh_year`, `columns` (standard name -> sheet column), `factors` (standard name ->
    multiplier into the standard unit, default 1), `earlier_removal_column` (any number there: tree
    removed in an earlier thinning, dropped) and `removal: {column, removed_value}` (the harvest mark).
  - `sample_heights`: columns of the sample-tree sheet (tape and Vertex heights).
  - `corners`: columns of the plot-corner sheet.
- `plots`: `id_column`, `y_mirrored` (plots whose local y is mirrored, y -> 30 - y).
- `grid`: AOI buffer, DTM/CHM resolution, voxel size for the removed-point test.
- `matching`: distance and removal thresholds of the stem-to-instance matching.
- `cleaning`: plausible DBH range, dm-typing threshold for standing heights.
- `dbh`: breast height, slice, search radius, ellipse axis ratio of the LiDAR DBH.
- `structure`: competition radius, small-tree DBH, edge distance and the other stand-structure limits.
- `baseline`: local-maximum windows (`diameter_m` + `per_height_m` * height) and top matching limits.
- `identity`, `volume`, `effort`: thresholds of `core/identity.py`, volume method and form-factor file,
  free-form effort section.

Plot styling is not configured here.

## Setting up a new site

1. Copy `config/site_155.yaml` to `config/site_<id>.yaml`; set `site` and `paths.output_dir`.
2. Point `paths` at the field sheets, corner sheet and LAZ files; set `instance_dim` per LAZ.
3. Write one `field_columns.layouts` entry per inventory sheet and fill `sample_heights` and `corners`
   (copy the column names from the sheet header row; check units and set `factors`).
4. Set `plots.y_mirrored` (empty list if no plot is mirrored; the orientation stage checks it) and
   review the thresholds in `grid`, `matching`, `cleaning`, `dbh`, `structure`, `baseline`.
5. Run `python -m core.validate config\site_<id>.yaml` until it prints `OK`, then run the pipeline.
