#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# export_gri_coefficients.R
#
# Purpose
#   Export the coefficients of the Bavarian GRI single-tree volume function
#   (Franz et al. 1973) from the R package ForestElementsR, together with a reference
#   grid of volumes computed by the package function v_gri(). The coefficient
#   table is used by a Python implementation of the same function; the
#   reference grid serves to verify that implementation.
#
# Usage (run from the project root; paths are resolved relative to the
#   location of this script, so the working directory does not matter)
#   Rscript scripts/export_gri_coefficients.R
#
# Requirements
#   R >= 4.4 and the package ForestElementsR (CRAN). Note: its dependency
#   Deriv currently has to be installed from the CRAN archive
#   (https://cran.r-project.org/src/contrib/Archive/Deriv/).
#
# Outputs
#   config/volume_gri_coefficients.csv
#       One row per species code (coding "tum_wwk_short") with the columns
#       species_code, species_name, c11, ..., c33. The element c{i}{j} is
#       C[i, j] of the 3 x 3 parameter matrix of the species group, where row i
#       multiplies the i-th element of c(1, log(d), log(d)^2) and column j the
#       j-th element of c(1, log(h), log(h)^2):
#         A      <- t(c(1, log(d), log(d)^2)) %*% C
#         form_h <- exp(A %*% c(1, log(h), log(h)^2))
#         V      <- max(0, pi / 40000 * d^2 * form_h)   [m3]
#       with V = 0 for d < 6.1 cm or h < 1.3 m.
#   tests/data/v_gri_reference.csv
#       Volumes (species_code, dbh_cm, height_m, v_gri_m3) from v_gri() on a
#       grid of all species codes, diameters and heights.
#
#   Harvest volume under bark (Efm o. R.) and increment (all species codes in
#   the coding "tum_wwk_short", 1 = spruce, 2 = fir, 3 = pine, 4 = larch,
#   5 = beech, 6 = oak, 7 = Douglas fir, 8 = ash/other broadleaves, 9 = alder,
#   10 = other conifers):
#   config/fer_harvest_reduction.csv
#       species_code, harv_loss, bark_red, factor. The factor is the one of
#       v_red_harvest_ubark(): V_Efm = V * (1 - harv_loss) * (1 - bark_red),
#       factor = (1 - harv_loss) * (1 - bark_red).
#   config/fer_yield_tables.json
#       The yield table that the package assigns to each species code in
#       ytables_tum_wwk_short_var_1 (codes 1 to 9; code 10 has no assignment
#       there). Layout: {"source": ..., "tables": {"<code>": {"table_name",
#       "name_international", "site_index_variable", "mai_variable",
#       "site_indexes": [...], "age": [...], "values": {"<variable>":
#       [[... one row per age, one value per site index ...]]}}}}. Missing
#       table entries are null. Variables: every variable of the table that
#       the functions of ForestElementsR use or report (h_q_m, ba_m2_ha,
#       v_m3_ha, pai_m3_ha_yr, mai_m3_ha_yr and the red_* variants, n_ha,
#       d_q_cm).
#   config/fer_gnfi3_parameters.csv
#       Parameters of d_age_gnfi3() and h_age_gnfi3() (growth functions of the
#       third national forest inventory) in the coding "tum_wwk_short":
#       species_code, d_alpha, d_beta, d_gamma, h_alpha, h_beta, h_gamma. With x
#       the known diameter (or height) at age t0, the value at age t is
#         x(t) = gamma * (x / gamma)^exp(-beta / alpha * (t^alpha - t0^alpha)).
#   tests/data/fer_efm_reference.csv
#       species_code, v_gri_m3, efm_m3 from v_red_harvest_ubark().
#   tests/data/fer_ytable_reference.csv
#       Results of site_index(), ytable_lookup() and stocking_level() on a grid
#       of species code x age x height or site index (x basal area), including
#       points outside the coverage of the table: fn, species_code, age, x
#       (height in m for site_index, site index otherwise), ba_m2_ha, result,
#       age_clamped, si_extrapolated. age_clamped is 1 where the package warns
#       that the age lies outside the table; si_extrapolated is 1 where the
#       package warns that the site index lies outside the table (for
#       site_index(): where the returned site index does).
#   tests/data/fer_gnfi3_reference.csv
#       fn (d or h), species_code, age_known, size_known, age, result from
#       d_age_gnfi3() and h_age_gnfi3().
#
# Licence note
#   The coefficients and all other exported package data (yield tables, growth
#   and reduction parameters) are data of ForestElementsR, licensed under
#   GPL (>= 3). The files in config/ are therefore not distributed with this
#   repository (they are git-ignored) and have to be generated locally. R is
#   used only for this one-off export; the calculations on site data are done
#   in Python.
#
# References
#   Franz, F., Bachler, B., Deckelmann, E., Kennel, R., Schmidt, A.,
#     Wotschikowski, U. (1973): Bayerische Waldinventur 1970/71,
#     Inventurabschnitt I: Grossrauminventur, Aufnahme- und
#     Auswertungsverfahren. Forstliche Forschungsberichte Muenchen 11.
#   Biber, P., Torano Caicoya, A. (2025): ForestElementsR: Data Structures
#     and Functions for Working with Forest Data. R package version 2.2.0,
#     https://CRAN.R-project.org/package=ForestElementsR
# ------------------------------------------------------------------------------

suppressPackageStartupMessages(library(ForestElementsR))

# --- project root from script location ----------------------------------------
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(file_arg) == 1L) {
  dirname(normalizePath(file_arg, winslash = "/"))
} else {
  normalizePath("scripts", winslash = "/") # interactive: assume project root
}
root <- normalizePath(file.path(script_dir, ".."), winslash = "/")

coef_file <- file.path(root, "config", "volume_gri_coefficients.csv")
ref_file  <- file.path(root, "tests", "data", "v_gri_reference.csv")
dir.create(dirname(coef_file), showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(ref_file),  showWarnings = FALSE, recursive = TRUE)

# --- coefficients -------------------------------------------------------------
par_tab <- ForestElementsR:::v_gri_param_orig
codes   <- as.integer(as.character(par_tab$species_id))

coding <- as.data.frame(fe_species_get_coding_table("tum_wwk_short"))
# The coding table lists individual species; the German group name (name_ger)
# is identical for all species of one code.
coding_ids <- as.integer(coding$species_id)
nm <- enc2utf8(as.character(coding$name_ger[match(codes, coding_ids)]))

C_list <- lapply(par_tab$C, function(m) matrix(as.numeric(m), 3, 3))
cmat <- t(vapply(C_list, function(m) as.vector(t(m)), numeric(9))) # row-major
colnames(cmat) <- c("c11", "c12", "c13", "c21", "c22", "c23",
                    "c31", "c32", "c33")
coefs <- data.frame(species_code = codes, species_name = nm, cmat,
                    stringsAsFactors = FALSE, check.names = FALSE)
coefs <- coefs[order(coefs$species_code), ]

out <- coefs
out[, colnames(cmat)] <- lapply(out[, colnames(cmat)],
                                function(x) format(x, digits = 17, trim = TRUE))
write.csv(out, coef_file, row.names = FALSE, fileEncoding = "UTF-8", na = "",
          quote = TRUE)

# --- reference grid from the package function ---------------------------------
grid <- expand.grid(species_code = coefs$species_code,
                    dbh_cm   = c(5, 6, 6.1, 7, 10, 15, 20, 30, 45, 60, 80),
                    height_m = c(1, 1.3, 3, 8, 15, 22, 30, 38, 45))
grid$v_gri_m3 <- unlist(lapply(seq_len(nrow(grid)), function(i) {
  as.numeric(v_gri(as.character(grid$species_code[i]),
                   grid$dbh_cm[i], grid$height_m[i]))
}))
ref_out <- grid
for (v in c("dbh_cm", "height_m", "v_gri_m3")) {
  ref_out[[v]] <- format(ref_out[[v]], digits = 17, trim = TRUE)
}
write.csv(ref_out, ref_file, row.names = FALSE, fileEncoding = "UTF-8",
          quote = FALSE)

# --- verification: recompute from the exported coefficient file ---------------
cf <- read.csv(coef_file, fileEncoding = "UTF-8", colClasses = "character")
cm <- as.matrix(sapply(cf[, colnames(cmat)], as.numeric))
rownames(cm) <- cf$species_code
v_from_csv <- function(code, d, h) {
  C <- matrix(cm[as.character(code), ], 3, 3, byrow = TRUE)
  if (d < 6.1 || h < 1.3) return(0)
  A <- c(1, log(d), log(d)^2) %*% C
  max(0, pi / 40000 * d^2 * exp(sum(A * c(1, log(h), log(h)^2))))
}
v_check <- mapply(v_from_csv, grid$species_code, grid$dbh_cm, grid$height_m)
denom   <- pmax(abs(grid$v_gri_m3), .Machine$double.xmin)
rel     <- ifelse(grid$v_gri_m3 == 0 & v_check == 0, 0,
                  abs(v_check - grid$v_gri_m3) / denom)
max_rel <- max(rel)
if (!is.finite(max_rel) || max_rel > 1e-12) {
  stop(sprintf("Coefficient layout does not reproduce v_gri(): max relative difference %.3g",
               max_rel))
}
message(sprintf("Verified %d grid points for %d species codes; max relative difference %.3g.",
                nrow(grid), nrow(coefs), max_rel))
message("Written: ", coef_file)
message("Written: ", ref_file)

# ==============================================================================
# Harvest volume under bark, yield tables and growth functions
# ==============================================================================
ns <- asNamespace("ForestElementsR")
tum <- function(code) as_fe_species_tum_wwk_short(as.integer(code))
fmt <- function(x) format(x, digits = 17, trim = TRUE)
rel_diff <- function(a, b) {
  d <- abs(a - b) / pmax(abs(a), .Machine$double.xmin)
  d[is.na(a) & is.na(b)] <- 0
  d[a == 0 & b == 0] <- 0
  d
}
check <- function(what, a, b) {
  m <- max(rel_diff(a, b))
  if (!is.finite(m) || m > 1e-10) {
    stop(sprintf("%s: exported data do not reproduce the package function (max relative difference %.3g)",
                 what, m))
  }
  message(sprintf("Verified %s (%d values); max relative difference %.3g.", what, length(a), m))
}
red_file   <- file.path(root, "config", "fer_harvest_reduction.csv")
ytab_file  <- file.path(root, "config", "fer_yield_tables.json")
gnfi_file  <- file.path(root, "config", "fer_gnfi3_parameters.csv")
efm_ref    <- file.path(root, "tests", "data", "fer_efm_reference.csv")
ytab_ref   <- file.path(root, "tests", "data", "fer_ytable_reference.csv")
gnfi_ref   <- file.path(root, "tests", "data", "fer_gnfi3_reference.csv")

# --- reduction factors for the harvest volume under bark ------------------------
red_tab <- ns$param_volred_harv_ubark_tum_wwk_short
red <- data.frame(species_code = as.integer(vctrs::vec_data(red_tab$species_id)),
                  harv_loss = as.numeric(red_tab$harv_loss),
                  bark_red = as.numeric(red_tab$bark_red))
red$factor <- (1 - red$harv_loss) * (1 - red$bark_red)
red <- red[order(red$species_code), ]
red_out <- red
for (v in c("harv_loss", "bark_red", "factor")) red_out[[v]] <- fmt(red_out[[v]])
write.csv(red_out, red_file, row.names = FALSE, fileEncoding = "UTF-8", quote = FALSE)

efm_grid <- expand.grid(species_code = red$species_code,
                        v_gri_m3 = c(0, 0.05, 0.5, 1, 2.5, 7.3))
efm_grid$efm_m3 <- as.numeric(v_red_harvest_ubark(tum(efm_grid$species_code), efm_grid$v_gri_m3))
check("harvest reduction factors",
      efm_grid$efm_m3, efm_grid$v_gri_m3 * red$factor[match(efm_grid$species_code, red$species_code)])
efm_out <- efm_grid
for (v in c("v_gri_m3", "efm_m3")) efm_out[[v]] <- fmt(efm_out[[v]])
write.csv(efm_out, efm_ref, row.names = FALSE, fileEncoding = "UTF-8", quote = FALSE)

# --- yield tables assigned to the species codes ----------------------------------
assign_tab <- get("ytables_tum_wwk_short_var_1")   # package data
table_codes <- as.integer(vctrs::vec_data(assign_tab$species_id))
table_names <- as.character(assign_tab$ytable_name)
yield_variables <- c("h_q_m", "ba_m2_ha", "v_m3_ha", "pai_m3_ha_yr", "mai_m3_ha_yr",
                     "red_v_m3_ha", "red_pai_m3_ha_yr", "red_mai_m3_ha_yr", "n_ha", "d_q_cm")
ytabs <- lapply(table_names, get)
names(ytabs) <- as.character(table_codes)

jnum <- function(x) vapply(x, function(v) if (is.na(v)) "null" else fmt(v), "")
jstr <- function(x) paste0("\"", gsub("\"", "\\\\\"", enc2utf8(x)), "\"")
jarr <- function(x, str = FALSE) paste0("[", paste(if (str) jstr(x) else jnum(x), collapse = ", "), "]")
jmat <- function(m) {                       # rows = ages, columns = site indexes
  rows <- vapply(seq_len(nrow(m)), function(i) jarr(m[i, ]), "")
  paste0("[\n          ", paste(rows, collapse = ",\n          "), "\n        ]")
}
roundtrip <- function(x) {                  # the printed number must read back to the same double
  ok <- !is.na(x)
  all(as.numeric(vapply(x[ok], fmt, "")) == x[ok])
}
entries <- vapply(names(ytabs), function(code) {
  yt <- ytabs[[code]]
  vars <- intersect(yield_variables, names(yt$values))
  vals <- vapply(vars, function(v) {
    m <- as.matrix(yt$values[[v]])
    stopifnot(nrow(m) == length(yt$age_coverage), ncol(m) == length(yt$site_indexes))
    if (!roundtrip(as.numeric(m))) stop("number formatting does not round-trip for ", v)
    paste0("        ", jstr(v), ": ", jmat(m))
  }, "")
  paste0("    ", jstr(code), ": {\n",
         "      \"table_name\": ", jstr(table_names[match(as.integer(code), table_codes)]), ",\n",
         "      \"name_international\": ", jstr(yt$name_international), ",\n",
         "      \"site_index_variable\": ", jarr(yt$site_index_variable, TRUE), ",\n",
         "      \"mai_variable\": ", jarr(yt$mai_variable, TRUE), ",\n",
         "      \"site_indexes\": ", jarr(yt$site_indexes), ",\n",
         "      \"age\": ", jarr(yt$age_coverage), ",\n",
         "      \"values\": {\n", paste(vals, collapse = ",\n"), "\n      }\n    }")
}, "")
json <- paste0("{\n  \"source\": \"ForestElementsR ", as.character(packageVersion("ForestElementsR")),
               ", ytables_tum_wwk_short_var_1 (GPL >= 3)\",\n",
               "  \"coding\": \"tum_wwk_short\",\n  \"tables\": {\n",
               paste(entries, collapse = ",\n"), "\n  }\n}\n")
con <- file(ytab_file, open = "wb")
writeBin(charToRaw(enc2utf8(json)), con)
close(con)

# --- growth functions of the third national forest inventory --------------------------
pd <- ns$param_d_age_gnfi_2012_tum_wwk_short
ph <- ns$param_h_age_gnfi_2012_tum_wwk_short
stopifnot(identical(as.integer(vctrs::vec_data(pd$species_id)), as.integer(vctrs::vec_data(ph$species_id))))
gn <- data.frame(species_code = as.integer(vctrs::vec_data(pd$species_id)),
                 d_alpha = pd$alpha, d_beta = pd$beta, d_gamma = pd$gamma,
                 h_alpha = ph$alpha, h_beta = ph$beta, h_gamma = ph$gamma)
gn <- gn[order(gn$species_code), ]
gn_out <- gn
for (v in setdiff(names(gn), "species_code")) gn_out[[v]] <- fmt(gn_out[[v]])
write.csv(gn_out, gnfi_file, row.names = FALSE, fileEncoding = "UTF-8", quote = FALSE)

grid_d <- expand.grid(species_code = gn$species_code, age_known = c(20, 40, 70, 100),
                      size_known = c(8, 25, 45), age = c(25, 45, 60, 75, 120))     # cm
grid_h <- expand.grid(species_code = gn$species_code, age_known = c(20, 40, 70, 100),
                      size_known = c(6, 20, 32), age = c(25, 45, 60, 75, 120))     # m
grid_d$result <- d_age_gnfi3(tum(grid_d$species_code), grid_d$age, grid_d$size_known, grid_d$age_known)
grid_h$result <- h_age_gnfi3(tum(grid_h$species_code), grid_h$age, grid_h$size_known, grid_h$age_known)
grow <- function(p, x, t, t0) p$gamma * (x / p$gamma)^exp(-p$beta / p$alpha * (t^p$alpha - t0^p$alpha))
gn <- read.csv(gnfi_file)                                   # recompute from the exported file
pick <- function(g, pre) {
  i <- match(g$species_code, gn$species_code)
  list(alpha = gn[[paste0(pre, "_alpha")]][i], beta = gn[[paste0(pre, "_beta")]][i],
       gamma = gn[[paste0(pre, "_gamma")]][i])
}
check("growth function parameters (diameter)", grid_d$result,
      grow(pick(grid_d, "d"), grid_d$size_known, grid_d$age, grid_d$age_known))
check("growth function parameters (height)", grid_h$result,
      grow(pick(grid_h, "h"), grid_h$size_known, grid_h$age, grid_h$age_known))
gref <- rbind(cbind(fn = "d", grid_d), cbind(fn = "h", grid_h))
gref$result <- fmt(gref$result)
write.csv(gref, gnfi_ref, row.names = FALSE, fileEncoding = "UTF-8", quote = FALSE)

# --- reference values of the yield-table functions --------------------------------------
# Warnings of the package (age or site index outside the table) are returned as flags.
flagged <- function(expr) {
  age <- FALSE; si <- FALSE
  val <- withCallingHandlers(expr, warning = function(w) {
    msg <- conditionMessage(w)
    if (grepl("Site index", msg)) si <<- TRUE
    if (grepl("Selected age", msg)) age <<- TRUE
    invokeRestart("muffleWarning")
  })
  c(val, age, si)
}
ages    <- c(5, 15, 25, 40, 60, 90, 130, 170, 210)
heights <- c(4, 10, 18, 25, 32, 40)
sis     <- c(0.6, 1.5, 2.5, 3.5, 5.4)
rows <- list()
add <- function(fn, code, age, x, ba, res) {
  rows[[length(rows) + 1L]] <<- data.frame(fn = fn, species_code = code, age = age, x = x, ba_m2_ha = ba,
                                           result = res[1], age_clamped = as.integer(res[2]),
                                           si_extrapolated = as.integer(res[3]))
}
for (code in table_codes) {
  yt <- ytabs[[as.character(code)]]
  si_range <- range(yt$site_indexes)
  for (a in ages) {
    for (h in heights) {
      r <- flagged(site_index(a, h, yt, "h_q_m"))
      r[3] <- as.numeric(!is.na(r[1]) && (r[1] < si_range[1] || r[1] > si_range[2]))
      add("site_index", code, a, h, NA, r)
    }
    for (s in sis) {
      add("lookup_pai", code, a, s, NA, flagged(ytable_lookup(a, s, "pai_m3_ha_yr", yt)))
      add("lookup_mai", code, a, s, NA, flagged(ytable_lookup(a, s, "mai_m3_ha_yr", yt)))
      add("lookup_ba", code, a, s, NA, flagged(ytable_lookup(a, s, "ba_m2_ha", yt)))
      add("stocking_level", code, a, s, 25, flagged(stocking_level(25, a, s, yt)))
    }
  }
}
yref <- do.call(rbind, rows)

# self-check: rebuild the tables from the printed numbers and evaluate the package functions again
rebuild <- function(code) {
  yt <- ytabs[[as.character(code)]]
  yt$values <- lapply(yt$values[intersect(yield_variables, names(yt$values))], function(m) {
    m <- as.matrix(m)
    m[] <- as.numeric(vapply(as.numeric(m), function(v) if (is.na(v)) NA_character_ else fmt(v), ""))
    m
  })
  yt
}
rb <- lapply(table_codes, rebuild)
names(rb) <- as.character(table_codes)
pick_rows <- yref[seq(1, nrow(yref), by = 7), ]
again <- vapply(seq_len(nrow(pick_rows)), function(i) {
  r <- pick_rows[i, ]
  yt <- rb[[as.character(r$species_code)]]
  suppressWarnings(switch(r$fn,
    site_index = site_index(r$age, r$x, yt, "h_q_m"),
    lookup_pai = ytable_lookup(r$age, r$x, "pai_m3_ha_yr", yt),
    lookup_mai = ytable_lookup(r$age, r$x, "mai_m3_ha_yr", yt),
    lookup_ba = ytable_lookup(r$age, r$x, "ba_m2_ha", yt),
    stocking_level = stocking_level(r$ba_m2_ha, r$age, r$x, yt)))
}, numeric(1))
check("yield tables", pick_rows$result, again)
yref_out <- yref
for (v in c("x", "ba_m2_ha", "result")) yref_out[[v]] <- ifelse(is.na(yref[[v]]), "NA", fmt(yref[[v]]))
write.csv(yref_out, ytab_ref, row.names = FALSE, fileEncoding = "UTF-8", quote = FALSE, na = "NA")

message("Written: ", red_file)
message("Written: ", ytab_file)
message("Written: ", gnfi_file)
message("Written: ", efm_ref)
message("Written: ", ytab_ref)
message("Written: ", gnfi_ref)
