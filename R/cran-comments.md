# cran-comments.md

## Test environments

- Local: Ubuntu 26.04 LTS, R 4.5.2 (2025-10-31).
- GitHub Actions: standard r-lib/actions matrix (R release, devel, oldrel)
  on macOS-latest, windows-latest, ubuntu-latest.

## R CMD check results

Tested on the local environment above with
`R CMD check --as-cran --no-manual`:

```
Status: OK
```

(0 errors, 0 warnings; 1 NOTE about being a new submission, which is expected.)

## Downstream dependencies

This is a new submission. No reverse dependencies yet.

## Submission notes

- This package provides an R implementation of two methods documented in
  Chang (2026) and Silva and Zanella (2024). The DOI / arXiv link in the
  DESCRIPTION cites both.
- Several heavy Suggests (`brms`, `rstan`, `numDeriv`, `rmarkdown`,
  `knitr`) are gated at call sites with `requireNamespace(..., quietly =
  TRUE)`. Routine `R CMD check` does not need them; users who actually
  fit Stan models via `fit_quilt_brms` will need `brms` + `rstan`.
- The vignette source is shipped under `inst/doc/ais_loo.Rmd` (not as a
  built vignette) so the package does not require Pandoc at check time.
- All exported R6 classes and functions are documented; no examples are
  declared `\dontrun{}` since the runtime-sensitive examples (which need
  a Stan toolchain) are kept in the unit-test file gated behind the
  `BQ_RUN_BRMS_TESTS` env var.
