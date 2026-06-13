# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Julia-style semantic versioning](https://pkgdocs.julialang.org/v1/compatibility/)
(for `0.x` releases, a bump of the minor version may include breaking changes).

## [0.3.0] — 2026-06-13

### Added

- **Univariate tools — dynamic (ODE-based) moment matching**, implementing
  Liquet & Nazarathy, *A dynamic view to moment matching of truncated
  distributions*, Stat. Probab. Lett. 104 (2015):
  - `dynamic_fit_locationscale(mean, std, a, b; kernel = Normal(), …)` for any
    symmetric location-scale family (validated for Normal, Laplace, Logistic,
    and Student-t kernels);
  - `dynamic_fit_exponential(mean, b; …)` for the exponential (a non
    location-scale family, on its own ODE path);
  - the `DynamicMomentMatch` result type (carrying the full parameter
    trajectory) and `solution`.

  Reproduces both worked examples from the paper, uses a dependency-free
  built-in RK4 integrator (via the `s = (1-z)/z` reparameterisation), and
  corrects a typo in the paper's printed coefficients (the explicit `pᵢ` list
  drops the integer factor `i` of the general Eq. (11)).
- `Statistics` `[compat]` entry, now that it is an independently-versioned
  standard library.

### Changed

- Migrated to **Optim 2** (built on DifferentiationInterface + ADTypes). New
  direct dependencies: `ADTypes`, `ForwardDiff`, `NLSolversBase`.
- Relaxed the `Parameters` `[compat]` to allow `0.13`.
- README and docs reframed: the package targets truncated distributions beyond
  what is in Distributions.jl, rather than being multivariate-only.

### Removed

- **Breaking:** dropped support for Optim 1.x (`[compat]` `Optim = "2"`). The
  Optim 2 autodiff API is incompatible with 1.x, which is the breaking change
  behind the `0.2.0 → 0.3.0` bump.

### Fixed

- Seeded the randomized-QMC (`:mvnormalcdf`) Kan–Robotti cross-check test and
  loosened its tolerance, removing an occasional CI flake.

## [0.2.0]

- Initial development releases: box-truncated multivariate normal distribution
  object, Kan–Robotti recursive moments with `:hcubature` and `:mvnormalcdf`
  backends, and the `fit_mvnormal` moment-matching layer (joint LBFGS +
  warm-start and hybrid block-coordinate descent).
