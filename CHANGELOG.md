# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Julia-style semantic versioning](https://pkgdocs.julialang.org/v1/compatibility/)
(for `0.x` releases, a bump of the minor version may include breaking changes).

## [0.5.0] — 2026-10-15

### Added

- **Box-truncated multivariate Student-t.** `TruncatedMvT(ν, μ, Σ, a, b;
  max_moment_levels)` with `mean`, `cov`, `tp`, `raw_moment` and
  `raw_moment_dict`. Raw moments come from a Student-t analogue of the
  Kan–Robotti recursion: each node uses a helper law with `ν − 2` degrees
  of freedom (from `∇f_{ν−2} = −Σ_H⁻¹(x−μ) f_ν`) and Student-t face
  conditionals with `ν − 1`. Non-integer `ν` is supported. On a box
  unbounded in some coordinate, a moment of order `L` needs `ν > L`
  (checked at construction).
- **t box probabilities** via the Genz–Bretz QMC routine of
  AdditionalDistributions.jl under the `:mvnormalcdf` backend (dimension
  ≥ 3; 2-D boxes use cubature, which is faster and more accurate there),
  or adaptive cubature under `:hcubature`. `set_kr_t_qmc_samples!` sets the
  QMC sample count (default `40_000`).
- **Moment matching for the truncated t.** `fit_mvt(ν, μ̂, Σ̂, a, b)`:
  warm-started (`warm_start_diagonal_t`) joint LBFGS on the closed-form
  gradient of the γ-weighted loss, assembled from truncated `t_ν` moments
  of order ≤ 2 and truncated `t_{ν+2}(μ, νΣ/(ν+2))` moments of order ≤ 4.
  One recursion tree serves both (`t_moment_matching_pair`, `TMomentPair`).
  Lower-level: `moment_loss_t`, `vector_moment_loss_t`,
  `vector_fg_true_loss_t`, `grad_true_loss_t`.
- New dependency: AdditionalDistributions.jl (`0.3`).

### Changed

- **Much faster Kan–Robotti recursion at higher `n`.** Face states are now
  shared: the conditional law given a set of fixed faces does not depend
  on the order in which they were fixed, so the recursion is a DAG with
  `Σ_d C(n,d) 2^d` states instead of a tree with `Σ_d n!/(n-d)! 2^d`; and
  each state is expanded only to the order its parent reads (children at
  order `L - 1`, none at order 0), so the depth is `L + 1` rather than `n`.
  An order-4 gradient now costs 0.15 s at `n = 5`, 1.5 s at `n = 7` and
  14 s at `n = 10` (previously 0.46 s, 79 s, and days). The truncated-t
  recursion shares states the same way (keyed by faces and helper steps).
- `fit_mvnormal`'s `:auto` dispatch now uses joint LBFGS up to
  `n_threshold = 8` (was 6).
- The hybrid block coordinate descent (`method = :bcd`,
  `block_coord_descent`) is now marked **experimental** in the docs: it
  returns approximate fits and its interface may change. Behaviour is
  unchanged, and `:auto` still uses it for `n ≥ 7`.

## [0.4.0] — 2026-07-07

### Added

- **Monte-Carlo moments for high dimension.** `mc_moments(d)` returns
  `(tp, μ, Σ)` by rejection sampling (in-place, allocation-light), and
  `tp(d)` / `mean(d)` / `cov(d)` fall back to it automatically for
  dimension `n > 6` (configurable via `set_moment_mc!`), where adaptive
  cubature is exponential and the Kan–Robotti tree is infeasible even to
  build. Fast at high `n` (n = 20 in ~0.1 s). Use
  `BasicBoxTruncatedMvNormal` at high `n`: the recursive `TruncatedMvNormal`
  constructor eagerly builds the full `2^{n-1}(n-1)!` child tree and is
  unconstructable past `n ≈ 8`, whereas the Basic type stores only
  `(μ, Σ)` and computes moments purely by MC.
- **Single covariance weight `gamma` for the moment-matching loss.** The
  loss is now `L = L1 + gamma·r(n)·L2` with `r(n) = 2/(n+1)` the ratio of
  mean to covariance parameter counts. `gamma` is scale-free: `gamma = 1`
  balances the two residual blocks for every `n`, and the new default
  `gamma = 0.2` weights the mean 5× the covariance. Set globally with
  `set_loss_gamma!` / `get_loss_gamma`, or per fit via the `gamma` keyword
  of `fit_mvnormal`. **Breaking:** the loss (and hence `moment_loss`,
  `info.loss`, and the fit thresholds) is no longer the unweighted
  `L1 + L2`.
- **Reproducible fits.** The `:mvnormalcdf` base case defaulted to
  `RandomDevice()` and ignored `Random.seed!`. A settable base-case RNG
  (`set_kr_base_rng!` / `get_kr_base_rng`) plus a `seed` keyword on
  `fit_mvnormal` (which seeds both the base case and the BCD softmax
  stream) make a fit — and its accept/reject path — repeat run to run.
- `fit_mvnormal` BCD path: new keyword arguments `bcd_polish` (run a short
  joint LBFGS polish from the BCD endpoint when the full n-dim loss is
  still above `ftarget`; default `false`) and `bcd_polish_iterations`
  (default 10). This makes the final "polish" step of the paper's
  Algorithm 2 available through the public front door; `info` gains
  `polished` and `loss_before_polish` fields.

### Fixed

- **Deep-tail robustness of the moment recursion.** The univariate
  truncated-normal `moments` recurrence divided by a naively-computed
  cdf difference, which cancels to exactly 0 when both bounds sit far
  in a tail (e.g. a finite box face 20σ out) and sent `Inf`/`NaN` up
  the Kan–Robotti tree; it now divides by the truncation mass that
  Distributions.jl computes stably in log space. Additionally, a
  recursion state whose box probability underflows to zero (or is
  non-finite) now zero-fills its raw-moment cache instead of recursing
  into degenerate children.
- `fit_mvnormal` BCD path: the loss monitor now matches the acceptance
  gate. With `bcd_accept_by = :full` the full n-dim loss (already
  computed for acceptance) is also the stopping criterion and the
  reported `info.loss`; previously the marginal-sum proxy was always
  used, so full-acceptance runs never noticed reaching `ftarget` and
  ground on to the iteration cap.
- **`mean(d)`, `cov(d)` and `tp(d)` on `TruncatedMvNormal` now use the
  Kan–Robotti moment cache** instead of silently falling back to generic
  n-dimensional adaptive cubature. The fallback was exponentially slow in
  dimension (≈ 220 s for a first `mean(d)` at n = 5 versus ≈ 0.4 s via
  the recursion) and contradicted the documented behaviour. The state's
  `tp_err` / `μ_err` / `Σ_err` fields are set to the base-case
  integrator's characteristic error scale (1e-5 for `:mvnormalcdf` at
  m = 10^4; 1e-6 for `:hcubature`), not a certified per-moment bound.

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
