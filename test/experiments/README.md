# Experiments and benchmarks

This directory holds the scripts that generated the data in the companion
paper (`paper_truncated_mv_normal`). They are *not* part of the regular
test suite — most take minutes to hours to run.

The unit tests are at the level above (`test/runtests.jl`).

## Scripts

* `quick_bcd_n6.jl`         — hybrid BCD experiment for n = 6, …, 10
                              (full table in §6.3 of the paper).
* `quick_bcd_n2to5.jl`      — same experiment for n = 2, …, 5.
* `quick_bcd_sweep.jl`,
  `quick_bcd_highn.jl`,
  `quick_bcd_smoke.jl`      — additional BCD configurations / smoke tests.
* `benchmark_kr_mvn_high_n.jl`,
  `benchmark_kr_mvn_improvements.jl` — joint LBFGS + warm-start timing
                              data (§6.1, §6.2).
* `experiment_true_loss_gradient.jl` — finite-difference cross-check of
                              the explicit true-loss gradient.
* `profile_kr.jl`           — profiling of the Kan–Robotti recursion.
* `quick_alloc_check.jl`,
  `quick_update_dist.jl`,
  `quick_n6_probe.jl`       — diagnostics used during development.

## Running

From the package root:

```bash
julia --color=no --project=. test/experiments/quick_bcd_n6.jl
```

Some scripts assume the working directory is the package root; run them
from there rather than from inside `test/experiments/`.
