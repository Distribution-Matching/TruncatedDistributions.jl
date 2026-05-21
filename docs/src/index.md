# TruncatedDistributions.jl

A Julia package for truncated multivariate distributions. The current
focus is the **box-truncated multivariate normal**: a first-class
distribution object with `mean`, `cov`, `pdf`, `logpdf`, `rand`, plus
arbitrary multivariate raw moments via the recursive moment formula of
[Kan and Robotti (2017)](https://doi.org/10.1080/10618600.2017.1322092).
A moment-matching parameter-fitting layer sits on top.

The type hierarchy (`TruncatedMvDistribution{D, R, S}`) is generic in the
underlying distribution `D`, the truncation region `R`, and the cached
state `S`, so the package is designed to grow to other multivariate
families and other region types.

For univariate truncation use `Distributions.truncated` from
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl); this
package complements it with the multivariate case.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Distribution-Matching/TruncatedDistributions.jl")
```

Julia 1.10 or newer.

## Contents

```@contents
Pages = ["quickstart.md", "fitting.md", "internals.md", "api.md"]
Depth = 2
```

## Citing

If you use this package in published work, please cite Kan and Robotti
(2017) for the recursive moment formula and the companion paper
*Moment Matching of Box Truncated Multivariate Normal Distributions*
for the fitting algorithms.
