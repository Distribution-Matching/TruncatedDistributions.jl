# TruncatedDistributions.jl

A Julia package for truncated distributions, providing functionality beyond
what is in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
(whose `truncated` already covers basic univariate truncation).

Its main focus is the **box-truncated multivariate normal**: a first-class
distribution object with `mean`, `cov`, `pdf`, `logpdf`, `rand`, plus
arbitrary multivariate raw moments via the recursive moment formula of
[Kan and Robotti (2017)](https://doi.org/10.1080/10618600.2017.1322092).
A moment-matching parameter-fitting layer sits on top.

The type hierarchy (`TruncatedMvDistribution{D, R, S}`) is generic in the
underlying distribution `D`, the truncation region `R`, and the cached
state `S`, so the package is designed to grow to other multivariate
families and other region types.

The package also provides specialized **univariate tools** (see
[Univariate tools](tools.md)) — currently a dynamic, ODE-based moment
matcher — addressing problems that `Distributions.truncated` does not.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Distribution-Matching/TruncatedDistributions.jl")
```

Julia 1.10 or newer.

## Contents

```@contents
Pages = ["quickstart.md", "fitting.md", "tools.md", "internals.md", "api.md"]
Depth = 2
```

## Citing

If you use this package in published work, please cite Kan and Robotti
(2017) for the recursive moment formula and the companion paper
*Moment Matching of Box Truncated Multivariate Normal Distributions*
for the fitting algorithms.
