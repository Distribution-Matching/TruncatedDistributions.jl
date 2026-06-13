# Univariate tools

Beyond the multivariate truncated-normal machinery, the package ships a small
set of **univariate tools**. These are standalone utilities for one-dimensional
truncated distributions; they do not use the multivariate types.

## Dynamic moment matching

This is an implementation of

> B. Liquet and Y. Nazarathy, *A dynamic view to moment matching of truncated
> distributions*, Statistics & Probability Letters **104** (2015) 87–93.
> [doi:10.1016/j.spl.2015.05.006](https://doi.org/10.1016/j.spl.2015.05.006)

### The idea

Given a target interval ``[a, b]`` and desired moments, we want the parameters of
a distribution whose **truncation to ``[a, b]`` has those moments**. Solving this
directly is a nonlinear root-find. The dynamic method instead follows a *homotopy*.

For ``z \in (0, 1]`` define the expanding interval

```math
(l_z, h_z) = \left(a - \tfrac{1-z}{z},\; b + \tfrac{1-z}{z}\right).
```

As ``z \to 0`` this opens up to ``(-\infty, \infty)`` — the untruncated problem,
whose moment-matching solution is explicit. At ``z = 1`` it is the target
``[a, b]``. Differentiating the moment-matching conditions with respect to ``z``
gives an ODE for the parameter path ``\theta(z)`` that holds the desired moments
fixed all along the way. Integrating it from ``z = \varepsilon`` to ``z = 1``
lands on the answer — and the whole trajectory simultaneously solves the problem
for every intermediate truncation interval.

Internally the ODE is integrated in the reparameterisation ``s = (1-z)/z``, which
removes the ``1/z^2`` stiffness near ``z = 0`` and lets a fixed-step RK4 reach the
published trajectories to ``\sim 10^{-9}``. No external ODE solver is needed.

### Location-scale families (e.g. the normal)

[`dynamic_fit_locationscale`](@ref) handles any symmetric location-scale family
``f(x) = \tfrac{1}{\sigma}\varphi\!\left(\tfrac{x-\mu}{\sigma}\right)`` (default
kernel `Normal()`). Reproducing Fig. 2 of the paper — target mean `0.1`, standard
deviation `0.6`, on ``[-0.9, 1.35]``:

```julia
using TruncatedDistributions

r = dynamic_fit_locationscale(0.1, 0.6, -0.9, 1.35)
solution(r)        # ≈ [-0.20604, 1.12262]; the paper reports (-0.20606, 1.12264)
```

`solution(r)` returns `[location, scale]`. The returned
[`DynamicMomentMatch`](@ref) also carries the full path in `r.z` / `r.params`
(request intermediate points with `save_z`), so you can plot the trajectory as in
the paper.

#### Other kernels

The `kernel` may be any symmetric continuous distribution from
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) — e.g. a
Laplace (double-exponential) or a Student-t base:

```julia
using TruncatedDistributions, Distributions

# Laplace kernel (its density has a kink at the location — handled fine)
r = dynamic_fit_locationscale(0.2, 1.0, -1.0, 3.0; kernel = Laplace())

# Student-t with 5 degrees of freedom. Heavier tails ⇒ use a smaller `epsilon`
# so the homotopy starts from a wide enough interval. (Needs ν > 2 for a finite
# kernel variance.)
r = dynamic_fit_locationscale(0.2, 1.0, -1.0, 3.0; kernel = TDist(5), epsilon = 1e-3)
```

Accuracy depends on the tail weight. Light-tailed kernels (Normal, Laplace,
Logistic) recover the parameters to ``\sim 10^{-9}`` at the default `epsilon`;
Student-t with moderate degrees of freedom needs a smaller `epsilon` (``\nu=5``
reaches ``\sim 10^{-8}`` at `epsilon = 1e-3`), and very heavy tails (``\nu=3``)
remain limited because the initial, near-untruncated moment match is itself only
approached slowly as the interval opens.

### Exponential family

The exponential is a one-parameter, *non* location-scale family, so it follows a
separate scalar ODE via [`dynamic_fit_exponential`](@ref). Reproducing Fig. 1 —
target mean `2.4`, truncation ``[0, 5]`` (feasible since `mean < b/2`):

```julia
using TruncatedDistributions

r = dynamic_fit_exponential(2.4, 5.0; save_z = [1/3, 2/3])
solution(r)[1]     # ≈ 0.048046; the paper reports 0.0480456
```

At the recorded intermediate points the rate matches the paper's `0.28701`
(on ``[0,7]``) and `0.140213` (on ``[0,5.5]``).

### A note on a typo in the paper

The code uses the **general** coefficient formula, Eq. (11),
``p_i = \theta_2^2\,(h_i - l_i - i\,n_{i-1})``. The *explicit* list printed at the
bottom of p. 91 drops the integer factor ``i``: it shows
``p_2 = \theta_2^2(h_2 - l_2 - n_0 m_1^*)`` and
``p_3 = \theta_2^2(h_3 - l_3 - n_0 m_2^*)``, which should read ``-2 n_0 m_1^*`` and
``-3 n_0 m_2^*``. With the printed coefficients the scale ODE produces `NaN`s and
fails to reproduce the paper's own Fig. 2; with the Eq. (11) form (used here) it
matches to five figures.
