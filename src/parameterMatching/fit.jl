"""
    fit_mvnormal(μ̂, Σ̂, a, b; method = :auto, n_threshold = 6, ...)

Find a multivariate-normal `(μ, Σ)` whose box truncation to `[a, b]` has
mean `μ̂` and covariance `Σ̂`. Returns `(μ, Σ, info)`, where `info` is a
NamedTuple carrying the final loss, wall time, method actually used and
algorithm-specific traces.

# Notes

The fit runs in the original `(μ, Σ)` coordinates with no global
standardisation; the loss
`L = ½‖μA − μ̂‖² + ½‖ΣA − Σ̂‖²_F` weights the mean- and
covariance-residual blocks equally. The coordinate warm-start does
per-axis `(μ, log σ²)` rescaling, which is usually enough for targets at
`O(1)` scale. For targets at very different scales, rescale the inputs
yourself or watch issue #7.

# Methods

* `:auto`  — pick `:lbfgs` for `length(μ̂) ≤ n_threshold` and `:bcd` for
            larger problems. The default `n_threshold = 6` reflects the
            crossover where a single full-dimension Kan–Robotti gradient
            call starts to cost minutes.
* `:lbfgs` — joint LBFGS over the packed `(μ, U)` parameter vector (with
            `U U^T = Σ^{-1}`), warm-started by a coordinate-wise diagonal
            fit. Recommended for small problems (n ≤ 6).
* `:bcd`   — hybrid block coordinate descent over blocks of size 1, 2, 3.
            Recommended for n ≥ 7 where the full-dimension polish is
            infeasible.

# Common keyword arguments

* `μ_init`, `Σ_init` — starting point for the optimiser. If omitted, the
                      coordinate warm-start is used.
* `ftarget`         — stop when the loss falls below this (default 1e-3).
* `time_limit`      — seconds (`:lbfgs` only; the BCD path is paced
                      instead by `iterations × bcd_inner_iters`).
* `iterations`      — outer-loop iteration cap (default 50 for LBFGS,
                      30 for BCD).
* `verbose`         — print one line per iteration.

# Returns

`(μ, Σ, info)`. The `info` NamedTuple has at least

* `method`     - symbol actually used
* `loss`       - final scalar loss `½‖μA − μ̂‖² + ½‖ΣA − Σ̂‖²_F`
* `time`       - wall-clock seconds
* `iterations` - outer iterations consumed

Method-specific entries (`hist`, `picks`, …) are passed through unchanged.

# Example

```julia
using TruncatedDistributions

μ̂ = [0.12, -0.12]
Σ̂ = [0.41 0.05; 0.05 0.41]
a = [-1.0, -1.5]
b = [ 1.5,  1.0]

μ, Σ, info = fit_mvnormal(μ̂, Σ̂, a, b)
```
"""
function fit_mvnormal(μ̂::AbstractVector, Σ̂::AbstractMatrix,
                     a::AbstractVector, b::AbstractVector;
                     method::Symbol = :auto,
                     n_threshold::Int = 6,
                     μ_init::Union{Nothing, AbstractVector} = nothing,
                     Σ_init::Union{Nothing, AbstractMatrix} = nothing,
                     ftarget::Float64 = 1e-3,
                     time_limit::Float64 = 60.0,
                     iterations::Union{Nothing, Int} = nothing,
                     verbose::Bool = false,
                     bcd_block_sizes::Vector{Int} = [1, 2, 3],
                     bcd_inner_iters::Int = 10,
                     bcd_accept_by::Symbol = :marginal,
                     bcd_selection::Symbol = :softmax,
                     bcd_softmax_T::Float64 = 1.0)
    n = length(μ̂)
    size(Σ̂) == (n, n) ||
        throw(DimensionMismatch("Σ̂ must be $(n)×$(n); got $(size(Σ̂))"))
    length(a) == n == length(b) ||
        throw(DimensionMismatch("a, b must have length $(n)"))
    all(a .< b) ||
        throw(ArgumentError("each a[i] must be strictly less than b[i]"))

    chosen = method === :auto ? (n <= n_threshold ? :lbfgs : :bcd) : method
    chosen in (:lbfgs, :bcd) ||
        throw(ArgumentError("method must be :auto, :lbfgs, or :bcd; got $method"))

    a_v = collect(Float64, a)
    b_v = collect(Float64, b)
    μ̂_v = collect(Float64, μ̂)
    Σ̂_m = Matrix{Float64}(Σ̂)

    # Warm-start unless caller supplied an initial point.
    if μ_init === nothing || Σ_init === nothing
        μ_ws, Σ_ws = warm_start_diagonal(μ̂_v, Σ̂_m, a_v, b_v)
        μ_init === nothing && (μ_init = μ_ws)
        Σ_init === nothing && (Σ_init = Σ_ws)
    end
    μ0 = collect(Float64, μ_init)
    Σ0 = Matrix{Float64}(Σ_init)

    if chosen === :lbfgs
        return _fit_mvnormal_lbfgs(μ̂_v, Σ̂_m, a_v, b_v, μ0, Σ0;
                                   ftarget = ftarget,
                                   time_limit = time_limit,
                                   iterations = iterations === nothing ? 50 : iterations,
                                   verbose = verbose)
    else  # :bcd
        return _fit_mvnormal_bcd(μ̂_v, Σ̂_m, a_v, b_v, μ0, Σ0;
                                 ftarget = ftarget,
                                 iterations = iterations === nothing ? 30 : iterations,
                                 verbose = verbose,
                                 block_sizes = bcd_block_sizes,
                                 inner_iters = bcd_inner_iters,
                                 accept_by = bcd_accept_by,
                                 selection = bcd_selection,
                                 softmax_T = bcd_softmax_T)
    end
end

# Joint LBFGS on the packed (μ, U) parameter vector with the explicit
# true-loss gradient. Equivalent to the "EXP-KR-MVN+WS" path in the paper.
function _fit_mvnormal_lbfgs(μ̂::Vector{Float64}, Σ̂::Matrix{Float64},
                             a::Vector{Float64}, b::Vector{Float64},
                             μ0::Vector{Float64}, Σ0::Matrix{Float64};
                             ftarget::Float64, time_limit::Float64,
                             iterations::Int, verbose::Bool)
    p0 = make_param_vec_from_μ_Σ(μ0, Σ0)
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Σ̂)
    opts = Optim.Options(show_trace = verbose,
                         iterations = iterations,
                         time_limit = time_limit,
                         callback = s -> s.value < ftarget)
    t = @elapsed res = optimize(Optim.only_fg!(fg!), p0, LBFGS(), opts)
    μ_fit, Σ_fit = make_μ_Σ_from_param_vec(res.minimizer)
    info = (; method = :lbfgs,
              loss = res.minimum,
              time = t,
              iterations = res.iterations,
              converged = Optim.converged(res),
              res = res)
    return μ_fit, Symmetric(Σ_fit), info
end

function _fit_mvnormal_bcd(μ̂::Vector{Float64}, Σ̂::Matrix{Float64},
                           a::Vector{Float64}, b::Vector{Float64},
                           μ0::Vector{Float64}, Σ0::Matrix{Float64};
                           ftarget::Float64,
                           iterations::Int, verbose::Bool,
                           block_sizes::Vector{Int},
                           inner_iters::Int,
                           accept_by::Symbol,
                           selection::Symbol,
                           softmax_T::Float64)
    # `time_limit` is honoured by BCD via the inner per-iteration cap of
    # `bcd_inner_iters`. The outer cap is `iterations`.
    t = @elapsed begin
        μ_fit, Σ_fit, hist, picks = block_coord_descent(
            μ̂, Σ̂, a, b;
            μ_init = μ0, Σ_init = Σ0,
            block_sizes = block_sizes,
            max_iters = iterations,
            inner_iters = inner_iters,
            ftarget = ftarget,
            monitor_full_loss = false,
            accept_by = accept_by,
            selection = selection,
            softmax_T = softmax_T,
            verbose = verbose)
    end
    info = (; method = :bcd,
              loss = isempty(hist) ? NaN : hist[end],
              time = t,
              iterations = length(picks),
              hist = hist,
              picks = picks)
    return μ_fit, Symmetric(Σ_fit), info
end
