# End-to-end checks of the fitting pipeline:
#
#   * `warm_start_diagonal` exactly matches each diagonal moment to high
#     precision (the off-diagonal correlation structure is preserved from Σ̂
#     but is *not* re-fitted; the joint LBFGS / BCD does that).
#   * `fit_mvnormal(...; method = :lbfgs)` recovers the original parameters
#     of a known truncated MVN given its (μ̂, Σ̂) targets.
#   * `fit_mvnormal(...; method = :bcd)` likewise.
#   * `fit_mvnormal(...; method = :auto)` dispatches to the right algorithm.

using LinearAlgebra
using PDMats
using Distributions

# Given a known (μ, Σ, a, b), compute the truncated mean and covariance via
# the Kan–Robotti recursion, then ask `fit_mvnormal` to recover (μ, Σ) from
# (μ̂, Σ̂). The recovered parameters should reproduce the same (μ̂, Σ̂) to
# within `tol_moments`, but they need not be exactly (μ, Σ) — the
# moment-matching problem can have multiple solutions.

function _targets_from(μ, Σ, a, b)
    set_kr_base_backend!(:mvnormalcdf)
    d = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ), a, b)
    return collect(mean(d)), Matrix(cov(d))
end

@testset "warm_start_diagonal — diagonal moments match" begin
    μ = [0.0, 0.0]
    Σ = [1.0 0.3; 0.3 1.0]
    a = [-1.0, -1.5]
    b = [ 1.5,  1.0]
    μ̂, Σ̂ = _targets_from(μ, Σ, a, b)
    μ_ws, Σ_ws = warm_start_diagonal(μ̂, Σ̂, a, b)

    for i in 1:2
        d_i = truncated(Normal(μ_ws[i], sqrt(Σ_ws[i, i])), a[i], b[i])
        m   = moments(d_i, 2)
        @test isapprox(m[1], μ̂[i];               atol = 1e-5)
        @test isapprox(m[2] - m[1]^2, Σ̂[i, i];   atol = 1e-5)
    end
    # Correlation matrix should be preserved exactly.
    σ_ws = sqrt.(diag(Σ_ws))
    σ_hat = sqrt.(diag(Σ̂))
    @test isapprox(Σ_ws ./ (σ_ws * σ_ws'),
                   Σ̂   ./ (σ_hat * σ_hat'); atol = 1e-10)
end

@testset "fit_mvnormal :lbfgs — n=2 recovery" begin
    μ = [0.0, 0.0]
    Σ = [1.0 0.3; 0.3 1.0]
    a = [-1.0, -1.5]
    b = [ 1.5,  1.0]
    μ̂, Σ̂ = _targets_from(μ, Σ, a, b)

    μ_fit, Σ_fit, info = fit_mvnormal(μ̂, Σ̂, a, b; method = :lbfgs,
                                      ftarget = 1e-8, iterations = 100,
                                      verbose = false)

    @test info.method === :lbfgs
    @test info.loss < 1e-6
    d_fit = RecursiveMomentsBoxTruncatedMvNormal(μ_fit,
                                                 PDMat(Matrix(Σ_fit)),
                                                 a, b)
    # LBFGS converges to ~sqrt(ftarget) ≈ 1e-4 per moment entry; allow a
    # little slack for the QMC-based KR base case used at ≥ 2D.
    @test isapprox(mean(d_fit), μ̂; atol = 5e-4)
    @test isapprox(Matrix(cov(d_fit)), Σ̂; atol = 5e-4)
end

@testset "fit_mvnormal :bcd — n=2 recovery" begin
    μ = [0.0, 0.0]
    Σ = [1.0 0.3; 0.3 1.0]
    a = [-1.0, -1.5]
    b = [ 1.5,  1.0]
    μ̂, Σ̂ = _targets_from(μ, Σ, a, b)

    μ_fit, Σ_fit, info = fit_mvnormal(μ̂, Σ̂, a, b; method = :bcd,
                                      ftarget = 1e-6, iterations = 25,
                                      verbose = false)
    @test info.method === :bcd
    @test info.loss < 1e-3
    d_fit = RecursiveMomentsBoxTruncatedMvNormal(μ_fit,
                                                 PDMat(Matrix(Σ_fit)),
                                                 a, b)
    @test isapprox(mean(d_fit), μ̂; atol = 5e-2)
    @test isapprox(Matrix(cov(d_fit)), Σ̂; atol = 5e-2)
end

@testset "fit_mvnormal :auto dispatch" begin
    # Build realisable n=2 targets so the small-n :lbfgs path runs cleanly.
    μ = [0.0, 0.0]; Σ = [1.0 0.3; 0.3 1.0]
    a = [-1.0, -1.5]; b = [1.5, 1.0]
    μ̂, Σ̂ = _targets_from(μ, Σ, a, b)
    _, _, info_small = fit_mvnormal(μ̂, Σ̂, a, b; method = :auto, iterations = 1)
    @test info_small.method === :lbfgs

    # The n=8 path should dispatch to :bcd. Use realisable targets here too
    # — at n_threshold = 6, n = 8 picks :bcd regardless of the targets'
    # quality, so a single iteration is enough to verify dispatch.
    n = 8
    μ̂_big = zeros(n)
    Σ̂_big = Matrix{Float64}(I, n, n) .* 0.4
    a_big = fill(-1.5, n); b_big = fill(1.5, n)
    _, _, info_big = fit_mvnormal(μ̂_big, Σ̂_big, a_big, b_big;
                                  method = :auto, iterations = 1,
                                  bcd_inner_iters = 1)
    @test info_big.method === :bcd
end

@testset "fit_mvnormal — input validation" begin
    μ̂ = [0.0, 0.0]; Σ̂ = [1.0 0.0; 0.0 1.0]
    @test_throws DimensionMismatch fit_mvnormal(μ̂, Σ̂, [0.0], [1.0])
    @test_throws DimensionMismatch fit_mvnormal(μ̂, [1.0;;], [-1.0,-1.0],[1.0,1.0])
    @test_throws ArgumentError fit_mvnormal(μ̂, Σ̂, [0.0, 0.0], [-1.0, 1.0])
    @test_throws ArgumentError fit_mvnormal(μ̂, Σ̂, [-1.0,-1.0], [1.0,1.0];
                                            method = :badmethod)
end
