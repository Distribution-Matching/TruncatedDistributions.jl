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
    d = TruncatedMvNormal(μ, Σ, a, b)
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

@testset "warm_start_diagonal — semi-infinite axis and infeasible target" begin
    # Intro example of the paper: axis 3 is [-1.5, ∞).
    a, b = [-1.0, -2.0, -1.5], [2.0, 1.0, Inf]
    μ̂ = [0.47, -0.19, -0.12]
    Σ̂ = [0.54 0.16 0.11; 0.16 0.49 0.05; 0.11 0.05 0.65]
    μ_ws, Σ_ws = warm_start_diagonal(μ̂, Σ̂, a, b)
    for i in 1:3
        m = moments(truncated(Normal(μ_ws[i], sqrt(Σ_ws[i, i])), a[i], b[i]), 2)
        @test m[1] ≈ μ̂[i] atol = 1e-9
        @test m[2] - m[1]^2 ≈ Σ̂[i, i] atol = 1e-9
    end
    # Variance above the uniform's on a bounded interval: no normal fits.
    @test_throws InfeasibleMomentsError warm_start_diagonal([0.0], fill(0.5, 1, 1), [-1.0], [1.0])
end

@testset "fit_mvnormal :lbfgs — n=2 recovery" begin
    μ = [0.0, 0.0]
    Σ = [1.0 0.3; 0.3 1.0]
    a = [-1.0, -1.5]
    b = [ 1.5,  1.0]
    μ̂, Σ̂ = _targets_from(μ, Σ, a, b)

    μ_fit, Σ_fit, info = fit_mvnormal(μ̂, Σ̂, a, b; method = :lbfgs,
                                      ftarget = 1e-10, iterations = 100,
                                      verbose = false)

    @test info.method === :lbfgs
    @test info.loss < 1e-8
    d_fit = TruncatedMvNormal(μ_fit, Matrix(Σ_fit), a, b)
    # L < ftarget bounds the RMS covariance error by √(2·ftarget/γ) ≈ 1.4e-5
    # at the default γ = 1; allow slack for the QMC-based KR base case
    # used at ≥ 2D.
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
    d_fit = TruncatedMvNormal(μ_fit, Matrix(Σ_fit), a, b)
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

    # The n=9 path should dispatch to :bcd. Use realisable targets here too
    # — at n_threshold = 8, n = 9 picks :bcd regardless of the targets'
    # quality, so a single iteration is enough to verify dispatch.
    n = 9
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

@testset "loss gamma — averaged Frobenius weighting" begin
    set_kr_base_backend!(:mvnormalcdf)
    μ = [0.3, -0.2]; Σ = [1.0 0.4; 0.4 1.0]
    a = [-1.0, -1.5]; b = [1.5, 1.0]
    μ̂, Σ̂ = _targets_from(μ, Σ, a, b)
    # perturb so both residual blocks are nonzero
    d = TruncatedMvNormal([0.1, 0.0], [1.2 0.2; 0.2 0.9], a, b)
    μ̂p = μ̂ .+ 0.05; Σ̂p = Σ̂ .+ 0.03
    prev = get_loss_gamma()
    try
        set_loss_gamma!(1.0);  L1 = moment_loss(d, μ̂p, Σ̂p)
        set_loss_gamma!(0.2);  L2 = moment_loss(d, μ̂p, Σ̂p)
        set_loss_gamma!(0.0);  L0 = moment_loss(d, μ̂p, Σ̂p)  # mean-only
        # L(γ) is affine in γ: L(γ) = L0 + γ·(L1 − L0). Check the 0.2 point.
        @test isapprox(L2, L0 + 0.2 * (L1 - L0); rtol = 1e-10)
        @test L0 < L2 < L1        # more covariance weight ⇒ larger loss here
        # Exact averaged form: ½·mean((μA−μ̂)²) + γ·½·mean((ΣA−Σ̂)²).
        μA = mean(d); ΣA = Matrix(cov(d))
        @test isapprox(L1, 0.5 * sum(abs2, μA .- μ̂p) / 2 +
                           0.5 * sum(abs2, ΣA .- Σ̂p) / 4; rtol = 1e-3)
    finally
        set_loss_gamma!(prev)
    end
    @test get_loss_gamma() == prev
end

@testset "fit_mvnormal — seeded reproducibility" begin
    set_kr_base_backend!(:mvnormalcdf)
    μ = [0.2, 0.0, -0.1]; Σ = [1.0 0.3 0.2; 0.3 1.0 0.1; 0.2 0.1 1.0]
    a = fill(-1.5, 3); b = fill(1.5, 3)
    μ̂, Σ̂ = _targets_from(μ, Σ, a, b)
    r1 = fit_mvnormal(μ̂, Σ̂, a, b; method = :bcd, bcd_accept_by = :full, seed = 7)
    r2 = fit_mvnormal(μ̂, Σ̂, a, b; method = :bcd, bcd_accept_by = :full, seed = 7)
    @test r1[1] == r2[1]              # μ identical
    @test Matrix(r1[2]) == Matrix(r2[2])   # Σ identical
    @test r1[3].loss == r2[3].loss
end

@testset "fit_mvnormal — BCD polish" begin
    set_kr_base_backend!(:mvnormalcdf)
    μ = [0.2, 0.0, -0.1]; Σ = [1.0 0.3 0.2; 0.3 1.0 0.1; 0.2 0.1 1.0]
    a = fill(-1.5, 3); b = fill(1.5, 3)
    μ̂, Σ̂ = _targets_from(μ, Σ, a, b)
    _, _, info = fit_mvnormal(μ̂, Σ̂, a, b; method = :bcd, bcd_accept_by = :marginal,
                              bcd_polish = true, seed = 7)
    @test haskey(info, :polished)
    @test info.loss < 1e-2                       # polished to (near) target
    @test isfinite(info.loss)
end

@testset "config setters — round-trip and restore" begin
    g0 = get_loss_gamma()
    @test set_loss_gamma!(0.7) == g0             # returns previous
    @test get_loss_gamma() == 0.7
    set_loss_gamma!(g0)
    @test get_loss_gamma() == g0

    prev_rng = get_kr_base_rng()
    @test set_kr_base_rng!(MersenneTwister(3)) === prev_rng
    set_kr_base_rng!(prev_rng)
    @test get_kr_base_rng() === prev_rng

    prev_mc = set_moment_mc!(above = 9, samples = 12_345)
    @test set_moment_mc!(above = prev_mc[1], samples = prev_mc[2]) == (9, 12_345)
end

@testset "log-Cholesky parameter vector" begin
    μ = [0.2, -0.1, 0.3]; Σ = [1.0 0.3 0.1; 0.3 1.2 0.2; 0.1 0.2 0.9]
    p = make_param_vec_from_μ_Σ(μ, Σ)
    μ2, Σ2 = make_μ_Σ_from_param_vec(p)
    @test μ2 ≈ μ
    @test Σ2 ≈ Σ
    # Diagonal entries of V are the logs of those of U, and any real vector is valid.
    U = cholesky(Symmetric(inv(Σ))).U
    @test p[[4, 7, 9]] ≈ log.(diag(U))
    q = copy(p); q[[4, 7, 9]] .= -1.0
    @test isposdef(Symmetric(make_μ_Σ_from_param_vec(q)[2]))
    # An overshooting line search gets a finite penalty, not an exception.
    a = [-1.0, -1.0, -1.0]; b = [1.0, 1.0, 1.0]
    q[4] = -1000.0
    G = similar(q)
    @test isfinite(vector_fg_true_loss(1.0, G, q, a, b, μ, Σ))
    @test all(iszero, G)
end

@testset "fit(TruncatedMvNormal / TruncatedMvT, ...) returns the fitted law" begin
    μ = [0.2, -0.1]; Σ = [1.0 0.3; 0.3 0.8]
    a = [-1.0, -1.5]; b = [1.5, 1.0]
    d0 = TruncatedMvNormal(μ, Σ, a, b)
    mt, St = mean(d0), Matrix(cov(d0))
    d = fit(TruncatedMvNormal, mt, St, a, b; ftarget = 1e-12, seed = 1)
    @test d isa TruncatedMvNormal
    @test params(d)[3] == a && params(d)[4] == b
    @test maximum(abs, mean(d) - mt) < 1e-4
    @test maximum(abs, cov(d) - St) < 1e-4
    μf, Σf, _ = fit_mvnormal(mt, St, a, b; ftarget = 1e-12, seed = 1)
    @test params(d)[1] ≈ μf

    ν = 5.0
    dt0 = TruncatedMvT(ν, μ, Σ, a, b)
    mtt, Stt = mean(dt0), Matrix(cov(dt0))
    dt = fit(TruncatedMvT, ν, mtt, Stt, a, b; ftarget = 1e-12, seed = 1)
    @test dt isa TruncatedMvT
    @test maximum(abs, mean(dt) - mtt) < 1e-4
    @test maximum(abs, cov(dt) - Stt) < 1e-4
end

@testset "fit on an unattainable target (example 2.3, rounded)" begin
    a, b = [-1.0, -Inf], [0.5, 1.0]
    μ̃, Σ̃ = [-0.2, -0.4], [0.2 0.2; 0.2 0.6]
    # The warm start throws on the first axis (no univariate solution);
    # the fit falls back to the naive start, warns, and still minimises.
    @test_throws InfeasibleMomentsError warm_start_diagonal(μ̃, Σ̃, a, b)
    d0 = @test_logs (:warn, r"Warm start failed") match_mode = :any fit(
        TruncatedMvNormal, μ̃, Σ̃, a, b; iterations = 20, seed = 1)
    @test d0 isa TruncatedMvNormal
    # From the naive start the fit runs, stalls above the target, and warns.
    d = @test_logs (:warn, r"Loss target not reached") match_mode = :any fit(
        TruncatedMvNormal, μ̃, Σ̃, a, b; μ_init = μ̃, Σ_init = Σ̃,
        ftarget = 1e-12, iterations = 20, seed = 1)
    @test d isa TruncatedMvNormal
end

@testset "fit rejects targets that no law on the box can have" begin
    a, b = [-1.0, -2.0], [2.0, 1.0]
    S = [0.5 0.1; 0.1 0.5]
    for f in (args -> fit_mvnormal(args...), args -> fit_mvt(5.0, args...))
        @test_throws ArgumentError f(([3.0, 0.0], S, a, b))              # mean outside A
        @test_throws ArgumentError f(([0.2, -0.2], [0.5 0.6; 0.6 0.5], a, b))  # not PD
        @test_throws ArgumentError f(([0.2, -0.2], [0.5 0.1; 0.3 0.5], a, b))  # not symmetric
        @test_throws ArgumentError f(([NaN, -0.2], S, a, b))                    # not finite
    end
    # A user start that puts no mass on the box (normal tails underflow).
    @test_throws ArgumentError fit_mvnormal([0.2, -0.2], S, a, b;
                                            μ_init = [60.0, 60.0], Σ_init = S)
end
