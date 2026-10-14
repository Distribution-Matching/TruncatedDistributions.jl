# Moment matching for the box-truncated multivariate t: closed-form
# gradient vs central finite differences, the shared recursion tree, the
# ν → ∞ Gaussian limit, and parameter recovery by fit_mvt.

using TruncatedDistributions, Distributions, LinearAlgebra, Random, Test
import TruncatedDistributions: make_param_vec_from_μ_Σ

@testset "truncated-t moment matching" begin
    prev_backend = set_kr_base_backend!(:hcubature)
    prev_gamma = set_loss_gamma!(1.0)
    try
        @testset "shared tree: helper node is the t_ν law" begin
            ν, μ, Σ = 5.5, [0.2, -0.1], [1.0 0.3; 0.3 0.8]
            a, b = [-1.0, -1.5], [1.5, Inf]
            P = t_moment_matching_pair(ν, μ, Σ, a, b)
            d = TruncatedMvT(ν, μ, Σ, a, b; max_moment_levels = 2)
            for κ in ([0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2])
                @test raw_moment(P.base, κ) ≈ raw_moment(d, κ) rtol = 1e-9
            end
            d♯ = TruncatedMvT(ν + 2, μ, ν / (ν + 2) .* Σ, a, b; max_moment_levels = 4)
            @test raw_moment(P.sharp, [2, 2]) ≈ raw_moment(d♯, [2, 2]) rtol = 1e-9
        end

        @testset "gradient vs central FD: n=$(length(μ)), b=$(b)" for (ν, μ, Σ, a, b) in (
                (4.5, [0.2, -0.1], [1.0 0.3; 0.3 0.8], [-1.0, -1.5], [1.5, 1.0]),
                (6.0, [0.2, -0.1], [1.0 0.3; 0.3 0.8], [-1.0, -1.5], [Inf, 1.0]),
                (2.5, [0.1, 0.0], [0.9 -0.2; -0.2 1.1], [-1.0, -1.0], [1.0, 1.2]),
                (7.0, [0.1, 0.0, -0.2], [1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 0.9],
                 [-1.0, -1.5, -1.2], [1.5, 1.0, 1.3]))
            μ̂ = μ .+ 0.1
            Σ̂ = 0.6 .* Σ
            p = make_param_vec_from_μ_Σ(μ .- 0.05, 1.1 .* Σ)
            G = similar(p)
            vector_fg_true_loss_t(nothing, G, p, ν, a, b, μ̂, Σ̂)
            h = 1e-5
            Gfd = map(eachindex(p)) do k
                e = zeros(length(p)); e[k] = h
                (vector_moment_loss_t(p .+ e, ν, a, b, μ̂, Σ̂) -
                 vector_moment_loss_t(p .- e, ν, a, b, μ̂, Σ̂)) / (2h)
            end
            @test norm(G - Gfd) / norm(Gfd) < 1e-5
        end

        @testset "ν → ∞ recovers the Gaussian moments" begin
            μ, Σ = [0.2, -0.1], [1.0 0.3; 0.3 0.8]
            a, b = [-1.0, -1.5], [1.5, Inf]
            dt = TruncatedMvT(1e7, μ, Σ, a, b; max_moment_levels = 2)
            dn = TruncatedMvNormal(μ, Σ, a, b)
            @test mean(dt) ≈ mean(dn) rtol = 1e-5
            @test Matrix(cov(dt)) ≈ Matrix(cov(dn)) rtol = 1e-5
        end

        @testset "fit_mvt recovers (μ, Σ): $(label)" for (label, ν, a, b) in (
                ("bounded", 4.5, [-1.0, -1.5], [1.5, 1.0]),
                ("semi-infinite", 6.0, [-1.0, -1.5], [1.5, Inf]))
            μ★, Σ★ = [0.3, -0.2], [1.0 0.4; 0.4 1.3]
            d = TruncatedMvT(ν, μ★, Σ★, a, b; max_moment_levels = 2)
            μ̂, Σ̂ = mean(d), Matrix(cov(d))
            μ, Σ, info = fit_mvt(ν, μ̂, Σ̂, a, b; gamma = 1.0, ftarget = 1e-12,
                                 iterations = 200)
            @test info.loss < 1e-10
            @test μ ≈ μ★ atol = 1e-3
            @test Matrix(Σ) ≈ Σ★ atol = 1e-3
        end

        @testset "warm start matches each axis in isolation" begin
            ν, a, b = 5.0, [-1.0, -1.5], [1.5, Inf]
            μ̂, Σ̂ = [0.1, 0.2], [0.4 0.1; 0.1 0.9]
            μ0, Σ0 = warm_start_diagonal_t(ν, μ̂, Σ̂, a, b)
            for i in 1:2
                d = TruncatedMvT(ν, [μ0[i]], fill(Σ0[i, i], 1, 1), [a[i]], [b[i]];
                                 max_moment_levels = 2)
                @test mean(d)[1] ≈ μ̂[i] atol = 1e-4
                @test cov(d)[1, 1] ≈ Σ̂[i, i] atol = 1e-4
            end
            # correlation structure of Σ̂ is preserved exactly
            @test Σ0[1, 2] / sqrt(Σ0[1, 1] * Σ0[2, 2]) ≈ Σ̂[1, 2] / sqrt(Σ̂[1, 1] * Σ̂[2, 2])
        end

        @testset "argument checks" begin
            @test_throws ArgumentError fit_mvt(2.0, [0.0], fill(0.5, 1, 1), [-1.0], [Inf])
            @test_throws ArgumentError fit_mvt(-1.0, [0.0], fill(0.5, 1, 1), [-1.0], [1.0])
            @test_throws DimensionMismatch fit_mvt(5.0, [0.0, 0.0], fill(0.5, 1, 1), [-1.0], [1.0])
        end
    finally
        set_kr_base_backend!(prev_backend)
        set_loss_gamma!(prev_gamma)
    end
end
