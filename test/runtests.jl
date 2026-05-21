using TruncatedDistributions
using Test
using Distributions
using PDMats
using LinearAlgebra
using SpecialFunctions: erf

import TruncatedDistributions: hcubature_inf

@testset "TruncatedDistributions" begin

    include("test_regions.jl")
    include("test_univariate.jl")
    include("test_distribution_api.jl")

    @testset "hcubature_inf — closed-form integrals" begin
        atol = 1e-6

        # 1D, finite (must agree with HCubature on the easy case)
        @test hcubature_inf(x -> 1.0,           [0.0],  [1.0])[1] ≈ 1.0             atol = atol
        @test hcubature_inf(x -> exp(-x[1]^2),  [-1.0], [1.0])[1] ≈ √π * erf(1.0)   atol = atol

        # 1D, doubly infinite
        @test hcubature_inf(x -> exp(-x[1]^2),       [-Inf], [Inf])[1] ≈ √π atol = atol
        @test hcubature_inf(x -> pdf(Normal(), x[1]), [-Inf], [Inf])[1] ≈ 1.0 atol = atol

        # 1D, half-infinite
        @test hcubature_inf(x -> exp(-x[1]), [0.0],  [Inf])[1] ≈ 1.0           atol = atol
        @test hcubature_inf(x -> exp(x[1]),  [-Inf], [0.0])[1] ≈ 1.0           atol = atol
        @test hcubature_inf(x -> exp(-x[1]), [2.0],  [Inf])[1] ≈ exp(-2.0)     atol = atol

        # 2D, doubly infinite — bivariate normal integrates to 1.
        Σ = [1.0 0.3; 0.3 1.0]
        d = MvNormal([0.0, 0.0], Σ)
        @test hcubature_inf(x -> pdf(d, collect(x)),
                            [-Inf, -Inf], [Inf, Inf])[1] ≈ 1.0 atol = 1e-5

        # 2D mixed.
        @test hcubature_inf(x -> exp(-x[1]),
                            [0.0, -1.0], [Inf, 1.0])[1] ≈ 2.0 atol = atol
        @test hcubature_inf(x -> exp(-x[1]) * pdf(Normal(), x[2]),
                            [0.0, -Inf], [Inf, Inf])[1] ≈ 1.0 atol = atol
    end

    @testset "Truncated MvNormal — Manjunath & Wilhelm (2021), Example 1" begin
        # Recursive moments must reproduce the published moments, both with
        # a large finite cap at -20 and with the true -Inf bound (the latter
        # exercises `hcubature_inf` in the base case).
        μ = [0.5, 0.5]
        Σ = PDMat([1.0 1.2; 1.2 2.0])
        b = [0.5, 1.0]
        for a in ([-1.0, -20.0], [-1.0, -Inf])
            d = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b)
            @test tp(d)       ≈ 0.398482903122761 atol = 1e-9
            @test mean(d)[1]  ≈ -0.1516343         atol = 1e-6
            @test mean(d)[2]  ≈ -0.3881151         atol = 1e-6
            @test cov(d)[1,1] ≈  0.1630439         atol = 1e-6
            @test cov(d)[1,2] ≈  0.1613371         atol = 1e-6
            @test cov(d)[2,2] ≈  0.6062505         atol = 1e-6
        end
    end

    @testset "KR backend toggle" begin
        prev = set_kr_base_backend!(:mvnormalcdf)
        @test get_kr_base_backend() === :mvnormalcdf
        set_kr_base_backend!(:hcubature)
        @test get_kr_base_backend() === :hcubature
        @test_throws ArgumentError set_kr_base_backend!(:bogus)
        set_kr_base_backend!(prev)
    end

    include("test_gradients.jl")
    include("test_kr_moments.jl")
    include("test_fit.jl")
end
