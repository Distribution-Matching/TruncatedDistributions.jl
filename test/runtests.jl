using TruncatedDistributions
using Test
using Distributions
using PDMats
using LinearAlgebra
using SpecialFunctions: erf

import TruncatedDistributions: hcubature_inf

@testset "TruncatedDistributions" begin

    @testset "hcubature_inf — closed-form integrals" begin

        atol = 1e-6

        # ----- 1D, finite (must agree with hcubature) -----
        @test hcubature_inf(x -> 1.0,           [0.0], [1.0])[1] ≈ 1.0       atol = atol
        @test hcubature_inf(x -> exp(-x[1]^2),  [-1.0], [1.0])[1] ≈ √π * erf(1.0) atol = atol

        # ----- 1D, doubly infinite -----
        # ∫_{-∞}^{∞} e^{-x²} dx = √π
        @test hcubature_inf(x -> exp(-x[1]^2), [-Inf], [Inf])[1] ≈ √π atol = atol
        # standard normal pdf integrates to 1
        @test hcubature_inf(x -> pdf(Normal(), x[1]), [-Inf], [Inf])[1] ≈ 1.0 atol = atol

        # ----- 1D, half-infinite -----
        # ∫_{0}^{∞} e^{-x} dx = 1
        @test hcubature_inf(x -> exp(-x[1]), [0.0], [Inf])[1] ≈ 1.0 atol = atol
        # ∫_{-∞}^{0} e^{x} dx = 1
        @test hcubature_inf(x -> exp(x[1]),  [-Inf], [0.0])[1] ≈ 1.0 atol = atol
        # ∫_{a}^{∞} e^{-x} dx = e^{-a}
        @test hcubature_inf(x -> exp(-x[1]), [2.0], [Inf])[1] ≈ exp(-2.0) atol = atol

        # ----- 2D, doubly infinite -----
        # ∫∫ N(0,I) = 1
        Σ = [1.0 0.3; 0.3 1.0]
        d = MvNormal([0.0, 0.0], Σ)
        @test hcubature_inf(x -> pdf(d, collect(x)), [-Inf, -Inf], [Inf, Inf])[1] ≈ 1.0 atol = 1e-5

        # ----- 2D, mixed (one half-infinite, one finite) -----
        # ∫_{-1}^{1} ∫_{0}^{∞} e^{-x} dx dy = 2
        @test hcubature_inf(x -> exp(-x[1]), [0.0, -1.0], [Inf, 1.0])[1] ≈ 2.0 atol = atol

        # ----- 2D, mixed (one half-infinite, one doubly infinite) -----
        # ∫_{-∞}^{∞} ∫_{0}^{∞} e^{-x} N(y;0,1) dy dx = 1
        @test hcubature_inf(x -> exp(-x[1]) * pdf(Normal(), x[2]),
                            [0.0, -Inf], [Inf, Inf])[1] ≈ 1.0 atol = atol
    end

    @testset "Truncated MvNormal — Manjunath & Wilhelm (2021), Example 1" begin
        # Recursive moments must reproduce the published moments.
        μ = [0.5, 0.5]
        Σ = PDMat([1.0 1.2; 1.2 2.0])
        a_finite = [-1.0, -20.0]      # large finite surrogate for -∞
        b        = [ 0.5,   1.0]
        d = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a_finite, b)
        @test tp(d)        ≈ 0.398482903122761      atol = 1e-9
        @test mean(d)[1]   ≈ -0.1516343              atol = 1e-6
        @test mean(d)[2]   ≈ -0.3881151              atol = 1e-6
        @test cov(d)[1,1]  ≈  0.1630439              atol = 1e-6
        @test cov(d)[1,2]  ≈  0.1613371              atol = 1e-6
        @test cov(d)[2,2]  ≈  0.6062505              atol = 1e-6
    end

    include("test_gradients.jl")

end
