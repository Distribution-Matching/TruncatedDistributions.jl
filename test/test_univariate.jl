# 1D truncated-normal `moments` recurrence — closed-form checks against
# Distributions.jl's `mean`/`var` and the standard formula for a
# half-Gaussian.

using Distributions
using SpecialFunctions: erf

@testset "1D truncated normal — symmetric box about 0" begin
    # truncated(Normal(0,1), -c, c) is symmetric ⇒ mean = 0,
    # variance = 1 - 2c φ(c) / (Φ(c) − Φ(−c)).
    for c in (0.5, 1.0, 2.0)
        d = truncated(Normal(0.0, 1.0), -c, c)
        m = moments(d, 2)
        @test isapprox(m[1], 0.0; atol = 1e-10)
        Z = erf(c / sqrt(2))            # P(|X| ≤ c) for N(0,1)
        expected_var = 1 - 2c * pdf(Normal(), c) / Z
        @test isapprox(m[2] - m[1]^2, expected_var; atol = 1e-8)
    end
end

@testset "1D truncated normal — agreement with mean / var" begin
    cases = [
        (μ = 0.5, σ = 1.0, a = -1.0, b =  2.0),
        (μ = 2.0, σ = 1.5, a =  0.0, b =  4.0),
        (μ = 0.0, σ = 1.0, a = -Inf, b =  1.0),
        (μ = 0.0, σ = 2.0, a = -1.0, b =  Inf),
    ]
    for c in cases
        d = truncated(Normal(c.μ, c.σ), c.a, c.b)
        m = moments(d, 2)
        @test isapprox(m[1],           mean(d);                  atol = 1e-8)
        @test isapprox(m[2] - m[1]^2,  var(d);                   atol = 1e-8)
    end
end

@testset "1D truncated normal — moment(d, 0) = 1" begin
    d = truncated(Normal(0.0, 1.0), -2.0, 2.0)
    @test moment(d, 0) == 1
end
