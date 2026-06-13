# Dynamic (ODE-based) univariate moment matching — Liquet & Nazarathy (2015).
#
# We check two things:
#   1. Reproduction of the two worked examples from the paper (Fig. 1 exponential,
#      Fig. 2 location-scale normal).
#   2. Round-trip recovery: truncate a known distribution, read off its moments,
#      and confirm the solver recovers the original parameters.

@testset "Dynamic moment matching (univariate tools)" begin

    @testset "Exponential — paper Fig. 1 (m₁*=2.4, b=5)" begin
        # Paper: θ(1/3)=0.28701 on [0,7], θ(2/3)=0.140213 on [0,5.5], θ(1)=0.0480456 on [0,5].
        r = dynamic_fit_exponential(2.4, 5.0; epsilon = 0.01, save_z = [1/3, 2/3])
        idx(z) = findfirst(zz -> isapprox(zz, z; atol = 1e-9), r.z)
        @test r.params[idx(1/3)][1] ≈ 0.28701   atol = 1e-4
        @test r.params[idx(2/3)][1] ≈ 0.140213  atol = 1e-4
        @test solution(r)[1]        ≈ 0.0480456 atol = 1e-4
    end

    @testset "Normal location-scale — paper Fig. 2" begin
        # [a,b]=[-0.9,1.35], m₁*=0.1, sd=0.6  ->  (θ₁,θ₂)=(-0.20606, 1.12264).
        r = dynamic_fit_locationscale(0.1, 0.6, -0.9, 1.35; epsilon = 0.01)
        loc, scale = solution(r)
        @test loc   ≈ -0.20606 atol = 1e-4
        @test scale ≈  1.12264 atol = 1e-4
    end

    @testset "Location-scale round-trip recovery (Normal)" begin
        for (loc0, scale0, a, b) in [(0.0, 1.0, -1.0, 1.0),
                                     (0.5, 1.0, -1.0, 2.0),
                                     (0.0, 2.0, -2.0, 3.0),
                                     (-0.3, 1.5, -1.0, 4.0)]
            d  = truncated(Normal(loc0, scale0), a, b)
            mt = moments(d, 2)
            μt, σt = mt[1], sqrt(mt[2] - mt[1]^2)
            r  = dynamic_fit_locationscale(μt, σt, a, b)
            loc, scale = solution(r)
            @test loc   ≈ loc0   atol = 1e-5
            @test scale ≈ scale0 atol = 1e-5
        end
    end

    @testset "Exponential round-trip recovery" begin
        for (rate0, b) in [(0.5, 6.0), (1.0, 4.0), (0.3, 10.0)]
            d  = truncated(Exponential(1 / rate0), 0.0, b)   # Distributions: scale = 1/rate
            μt = mean(d)
            r  = dynamic_fit_exponential(μt, b)
            @test solution(r)[1] ≈ rate0 atol = 1e-5
        end
    end

    @testset "Location-scale with non-normal kernels" begin
        # Truncated mean/std of the family (kernel, loc0, scale0) on [a,b], computed
        # self-consistently by quadrature of pdf(kernel,(x-loc0)/scale0)/scale0.
        function _trunc_moments(kernel, loc0, scale0, a, b)
            f(x) = pdf(kernel, (x - loc0) / scale0) / scale0
            Z  = hcubature_inf(x -> f(x[1]),          [a], [b]; rtol = 1e-10, maxevals = 10^7)[1]
            m1 = hcubature_inf(x -> x[1] * f(x[1]),   [a], [b]; rtol = 1e-10, maxevals = 10^7)[1] / Z
            m2 = hcubature_inf(x -> x[1]^2 * f(x[1]), [a], [b]; rtol = 1e-10, maxevals = 10^7)[1] / Z
            return m1, sqrt(m2 - m1^2)
        end
        function _check(kernel, loc0, scale0, a, b; epsilon = 0.01, nsteps = 200_000, atol = 1e-6)
            μt, σt = _trunc_moments(kernel, loc0, scale0, a, b)
            r = dynamic_fit_locationscale(μt, σt, a, b;
                                          kernel = kernel, epsilon = epsilon, nsteps = nsteps)
            loc, scale = solution(r)
            @test loc   ≈ loc0   atol = atol
            @test scale ≈ scale0 atol = atol
        end
        # Laplace (kinked density at the location) and Logistic recover as tightly
        # as the normal at the default epsilon.
        _check(Laplace(),   0.3, 1.2, -1.0, 3.0)
        _check(Laplace(),  -0.2, 0.8, -2.0, 2.0)
        _check(Logistic(), -0.2, 0.8, -2.0, 2.0)
        _check(Logistic(),  0.5, 1.0, -1.5, 3.0)
        # Student-t (finite variance needs ν > 2). Heavier tails need a smaller
        # epsilon so the homotopy starts from a wide enough interval.
        _check(TDist(5), 0.3, 1.2, -1.0, 3.0; epsilon = 0.001,  atol = 1e-5)
        _check(TDist(5), 0.0, 1.0, -2.0, 2.5; epsilon = 0.001,  atol = 1e-5)
        _check(TDist(4), -0.2, 0.9, -2.0, 2.0; epsilon = 0.0005, nsteps = 300_000, atol = 1e-4)
    end

    @testset "trajectory + show" begin
        r = dynamic_fit_locationscale(0.0, 1.0, -2.0, 2.0; save_z = range(0.05, 1.0; length = 10))
        @test length(r.z) == length(r.params)
        @test issorted(r.z)
        @test r.z[end] == 1.0
        @test occursin("location-scale", sprint(show, r))
    end

    @testset "input validation" begin
        @test_throws ArgumentError dynamic_fit_locationscale(0.0, 1.0, 2.0, 1.0)   # a ≥ b
        @test_throws ArgumentError dynamic_fit_locationscale(0.0, -1.0, -1.0, 1.0) # std ≤ 0
        @test_throws ArgumentError dynamic_fit_exponential(3.0, 5.0)               # mean ≥ b/2
    end
end
