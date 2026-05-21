# Distribution-object API around TruncatedMvNormal:
# pdf, logpdf, rand, sizing, var/std/cor, params, minimum/maximum,
# update_distribution! cache invalidation, BasicBoxTruncatedMvNormal sanity.

using LinearAlgebra
using PDMats
using Distributions
using Random

@testset "TruncatedMvNormal alias and constructors" begin
    μ = [0.0, 0.0]
    Σ = [1.0 0.3; 0.3 1.0]
    a = [-1.0, -1.5]
    b = [ 1.5,  1.0]

    d1 = TruncatedMvNormal(μ, Σ, a, b)                  # plain Matrix Σ
    d2 = TruncatedMvNormal(μ, PDMat(Σ), a, b)           # PDMat Σ
    d3 = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ), a, b)

    @test d1 isa TruncatedMvNormal
    @test d1 isa RecursiveMomentsBoxTruncatedMvNormal
    @test TruncatedMvNormal === RecursiveMomentsBoxTruncatedMvNormal

    # All three should give the same moments to high precision.
    @test isapprox(mean(d1), mean(d2); atol = 1e-8)
    @test isapprox(mean(d2), mean(d3); atol = 1e-8)
end

@testset "sizing, params, minimum/maximum" begin
    μ = [0.0, 0.0]; Σ = [1.0 0.3; 0.3 1.0]
    a = [-1.0, -1.5]; b = [1.5, 1.0]
    d = TruncatedMvNormal(μ, Σ, a, b)

    @test length(d) == 2
    @test size(d) == (2,)
    @test eltype(d) === Float64
    @test minimum(d) == a
    @test maximum(d) == b

    μp, Σp, ap, bp = params(d)
    @test μp == μ
    @test Matrix(Σp) ≈ Σ
    @test ap == a
    @test bp == b
end

@testset "insupport" begin
    d = TruncatedMvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0],
                          [-1.0, -1.0], [1.0, 1.0])
    @test insupport(d, [0.0, 0.0])
    @test insupport(d, [1.0, -1.0])    # box is closed
    @test !insupport(d, [1.5, 0.0])
    @test !insupport(d, [0.0, -1.5])
end

@testset "pdf and logpdf" begin
    μ = [0.5, 0.5]
    Σ = PDMat([1.0 1.2; 1.2 2.0])
    a = [-1.0, -Inf]; b = [0.5, 1.0]
    d = TruncatedMvNormal(μ, Σ, a, b)

    # Inside: pdf = pdf(untruncated) / tp; logpdf = logpdf(untruncated) - log(tp).
    x_in = [0.0, 0.0]
    pdf_in    = pdf(d, x_in)
    logpdf_in = logpdf(d, x_in)
    @test pdf_in > 0
    @test isapprox(pdf(MvNormal(μ, Σ), x_in) / tp(d), pdf_in; atol = 1e-10)
    @test isapprox(log(pdf_in), logpdf_in; atol = 1e-10)

    # Outside: pdf returns 0.0 (scalar), logpdf returns -Inf.
    @test pdf(d, [10.0, 10.0]) === 0.0
    @test logpdf(d, [10.0, 10.0]) == -Inf
end

@testset "rand — single and batch sampling" begin
    Random.seed!(20260521)
    d = TruncatedMvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0],
                           [-1.0, -1.0], [1.0, 1.0])

    x = rand(d)
    @test x isa Vector{Float64}
    @test length(x) == 2
    @test insupport(d, x)

    n = 500
    X = rand(d, n)
    @test size(X) == (2, n)
    for k in 1:n
        @test insupport(d, X[:, k])
    end
end

@testset "var / std / cor agree with cov" begin
    d = TruncatedMvNormal([0.0, 0.0], [1.0 0.3; 0.3 1.0],
                           [-1.0, -1.5], [1.5, 1.0])
    Σ = Matrix(cov(d))
    @test isapprox(var(d),  diag(Σ);                atol = 1e-10)
    @test isapprox(std(d),  sqrt.(diag(Σ));         atol = 1e-10)
    @test isapprox(cor(d),  Σ ./ (std(d) * std(d)'); atol = 1e-10)
end

@testset "update_distribution! refreshes the cache" begin
    μ1 = [0.0, 0.0]; Σ1 = [1.0 0.0; 0.0 1.0]      # plain Matrix — gets auto-wrapped
    a = [-1.0, -1.0]; b = [1.0, 1.0]
    d = TruncatedMvNormal(μ1, Σ1, a, b)

    m1 = copy(mean(d))
    c1 = Matrix(cov(d))
    @test isapprox(m1, [0.0, 0.0]; atol = 1e-6)

    # Bias the underlying mean — truncated mean should shift positive.
    μ2 = [0.6, 0.6]; Σ2 = [1.0 0.4; 0.4 1.0]
    update_distribution!(d.state, μ2, Σ2)
    d2 = outer_dist_from_state(d.state)
    m2 = mean(d2)
    c2 = Matrix(cov(d2))

    @test m2[1] > 0.1
    @test m2[2] > 0.1
    @test c2[1, 2] > 0          # off-diagonal correlation appears
    @test !isapprox(m1, m2; atol = 1e-3)
    @test !isapprox(c1, c2; atol = 1e-3)
end

@testset "BasicBoxTruncatedMvNormal — direct cubature path" begin
    μ = [0.0, 0.0]; Σ = [1.0 0.3; 0.3 1.0]
    a = [-1.0, -1.5]; b = [1.5, 1.0]
    d_basic = BasicBoxTruncatedMvNormal(μ, Σ, a, b)
    d_rec   = TruncatedMvNormal(μ, Σ, a, b)
    @test isapprox(tp(d_basic),     tp(d_rec);     atol = 1e-5)
    @test isapprox(mean(d_basic),   mean(d_rec);   atol = 1e-4)
    @test isapprox(Matrix(cov(d_basic)), Matrix(cov(d_rec));  atol = 1e-4)
end

@testset "show methods don't crash" begin
    d = TruncatedMvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0],
                           [-1.0, -1.0], [1.0, 1.0])
    io = IOBuffer()
    show(io, d)
    s = String(take!(io))
    @test occursin("TruncatedMvNormal", s)

    d_basic = BasicBoxTruncatedMvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0],
                                         [-1.0, -1.0], [1.0, 1.0])
    show(io, d_basic)
    s2 = String(take!(io))
    @test occursin("Basic", s2)
end
