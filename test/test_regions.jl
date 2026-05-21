# Membership predicates for the truncation regions.

using LinearAlgebra
using PDMats

@testset "BoxTruncationRegion" begin
    r = BoxTruncationRegion([-1.0, -2.0], [1.0, 2.0])
    @test intruncationregion(r, [0.0, 0.0])
    @test intruncationregion(r, [-1.0, -2.0])  # closed box
    @test intruncationregion(r, [1.0, 2.0])
    @test !intruncationregion(r, [1.5, 0.0])
    @test !intruncationregion(r, [0.0, -2.5])
end

@testset "BoxTruncationRegion — ±Inf bounds" begin
    r = BoxTruncationRegion([-Inf, 0.0], [0.0, Inf])
    @test intruncationregion(r, [-1e6, 1e6])
    @test intruncationregion(r, [0.0, 0.0])
    @test !intruncationregion(r, [0.5, 1.0])
    @test !intruncationregion(r, [-1.0, -0.5])
end

@testset "EllipticalTruncationRegion" begin
    # Unit disc: (x - 0)' I (x - 0) <= 1
    r = EllipticalTruncationRegion(PDMat(Matrix{Float64}(I, 2, 2)),
                                   [0.0, 0.0], 1.0)
    @test intruncationregion(r, [0.0, 0.0])
    @test intruncationregion(r, [0.7, 0.7])   # ‖x‖² ≈ 0.98
    @test !intruncationregion(r, [0.8, 0.8])  # ‖x‖² ≈ 1.28
end
