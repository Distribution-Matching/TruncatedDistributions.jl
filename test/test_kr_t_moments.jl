# Numerical correctness of the Student-t analogue of the Kan–Robotti
# recursion (boxTruncatedMvTRecursiveMomentsState.jl): every raw moment up
# to order L is compared with an independent HCubature integral of
# x^κ · f_ν(x) over the box. Infinite faces are mapped by x = a + t/(1-t)
# so that heavy-tailed integrands near ν ≈ L stay finite.

using HCubature

function _t_reference_moment(ν, μ, Σ, a, b, κ)
    D = MvTDist(ν, μ, Σ)
    n = length(a)
    g(t) = begin
        x = similar(t); J = 1.0
        for i in 1:n
            if isfinite(b[i])
                x[i] = a[i] + (b[i] - a[i]) * t[i]; J *= b[i] - a[i]
            else
                x[i] = a[i] + t[i] / (1 - t[i]); J /= (1 - t[i])^2
            end
        end
        prod(x[i]^κ[i] for i in 1:n) * pdf(D, x) * J
    end
    return hcubature(g, zeros(n), ones(n); rtol = 1e-9, maxevals = 10^6)[1]
end

function _t_multi_indices(n, L)
    out = Vector{Vector{Int}}()
    rec(p, r) = length(p) == n ? push!(out, copy(p)) :
        for v in 0:r; push!(p, v); rec(p, r - v); pop!(p); end
    rec(Int[], L)
    return out
end

@testset "Student-t recursion vs HCubature" begin
    prev = set_kr_base_backend!(:hcubature)
    try
        cases = [
            (ν = 4.5,  μ = [0.2],       Σ = fill(1.3, 1, 1),       a = [-1.2],       b = [Inf],      L = 3),
            (ν = 7.3,  μ = [-0.1],      Σ = fill(0.8, 1, 1),       a = [-1.0],       b = [2.0],      L = 4),
            (ν = 5.0,  μ = [0.3, -0.2], Σ = [1.0 0.4; 0.4 1.5],    a = [-1.0, -2.0], b = [1.5, 1.0], L = 4),
            (ν = 4.5,  μ = [0.0, 0.5],  Σ = [1.2 -0.3; -0.3 0.7],  a = [-1.5, -1.0], b = [Inf, 2.0], L = 3),
            (ν = 30.0, μ = [0.1, 0.1],  Σ = [1.0 0.6; 0.6 1.0],    a = [-1.0, -1.0], b = [1.0, 1.0], L = 4),
        ]
        for c in cases
            d = TruncatedMvT(c.ν, c.μ, c.Σ, c.a, c.b; max_moment_levels = c.L)
            @testset "ν=$(c.ν) n=$(length(c.μ)) L=$(c.L)" begin
                for κ in _t_multi_indices(length(c.μ), c.L)
                    @test raw_moment(d, κ) ≈ _t_reference_moment(c.ν, c.μ, c.Σ, c.a, c.b, κ) rtol = 1e-6 atol = 1e-8
                end
            end
        end
        # mean / cov read the cache consistently
        c = cases[3]
        d = TruncatedMvT(c.ν, c.μ, c.Σ, c.a, c.b)
        m0 = _t_reference_moment(c.ν, c.μ, c.Σ, c.a, c.b, [0, 0])
        @test mean(d) ≈ [_t_reference_moment(c.ν, c.μ, c.Σ, c.a, c.b, e) / m0 for e in ([1, 0], [0, 1])] rtol = 1e-6
        @test issymmetric(Matrix(cov(d)))
        # unbounded box: order-L moments need ν > L
        @test_throws ArgumentError TruncatedMvT(2.5, [0.0], fill(1.0, 1, 1), [0.0], [Inf]; max_moment_levels = 3)
    finally
        set_kr_base_backend!(prev)
    end
end

# A state of the t recursion with at most 2 degrees of freedom has no helper
# t_{ν-2}, so its moments are computed by direct cubature instead. For the
# gradient's root t_{ν+2} at order 4 this happens exactly when ν ≤ 3 (every
# state has degrees of freedom minus order equal to ν - 2). Check that the
# path is taken and that the moments still match the reference, on a bounded
# box with ν = 1.5 and on a one-sided box with ν = 2.5.
@testset "Student-t states without a helper (low degrees of freedom)" begin
    function n_direct(s, seen = IdDict())
        haskey(seen, s) && return 0
        seen[s] = true
        c = (s.direct && s.rawMomentsComputed) ? 1 : 0
        for ch in vcat(s.children_a, s.children_b)
            c += n_direct(ch, seen)
        end
        s.helper !== nothing && (c += n_direct(s.helper, seen))
        return c
    end
    prev = set_kr_base_backend!(:hcubature)
    try
        μ = [0.2, -0.1]
        Σ = [1.0 0.4; 0.4 1.0]
        for (ν, a, b) in ((1.5, [-1.0, -1.5], [2.0, 1.0]),
                          (2.5, [-1.0, -1.5], [2.0, Inf]))
            νr, Σr = ν + 2, ν / (ν + 2) .* Σ          # the gradient's root law
            d = TruncatedMvT(νr, μ, Σr, a, b; max_moment_levels = 4)
            @testset "ν=$ν, b=$b" begin
                for κ in _t_multi_indices(2, 4)
                    @test raw_moment(d, κ) ≈ _t_reference_moment(νr, μ, Σr, a, b, κ) rtol = 1e-6 atol = 1e-8
                end
                @test n_direct(d.state) > 0
            end
        end
    finally
        set_kr_base_backend!(prev)
    end
end
