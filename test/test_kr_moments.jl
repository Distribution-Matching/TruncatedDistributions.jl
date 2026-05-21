# Numerical correctness of the Kan–Robotti recursion.
#
# For every bundled example we evaluate every primitive moment
#
#   m^{(p)}_{i_1,…,i_p}  =  ∫_A x_{i_1}…x_{i_p} f(x) dx
#
# with p running up to max_moment_levels via the KR recursion, and compare
# against an independent HCubature integration of the same integrand. The
# two should agree to ~1e-6 absolute on every entry.
#
# This is the strong cross-check on the recursion: HCubature has nothing in
# common with KR's code path (no recursion, no per-axis conditioning, no
# precomputed pdf at the box face), so an agreement here narrows the space
# of bugs that could still be lurking in the recursion.

using LinearAlgebra
using PDMats
import TruncatedDistributions: hcubature_inf

# Enumerate every multi-index (kappa) with sum(kappa) ≤ L and length(kappa) = n,
# returned as Vector{Vector{Int}}.
function _all_multi_indices(n::Int, L::Int)
    out = Vector{Vector{Int}}()
    function rec(prefix, remaining)
        if length(prefix) == n
            push!(out, copy(prefix))
            return
        end
        for v in 0:remaining
            push!(prefix, v)
            rec(prefix, remaining - v)
            pop!(prefix)
        end
    end
    rec(Int[], L)
    return out
end

# Reference: m^{(κ)} via direct HCubature on x_1^{κ_1} … x_n^{κ_n} · f(x).
function _hcubature_moment(μ, Σ, a, b, κ)
    n     = length(μ)
    Σinv  = inv(Σ)
    detΣ  = det(Σ)
    nc    = (2π)^(-n/2) / sqrt(detΣ)
    f(x)  = nc * exp(-0.5 * dot(x .- μ, Σinv * (x .- μ)))
    monomial(x) = prod(x[i]^κ[i] for i in 1:n)
    return hcubature_inf(x -> monomial(x) * f(x), a, b; rtol = 1e-8, maxevals = 10^6)[1]
end

@testset "Kan–Robotti raw moments vs HCubature reference" begin
    # Run the cross-check with each KR base-case backend in turn so a
    # regression in either path is caught. The hcubature backend is the
    # default and the historical one; the mvnormalcdf backend was added
    # as a faster alternative for the Float64 LBFGS path.
    for backend in (:hcubature, :mvnormalcdf)
        prev = set_kr_base_backend!(backend)
        try
            @testset "KR backend = $backend" begin
                for n in [2, 3]
                    get_num_examples(n) == 0 && continue
                    for i in 1:get_num_examples(n)
                        ne = get_example(n = n, index = i)
                        μ  = collect(ne.μ); Σ = PDMat(Matrix(ne.Σ))
                        a  = collect(ne.a); b = collect(ne.b)
                        L  = 4
                        d  = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b; max_moment_levels = L)

                        @testset "n=$n idx=$i" begin
                            for κ in _all_multi_indices(n, L)
                                m_kr  = raw_moment(d, κ)
                                m_ref = _hcubature_moment(collect(ne.μ), Matrix(ne.Σ), a, b, κ)
                                # mvnormalcdf is randomised QMC; with m = 10_000 samples
                                # its per-call error is ~1e-5 but compounds modestly
                                # through the recursion, so a slightly looser tolerance
                                # is appropriate.
                                tol = backend === :mvnormalcdf ? 5e-4 : 1e-5
                                @test m_kr ≈ m_ref atol = 1e-5 rtol = tol
                            end
                        end
                    end
                end
            end
        finally
            set_kr_base_backend!(prev)
        end
    end
end
