# Measure the per-call cost of rebuilding the KR tree state.
# Compares (construction-only) vs (one full moment compute) at n=3,4,5.
#
# Run: julia --project=. test/quick_alloc_check.jl

using TruncatedDistributions
using Distributions, PDMats, LinearAlgebra, Printf

function bench_n(n::Int; max_levels::Int = 4)
    # Simple PD covariance and box
    Σ = Matrix{Float64}(I, n, n) + 0.2 * ones(n, n)
    μ = zeros(n)
    a = fill(-1.5, n); b = fill( 1.5, n)
    set_kr_base_backend!(:mvnormalcdf)

    # Warmup
    d = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ), a, b; max_moment_levels = max_levels)
    compute_moments(d.state)

    # Time JUST construction
    t_ctor = @elapsed for _ in 1:5
        RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ), a, b; max_moment_levels = max_levels)
    end
    t_ctor /= 5

    # Time construction + full moment compute (= one full vector_fg_true_loss base cost)
    t_full = @elapsed for _ in 1:5
        d2 = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ), a, b; max_moment_levels = max_levels)
        compute_moments(d2.state)
    end
    t_full /= 5

    # Allocations
    alloc_ctor = @allocated RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ), a, b; max_moment_levels = max_levels)
    alloc_full = @allocated begin
        d3 = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ), a, b; max_moment_levels = max_levels)
        compute_moments(d3.state)
    end

    @printf("n=%d  ctor: %.2fms %s  |  ctor+moments: %.2fms %s  |  ctor/total: %.0f%%\n",
            n, t_ctor*1e3, _fmt_bytes(alloc_ctor),
            t_full*1e3, _fmt_bytes(alloc_full),
            100 * t_ctor / t_full)
end

_fmt_bytes(b) = b < 1024 ? "$(b)B" :
                b < 1024^2 ? @sprintf("%.1fKB", b/1024) :
                @sprintf("%.1fMB", b/1024^2)

if abspath(PROGRAM_FILE) == @__FILE__
    println("# KR-tree construction vs full moment compute (with mvnormalcdf base)")
    for n in [2, 3, 4, 5, 6]
        try; bench_n(n); catch e; @printf("n=%d  FAIL: %s\n", n, sprint(showerror, e)); end
    end
end
