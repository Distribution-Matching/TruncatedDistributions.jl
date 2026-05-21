# Verify update_distribution! produces identical moments to a fresh constructor,
# then measure the cost saving across N back-to-back LBFGS-style evaluations.
#
# Run: julia --project=. test/quick_update_dist.jl

using TruncatedDistributions
using Distributions, PDMats, LinearAlgebra, Printf, Random

function test_correctness(n::Int; max_levels::Int = 4)
    Σ1 = Matrix{Float64}(I, n, n) + 0.2 * ones(n, n)
    Σ2 = Matrix{Float64}(I, n, n) + 0.4 * ones(n, n)
    μ1 = zeros(n); μ2 = 0.1 .* ones(n)
    a = fill(-1.5, n); b = fill(1.5, n)
    set_kr_base_backend!(:mvnormalcdf)

    d_fresh = RecursiveMomentsBoxTruncatedMvNormal(μ2, PDMat(Σ2), a, b;
                                                    max_moment_levels = max_levels)
    compute_moments(d_fresh.state)

    # Build with (μ1, Σ1), then refresh to (μ2, Σ2)
    d_reuse = RecursiveMomentsBoxTruncatedMvNormal(μ1, PDMat(Σ1), a, b;
                                                    max_moment_levels = max_levels)
    update_distribution!(d_reuse.state, μ2, PDMat(Σ2))
    compute_moments(d_reuse.state)

    # Compare moments
    max_diff = 0.0
    for k in keys(d_fresh.state.rawMomentDict)
        v1 = d_fresh.state.rawMomentDict[k]
        v2 = d_reuse.state.rawMomentDict[k]
        if isfinite(v1) && isfinite(v2)
            max_diff = max(max_diff, abs(v1 - v2))
        end
    end
    @printf("n=%d  max |moment_fresh - moment_reuse| = %.3e\n", n, max_diff)
end

function bench_reuse(n::Int; iters::Int = 5, max_levels::Int = 4)
    Σ0 = Matrix{Float64}(I, n, n) + 0.2 * ones(n, n)
    μ0 = zeros(n)
    a = fill(-1.5, n); b = fill(1.5, n)
    set_kr_base_backend!(:mvnormalcdf)
    rng = MersenneTwister(42)

    # Warmup
    d_warm = RecursiveMomentsBoxTruncatedMvNormal(μ0, PDMat(Σ0), a, b;
                                                   max_moment_levels = max_levels)
    compute_moments(d_warm.state)
    update_distribution!(d_warm.state, μ0, PDMat(Σ0))
    compute_moments(d_warm.state)

    # Fresh-construct path: iters back-to-back constructions
    t_fresh = @elapsed begin
        for _ in 1:iters
            μ = randn(rng, n); Σ = Σ0 + 0.05 * randn(rng) * Matrix{Float64}(I, n, n)
            df = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(0.5*(Σ+Σ')), a, b;
                                                       max_moment_levels = max_levels)
            compute_moments(df.state)
        end
    end

    # Reuse path: one construction, then iters refreshes
    rng2 = MersenneTwister(42)
    dr = RecursiveMomentsBoxTruncatedMvNormal(μ0, PDMat(Σ0), a, b;
                                               max_moment_levels = max_levels)
    t_reuse = @elapsed begin
        for _ in 1:iters
            μ = randn(rng2, n); Σ = Σ0 + 0.05 * randn(rng2) * Matrix{Float64}(I, n, n)
            update_distribution!(dr.state, μ, PDMat(0.5*(Σ+Σ')))
            compute_moments(dr.state)
        end
    end

    @printf("n=%d  iters=%d  fresh: %.2fms  reuse: %.2fms  speedup: %.2fx\n",
            n, iters, t_fresh*1e3, t_reuse*1e3, t_fresh / t_reuse)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# Correctness: update_distribution! vs fresh construction")
    for n in [2, 3, 4]
        test_correctness(n)
    end
    println()
    println("# Benchmark: 30 back-to-back evaluations, fresh vs reuse")
    for n in [2, 3, 4, 5]
        bench_reuse(n; iters = 30)
    end
    println()
    println("# Benchmark: 5 back-to-back evaluations at n=6 (slow per-call)")
    bench_reuse(6; iters = 5)
end
