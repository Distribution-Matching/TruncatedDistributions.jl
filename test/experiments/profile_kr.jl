"""
Profile a single Kan–Robotti moment recursion at n=3 to find the
hot spot. Three measurements:

  1. @time on one dist construction + a few moment lookups (allocations).
  2. @benchmark on the same to amortise JIT (timing).
  3. Profile.print to attribute time to call sites.

Run from the package root:
    julia --project=. test/profile_kr.jl
"""

using TruncatedDistributions, PDMats, LinearAlgebra
using Profile

ne   = get_example(n = 3, index = 1)
μ    = collect(ne.μ); Σ = PDMat(Matrix(ne.Σ))
a    = collect(ne.a); b = collect(ne.b)
μ̂    = round.(mean(dist_from_example(ne)); digits = 1)
Σ̂    = Matrix(round.(cov(dist_from_example(ne));  digits = 1))

# Warm-up JIT
let
    d = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 4)
    moment_loss(d, μ̂, Σ̂)
    grad_true_loss(d, μ̂, Σ̂)
end

# -----------------------------------------------------------------
# (1) @time: one dist + (f, g)
# -----------------------------------------------------------------
println("\n--- (1) Single dist + moment_loss + grad_true_loss ---")
@time begin
    d = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 4)
    moment_loss(d, μ̂, Σ̂)
    grad_true_loss(d, μ̂, Σ̂)
end

# -----------------------------------------------------------------
# (2) Loop ×20 to amortise allocations from one-shot
# -----------------------------------------------------------------
println("\n--- (2) 20× repetitions of the same thing ---")
@time for _ in 1:20
    d = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 4)
    moment_loss(d, μ̂, Σ̂)
    grad_true_loss(d, μ̂, Σ̂)
end

# -----------------------------------------------------------------
# (3) Profile: ~50 iterations, then print top frames by self-time
# -----------------------------------------------------------------
println("\n--- (3) Profile ---")
Profile.init(n = 10^7, delay = 0.001)
Profile.clear()
@profile begin
    for _ in 1:50
        d = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 4)
        moment_loss(d, μ̂, Σ̂)
        grad_true_loss(d, μ̂, Σ̂)
    end
end

# Aggregate by function (no recursion), top 25 by self-time.
Profile.print(IOContext(stdout, :displaysize => (200, 200));
              mincount = 5,
              format = :flat,
              sortedby = :count,
              C = false)
