# Tightly-scoped probe: how long does ONE EXP-KR-MVN gradient evaluation
# take at n=6 with the mvnormalcdf backend? Just measure that single number
# so we can budget the full run.
#
# Run: julia --color=no --project=. test/quick_n6_probe.jl

using TruncatedDistributions
using Distributions, PDMats, LinearAlgebra, Printf
using Optim

println("# probing n=6 single gradient call cost"); flush(stdout)
set_kr_base_backend!(:mvnormalcdf)

n = 6
μ  = zeros(n)
Σ  = let M = Matrix{Float64}(I, n, n)
    for i in 1:(n-1); M[i, i+1] = 0.3; M[i+1, i] = 0.3; end
    M
end
a = fill(-2.0, n); b = fill(2.0, n)
μ̂ = zeros(n)
Σ̂ = copy(Σ)

println("# building dist..."); flush(stdout)
t_build = @elapsed (d = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ), a, b;
                                                              max_moment_levels = 4))
@printf("# build: %.2fs\n", t_build); flush(stdout)

println("# computing moments..."); flush(stdout)
t_moments = @elapsed compute_moments(d.state)
@printf("# moments: %.2fs\n", t_moments); flush(stdout)

println("# computing gradient via vector_fg_true_loss..."); flush(stdout)
p = make_param_vec_from_μ_Σ(μ, Σ)
G = similar(p)
t_grad = @elapsed (val = vector_fg_true_loss(val_p -> nothing, G, p, a, b, μ̂, Σ̂))
@printf("# 1 fg call: %.2fs, val=%.3e\n", t_grad, val); flush(stdout)

println("# running 3 fg calls back-to-back..."); flush(stdout)
t3 = @elapsed for _ in 1:3
    vector_fg_true_loss(val_p -> nothing, G, p, a, b, μ̂, Σ̂)
end
@printf("# 3 calls: %.2fs (mean %.2fs/call)\n", t3, t3/3); flush(stdout)
