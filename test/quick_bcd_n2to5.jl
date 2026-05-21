# Apples-to-apples MVN+WS vs Hybrid BCD on n=2,3,4,5 — the n<6 rows for the
# paper's §6.3 table. Matches §6.2's comparison style (joint LBFGS from
# warm-start) for the baseline, and runs the new k∈{1,2,3} hybrid with
# :full acceptance for the hybrid column.
#
# Run: julia --color=no --project=. test/quick_bcd_n2to5.jl

using TruncatedDistributions
using Distributions, PDMats, LinearAlgebra, Printf
using Optim

const OPTS = Optim.Options(show_trace = false, iterations = 50,
                           time_limit = 60.0,
                           callback = s -> s.value < 1e-3)

function target(ne; digits = 1)
    d = dist_from_example(ne)
    return d, round.(mean(d); digits = digits), round.(cov(d); digits = digits)
end

joint(p0, a, b, μ̂, Σ̂) = optimize(Optim.only_fg!(
        (F, G, p) -> vector_fg_true_loss(F, G, p, a, b, μ̂, Matrix(Σ̂))),
    p0, LBFGS(), OPTS)

function m0_of(ne)
    return mvnormcdf(MvNormal(collect(ne.μ), Matrix(ne.Σ)),
                     collect(ne.a), collect(ne.b); m = 10_000)[1]
end

function run_row(label, ne; iters = 25, inner = 15)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    set_kr_base_backend!(:mvnormalcdf)
    p_mass = m0_of(ne)

    # MVN+WS
    local L_ws, t_ws
    t_ws = @elapsed begin
        μ_w, Σ_w = warm_start_diagonal(μ̂, Σ̂, a, b)
        p_w      = make_param_vec_from_μ_Σ(μ_w, Σ_w)
        r_w      = joint(p_w, a, b, μ̂, Σ̂)
        L_ws = r_w.minimum
    end

    # Hybrid: WS  →  k∈{1,2,3} BCD (greedy, :full)  →  polish if needed
    local L_bcd, n_it, L_pol
    t_hy = @elapsed begin
        μ_w2, Σ_w2 = warm_start_diagonal(μ̂, Σ̂, a, b)
        μ_b, Σ_b, hist = block_coord_descent(
            μ̂, Σ̂, a, b;
            μ_init = μ_w2, Σ_init = Σ_w2,
            block_sizes = [1, 2, 3],
            max_iters = iters,
            inner_iters = inner,
            ftarget = 1e-3,
            monitor_full_loss = true,
            accept_by = :full,
            verbose = false)
        L_bcd = hist[end]; n_it = length(hist) - 1
        if L_bcd < 1e-3
            L_pol = L_bcd
        else
            p_b = make_param_vec_from_μ_Σ(μ_b, Σ_b)
            r_p = joint(p_b, a, b, μ̂, Σ̂)
            L_pol = r_p.minimum
        end
    end

    @printf("[%-15s] m0=%.4f  MVN+WS %.2fs(%.2e)  Hybrid %.2fs(%.2e, iters=%d)\n",
            label, p_mass, t_ws, L_ws, t_hy, L_pol, n_it)
    flush(stdout)
end

function warmup()
    ne = get_example(n = 2, index = 4)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    set_kr_base_backend!(:mvnormalcdf)
    try; joint(make_param_vec_from_μ_Σ(μ̂, Σ̂), a, b, μ̂, Σ̂); catch; end
    try; block_coord_descent(μ̂, Σ̂, a, b;
                              block_sizes = [1, 2],
                              max_iters = 2, inner_iters = 3,
                              accept_by = :full, verbose = false); catch; end
end

using MvNormalCDF

if abspath(PROGRAM_FILE) == @__FILE__
    println("# MVN+WS vs Hybrid BCD on n=2,3,4,5")
    println("# Warmup..."); flush(stdout)
    warmup()
    println("# Warmup done\n"); flush(stdout)

    for n in [2, 3, 4, 5]
        for i in 1:get_num_examples(n)
            run_row("n=$n idx=$i", get_example(n = n, index = i))
        end
    end
end
