# Smoke test for block_coord_descent. Runs n=3, then n=4 — fast feedback loop.
# Compares wall-clock and final loss against the joint-LBFGS baseline.
#
# Run:
#     julia --project=. test/quick_bcd_smoke.jl

using TruncatedDistributions
using Distributions, PDMats, LinearAlgebra, Printf
using Optim

const OPTS = Optim.Options(show_trace = false,
                           iterations = 50,
                           time_limit = 60.0,
                           callback   = s -> s.value < 1e-3)

function target(ne; digits::Int = 1)
    d  = dist_from_example(ne)
    μ̂  = round.(mean(d); digits = digits)
    Σ̂  = round.(cov(d);  digits = digits)
    return d, μ̂, Σ̂
end

function joint_lbfgs(p0, a, b, μ̂, Σ̂)
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Matrix(Σ̂))
    optimize(Optim.only_fg!(fg!), p0, LBFGS(), OPTS)
end

function run_one(label, ne; iters::Int = 25, inner::Int = 15,
                 block_sizes = [1, 2, 3])
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)

    set_kr_base_backend!(:mvnormalcdf)

    # Baseline: joint LBFGS from (μ̂, Σ̂)
    p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
    t_joint = @elapsed (r_joint = joint_lbfgs(p0, a, b, μ̂, Σ̂))

    # Hybrid: warm-start  →  BCD (mixed k)  →  short joint-LBFGS polish
    μ_ws, Σ_ws = warm_start_diagonal(μ̂, Σ̂, a, b)
    local r_hyb, L_bcd, n_it
    t_hyb = @elapsed begin
        μ_b, Σ_b, hist = block_coord_descent(
            μ̂, Σ̂, a, b;
            μ_init = μ_ws, Σ_init = Σ_ws,
            block_sizes = block_sizes,
            max_iters = iters,
            inner_iters = inner,
            ftarget = 1e-3,
            exclude_recent = 0,
            accept_by = :full,
            verbose = true)
        L_bcd = hist[end]; n_it = length(hist) - 1
        if L_bcd < 1e-3
            r_hyb = (minimum = L_bcd,)
        else
            p_b = make_param_vec_from_μ_Σ(μ_b, Σ_b)
            r_hyb = joint_lbfgs(p_b, a, b, μ̂, Σ̂)
        end
    end

    @printf("[%-18s] joint %.2fs(%.2e)  hybrid %.2fs(BCD %.2e → %.2e, iters=%d)\n",
            label, t_joint, r_joint.minimum,
            t_hyb, L_bcd, r_hyb.minimum, n_it)
    return (t_joint, r_joint.minimum, t_hyb, r_hyb.minimum, L_bcd, n_it)
end

function warmup()
    ne = get_example(n = 2, index = 4)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    set_kr_base_backend!(:mvnormalcdf)
    try; joint_lbfgs(make_param_vec_from_μ_Σ(μ̂, Σ̂), a, b, μ̂, Σ̂); catch; end
    try; block_coord_descent(μ̂, Σ̂, a, b;
                              block_sizes = [1, 2],
                              max_iters = 2, inner_iters = 3,
                              ftarget = 0.0, verbose = false); catch; end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# BCD smoke test — warmup pass first")
    warmup()
    println("# Warmup complete")
    println()

    # n=3: both indices
    for i in 1:get_num_examples(3)
        run_one("n=3 idx=$i", get_example(n = 3, index = i))
    end
    # n=4: single bundled example
    for i in 1:get_num_examples(4)
        run_one("n=4 idx=$i", get_example(n = 4, index = i))
    end
end
