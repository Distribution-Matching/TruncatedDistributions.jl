# High-n probe for hybrid BCD + joint-LBFGS polish.
# Skips the plain joint baseline at n>=6 (infeasible single-shot) and just
# reports hybrid wall-clock + per-sweep traces.
#
# Run:
#     julia --project=. test/quick_bcd_highn.jl

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

const POLISH_OPTS = Optim.Options(show_trace = false,
                                  iterations = 15,
                                  time_limit = 90.0,
                                  callback = s -> s.value < 1e-3)

function polish_lbfgs(p0, a, b, μ̂, Σ̂)
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Matrix(Σ̂))
    optimize(Optim.only_fg!(fg!), p0, LBFGS(), POLISH_OPTS)
end

function run_row(label, ne; iters::Int = 30, inner::Int = 15,
                 block_sizes = [1, 2, 3])
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    n = length(μ̂)
    set_kr_base_backend!(:mvnormalcdf)

    println()
    println("====== $label  (n=$n)  ======")
    flush(stdout)

    # n=6,7 strategy: skip the MVN+WS baseline (each n=6 KR gradient eval is
    # ~4s, so the 50-iter joint LBFGS easily blows the wall-clock budget and
    # often runs the system out of memory via repeated O(n!) tree
    # allocations). Use :marginal acceptance + softmax sampling for BCD so
    # we don't pay the full n-dim KR cost on every iteration's acceptance
    # gate; the final joint polish absorbs the marginal-fitting bias.
    use_full_acceptance = n <= 5
    use_softmax         = n >= 6

    # Baseline column.
    local L_ws, t_ws
    if n <= 5
        t_ws = @elapsed begin
            μ_ws, Σ_ws = warm_start_diagonal(μ̂, Σ̂, a, b)
            p_ws       = make_param_vec_from_μ_Σ(μ_ws, Σ_ws)
            r_ws       = joint_lbfgs(p_ws, a, b, μ̂, Σ̂)
            L_ws = r_ws.minimum
        end
        @printf("  [MVN+WS]  %.2fs  loss %.2e\n", t_ws, L_ws)
    else
        L_ws = NaN; t_ws = NaN
        println("  [MVN+WS]  skipped (n>5: joint LBFGS infeasible per §6.1)")
    end
    flush(stdout)

    # Hybrid: warm-start  →  BCD with k∈{1,2,3}  →  joint-LBFGS polish.
    local L_bcd, n_it, L_polish
    t = @elapsed begin
        μ_ws2, Σ_ws2 = warm_start_diagonal(μ̂, Σ̂, a, b)
        μ_b, Σ_b, hist = block_coord_descent(
            μ̂, Σ̂, a, b;
            μ_init = μ_ws2, Σ_init = Σ_ws2,
            block_sizes = block_sizes,
            max_iters = iters,
            inner_iters = inner,
            ftarget = 1e-3,
            monitor_full_loss = n <= 5,
            accept_by = use_full_acceptance ? :full : :marginal,
            selection = use_softmax ? :softmax : :greedy,
            softmax_T = 1.0,
            verbose = true)
        L_bcd = hist[end]; n_it = length(hist) - 1
        # At n>=6 the :marginal acceptance plateau is biased above the
        # true min, so always polish. Use a tighter cap for n>=6 so a
        # stuck polish does not run away on memory.
        run_polish = (L_bcd >= 1e-3) || (n >= 6)
        if run_polish
            p_b = make_param_vec_from_μ_Σ(μ_b, Σ_b)
            polish_fn = n <= 5 ? joint_lbfgs : polish_lbfgs
            r_hyb = polish_fn(p_b, a, b, μ̂, Σ̂)
            L_polish = r_hyb.minimum
        else
            L_polish = L_bcd
        end
    end
    @printf("  [Hybrid]  %.2fs  BCD %s %.2e (iters=%d) -> polish %.2e\n",
            t, n <= 5 ? "plateau" : "marg-end", L_bcd, n_it, L_polish)
    flush(stdout); flush(stderr)
end

function warmup()
    println("# warmup: n=2 example..."); flush(stdout)
    ne = get_example(n = 2, index = 4)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    set_kr_base_backend!(:mvnormalcdf)
    try; joint_lbfgs(make_param_vec_from_μ_Σ(μ̂, Σ̂), a, b, μ̂, Σ̂); catch; end
    try; block_coord_descent(μ̂, Σ̂, a, b;
                              block_sizes = [1, 2],
                              max_iters = 2, inner_iters = 3,
                              accept_by = :full, verbose = false); catch; end
    println("# warmup: n=2 done; pre-compiling at n=6..."); flush(stdout)
    # Warm n=6 specializations so the actual experiment isn't paying JIT
    # cost when it should be paying numerical cost.
    ne6 = get_example(n = 6, index = 1)
    d6, μ̂6, Σ̂6 = target(ne6)
    a6 = collect(d6.region.a); b6 = collect(d6.region.b)
    p06 = make_param_vec_from_μ_Σ(μ̂6, Σ̂6)
    G6  = similar(p06)
    try; vector_fg_true_loss(val_p -> nothing, G6, p06, a6, b6, μ̂6, Matrix(Σ̂6)); catch; end
    try; warm_start_diagonal(μ̂6, Σ̂6, a6, b6); catch; end
    println("# warmup: n=6 done"); flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# Hybrid BCD high-n probe — warmup first")
    warmup()
    println("# Warmup complete")

    # n=5 was already characterised on a previous run; here we focus the
    # budget on n=6 and n=7, where the strategy actually changes.
    for n in [6, 7]
        for i in 1:get_num_examples(n)
            iters = n >= 6 ? 15 : 30
            run_row("n=$n idx=$i", get_example(n = n, index = i); iters = iters)
        end
    end
end
