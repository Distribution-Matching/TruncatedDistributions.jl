# n=6 Hybrid BCD only. Skips MVN+WS (infeasible single-shot at n=6 due to
# the KR tree size) and just times the hybrid path:
#   warm_start_diagonal → BCD with k∈{1,2,3} → joint-LBFGS polish.
#
# Uses marginal acceptance at n=6 so we don't pay the full-n KR cost on
# every BCD iteration's acceptance gate.
#
# Run: julia --color=no --project=. test/quick_bcd_n6.jl

using TruncatedDistributions
using Distributions, PDMats, LinearAlgebra, Printf
using Optim, Random

const POLISH_OPTS = Optim.Options(show_trace = false,
                                  iterations = 8,
                                  time_limit = 240.0,
                                  callback = s -> s.value < 1e-3)

# Compute (μ̂, Σ̂) targets by Monte Carlo: draw samples from the untruncated
# MvNormal, keep those inside the box, average. This is purely a way to
# produce reasonable targets for the BCD experiments without paying any
# n-dim KR cost up front. The downstream BCD/polish does not use the
# samples, only the resulting (μ̂, Σ̂).
#
# MC error budget at n_samples=200_000:
#   * SE on each μ̂_i  ≈ σ_post / √N ≲ 1 / √(2·10⁵) ≈ 2.2·10⁻³
#   * SE on each Σ̂_{ij} ≈ √(2/N) σ² ≈ 3.2·10⁻³
#   * Rounding to 1 decimal digit snaps each target to a 0.05-wide cell,
#     well outside the MC SE. The rounded targets are therefore
#     deterministic — the BCD is chasing a fixed target, not noise.
#   * Even if we did NOT round: cumulative L contribution from MC error at
#     n=8 with 44 moment entries is ≈ 44 · (3·10⁻³)² ≈ 4·10⁻⁴, just below
#     our 10⁻³ convergence threshold. Rounding gives us additional slack.
function target(ne; n_samples::Int = 200_000, digits = 1,
                seed::Int = 42)
    und_d = MvNormal(collect(ne.μ), Matrix(ne.Σ))
    a = collect(ne.a); b = collect(ne.b)
    rng = Random.MersenneTwister(seed)
    n = length(ne.μ)
    accepted_count = 0
    sum1 = zeros(n)
    sum2 = zeros(n, n)
    # Block-sample to avoid materialising n_samples * n bytes per attempt.
    block = 10_000
    attempts = 0
    while accepted_count < n_samples && attempts < 50 * n_samples
        X = rand(rng, und_d, block)
        for k in 1:block
            attempts += 1
            x = @view X[:, k]
            inside = true
            for i in 1:n
                (x[i] < a[i] || x[i] > b[i]) && (inside = false; break)
            end
            if inside
                sum1 .+= x
                sum2 .+= x * x'
                accepted_count += 1
                accepted_count >= n_samples && break
            end
        end
    end
    μ = sum1 ./ accepted_count
    Σ = sum2 ./ accepted_count .- μ * μ'
    # Symmetrise; round to requested digits for clean targets.
    Σ = 0.5 .* (Σ .+ Σ')
    set_kr_base_backend!(:mvnormalcdf)
    # Return a lightweight stand-in carrying only the box bounds (which
    # the caller needs). Constructing a full RecursiveMomentsBoxTruncatedMvNormal
    # at n≥8 would allocate the entire O(n!) KR tree just to read out
    # `d.region.a` / `d.region.b` — wasted work since we don't query
    # any moments from this distribution.
    d_lightweight = (region = (a = collect(ne.a), b = collect(ne.b)),)
    return d_lightweight, round.(μ; digits = digits), round.(Σ; digits = digits)
end

function polish(p0, a, b, μ̂, Σ̂)
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Matrix(Σ̂))
    optimize(Optim.only_fg!(fg!), p0, LBFGS(), POLISH_OPTS)
end

function warmup()
    println("# warmup: n=2..."); flush(stdout)
    ne = get_example(n = 2, index = 4)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    set_kr_base_backend!(:mvnormalcdf)
    try; block_coord_descent(μ̂, Σ̂, a, b;
                              block_sizes = [1, 2],
                              max_iters = 2, inner_iters = 3,
                              accept_by = :marginal, verbose = false); catch; end
    try; warm_start_diagonal(μ̂, Σ̂, a, b); catch; end
    println("# warmup: n=2 done"); flush(stdout)

    println("# JIT warmup at n=6..."); flush(stdout)
    ne6 = get_example(n = 6, index = 1)
    d6, μ̂6, Σ̂6 = target(ne6)
    a6 = collect(d6.region.a); b6 = collect(d6.region.b)
    p06 = make_param_vec_from_μ_Σ(μ̂6, Σ̂6)
    G6  = similar(p06)
    println("#   computing one gradient call..."); flush(stdout)
    t1 = @elapsed vector_fg_true_loss(p -> nothing, G6, p06, a6, b6, μ̂6, Matrix(Σ̂6))
    @printf("#   first gradient eval %.2fs\n", t1); flush(stdout)
    println("# warmup done"); flush(stdout)
end

function run_n6(label, ne)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    n = length(μ̂)
    set_kr_base_backend!(:mvnormalcdf)

    println(); println("====== $label  (n=$n)  ======")
    flush(stdout)

    # At n>=7 a single full-dim KR gradient call costs ~85s (n=7) or many
    # minutes (n=8), so we run *no* polish step at those sizes — pure
    # warm-start + BCD with k≤3 sub-problems. We report the BCD's
    # marginal-loss endpoint as the algorithm output.
    do_polish = n <= 6

    local L_out, n_it, t_polish
    t = @elapsed begin
        println("  ws..."); flush(stdout)
        t_ws = @elapsed (μ_w, Σ_w) = warm_start_diagonal(μ̂, Σ̂, a, b)
        @printf("  ws done %.2fs\n", t_ws); flush(stdout)

        println("  BCD..."); flush(stdout)
        t_bcd = @elapsed (μ_b, Σ_b, hist, picks) = block_coord_descent(
            μ̂, Σ̂, a, b;
            μ_init = μ_w, Σ_init = Σ_w,
            block_sizes = [1, 2, 3],
            max_iters = 15,
            inner_iters = 10,
            ftarget = 1e-3,
            monitor_full_loss = false,    # don't pay full-n KR per iter
            accept_by = :marginal,
            selection = :softmax,
            softmax_T = 1.0,
            verbose = true)
        L_bcd = hist[end]; n_it = length(hist) - 1
        @printf("  BCD done %.2fs  marg-loss-end=%.2e  iters=%d\n",
                t_bcd, L_bcd, n_it); flush(stdout)
        # Per-block summary: how many picks went to k=1, 2, 3, and how many
        # of each were accepted.
        for k in 1:3
            picks_k = filter(p -> p[1] == k, picks)
            isempty(picks_k) && continue
            acc = count(p -> p[3], picks_k)
            @printf("    k=%d picks: %d (%d acc, %d rej)\n",
                    k, length(picks_k), acc, length(picks_k) - acc)
        end
        flush(stdout)

        if do_polish
            println("  polish..."); flush(stdout)
            p_b = make_param_vec_from_μ_Σ(μ_b, Σ_b)
            t_polish = @elapsed (r_p = polish(p_b, a, b, μ̂, Σ̂))
            L_out = r_p.minimum
            @printf("  polish done %.2fs  L=%.2e\n", t_polish, L_out); flush(stdout)
        else
            t_polish = 0.0
            L_out = L_bcd      # report marg-loss endpoint at n>=7
            println("  (no polish at n>=7; reporting marg-loss endpoint)"); flush(stdout)
        end
    end

    @printf("[%-12s] %s %.2fs  BCD-iters=%d  L=%.2e\n",
            label, do_polish ? "HYBRID" : "BCD-only", t, n_it, L_out)
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# Hybrid BCD at n=6, 7, 8, 9, 10 (no MVN+WS baseline)")
    warmup()
    for n in [6, 7, 8, 9, 10]
        for i in 1:get_num_examples(n)
            run_n6("n=$n idx=$i", get_example(n = n, index = i))
        end
    end
end
