"""
Benchmark every available moment-correction solver on the bundled MV normal
examples (n = 2, 3, 4). For each (example, method) we record:

  * `time`      — wall-clock seconds for one solve
  * `final`     — moment_loss(d_fit, μ̂, Σ̂) at the returned point
  * `μ_err`     — ‖mean(d_fit) − μ̂‖
  * `Σ_err`     — ‖cov(d_fit)  − Σ̂‖
  * `ok`        — solver did not throw

Solvers compared:

  full_gradient        – `loss_based_fit` (explicit surrogate gradient, GD)
  pair_gradient        – pair-coordinate descent on top of `loss_based_fit`
  optim_lbfgs_fd       – `Optim.LBFGS` with finite-difference gradient on L
  optim_lbfgs_surr     – `Optim.LBFGS` with the explicit surrogate gradient
                         on L̃ (μA frozen at μ̂)
  optim_lbfgs_true     – `Optim.LBFGS` with the explicit true-loss gradient
                         on L (μA recomputed via chain rule)
  prima_newuoa         – derivative-free trust-region (NEWUOA)

Run from the package root:

    julia --project=. test/benchmark_solvers.jl
"""

using TruncatedDistributions
using Distributions
using PDMats
using LinearAlgebra
using Printf

const METHODS = (
    full_gradient    = correct_to_moments_with_full_gradient,
    pair_gradient    = correct_to_moments_with_pair_gradient_descent,
    optim_lbfgs_fd   = correct_to_moments_with_optim,
    optim_lbfgs_surr = correct_to_moments_with_optim_surrogate_grad,
    optim_lbfgs_true = correct_to_moments_with_optim_explicit_grad,
    prima_newuoa     = correct_to_moments_with_prima,
)

function run_method(fit_fn, d, μ̂, Σ̂)
    try
        local d_fit
        elapsed = @elapsed begin
            d_fit = fit_fn(d, μ̂, Σ̂)
        end
        return (time   = elapsed,
                final  = moment_loss(d_fit, μ̂, Σ̂),
                μ_err  = norm(mean(d_fit) - μ̂),
                Σ_err  = norm(cov(d_fit)  - Σ̂),
                ok     = true)
    catch e
        @warn "method failed" exception = (e, catch_backtrace())
        return (time = NaN, final = NaN, μ_err = NaN, Σ_err = NaN, ok = false)
    end
end

# Build a near-by target by rounding the truncated moments of the bundled
# example. This guarantees feasibility while keeping the target slightly
# off the underlying (μ, Σ).
function target_from_example(ne; digits::Int = 3)
    d  = dist_from_example(ne)
    μ̂  = round.(mean(d); digits = digits)
    Σ̂  = round.(cov(d);  digits = digits)
    return d, μ̂, Σ̂
end

function sweep(; digits::Int = 3, sizes = [2, 3, 4],
                 methods = METHODS, skip_slow = String[])
    rows = NamedTuple[]
    for n in sizes
        get_num_examples(n) == 0 && continue
        for i in 1:get_num_examples(n)
            ne = get_example(n = n, index = i)
            d, μ̂, Σ̂ = target_from_example(ne; digits = digits)
            initial = moment_loss(d, μ̂, Σ̂)
            println("\n[example] n=$n index=$i tp=$(round(tp(d), digits=4)) initial=$(round(initial, digits=6))")
            flush(stdout)
            for (name, fn) in pairs(methods)
                if string(name) in skip_slow
                    println("  $(rpad(string(name), 17))  SKIPPED")
                    flush(stdout)
                    push!(rows, (n = n, index = i, method = name, initial = initial,
                                 time = NaN, final = NaN, μ_err = NaN, Σ_err = NaN, ok = false))
                    continue
                end
                r = run_method(fn, d, μ̂, Σ̂)
                push!(rows, (n = n, index = i, method = name, initial = initial, r...))
                println("  $(rpad(string(name), 17))  time=$(round(r.time, digits=3))s  final=$(round(r.final, digits=6))  ok=$(r.ok)")
                flush(stdout)
            end
        end
    end
    return rows
end

function print_table(rows)
    println()
    @printf("%4s %5s %-18s %10s %14s %10s %10s %5s\n",
            "n", "idx", "method", "time(s)", "final_loss", "‖μ_err‖", "‖Σ_err‖", "ok")
    println(repeat("-", 88))
    for r in rows
        @printf("%4d %5d %-18s %10.4f %14.3e %10.3e %10.3e %5s\n",
                r.n, r.index, string(r.method), r.time, r.final, r.μ_err, r.Σ_err, r.ok)
    end
end

# Pretty-print a LaTeX tabular row suitable for pasting into the paper.
function print_latex(rows; methods = collect(keys(METHODS)))
    by_case = Dict{Tuple{Int,Int},Dict{Symbol,NamedTuple}}()
    for r in rows
        d = get!(by_case, (r.n, r.index), Dict{Symbol,NamedTuple}())
        d[r.method] = r
    end
    println("\n% LaTeX rows (n, idx, method, time, final_loss)")
    for ((n, idx), d) in sort(collect(by_case))
        for m in methods
            haskey(d, m) || continue
            r = d[m]
            time_s  = r.ok ? @sprintf("%.3f", r.time)  : "---"
            loss_s  = r.ok ? @sprintf("%.2e", r.final) : "---"
            @printf("  %d & %d & %s & %s & %s \\\\\n",
                    n, idx, replace(string(m), "_" => "\\_"), time_s, loss_s)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Rounding the truncated moments to 1 decimal digit gives a near-by but
    # non-trivial moment-matching target. Earlier (digits=3) the rounded
    # target was within 1e-7 of the truncated moments, so most solvers
    # terminated at iteration 0 and wall-clock was dominated by JIT compile.
    # full_gradient and pair_gradient are skipped: they're robust on these
    # cases but minutes-per-call. The headline comparison is FD-LBFGS vs
    # explicit-grad LBFGS vs derivative-free vs explicit-grad-on-surrogate.
    rows = sweep(; digits = 1, skip_slow = ["full_gradient", "pair_gradient"])
    print_table(rows)
    print_latex(rows)
end
