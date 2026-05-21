"""
Experiment: numerical verification of the true-loss gradient.

Compares two computations of ∇L on a handful of small test cases:
  - analytic ∇L via grad_true_loss (chain rule on primitive moments,
    using the Gaussian score-function identity for ∂m^(p)/∂(μ,U));
  - central finite differences on moment_loss(d, μ̂, Σ̂).

This exists only to confirm that the loss function L and its gradient
agree numerically. It does no optimization.

Run from the package root with:
    julia --project=. test/experiment_true_loss_gradient.jl
"""

using TruncatedDistributions
using Distributions, PDMats, LinearAlgebra
using Printf

# ----- helpers -------------------------------------------------------------

build_dist(p, a, b) = let
    μ, Σ = make_μ_Σ_from_param_vec(p)
    RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ), a, b; max_moment_levels = 4)
end

function fd_grad(f::Function, p::Vector{Float64}; h = 1e-6)
    g = zeros(length(p))
    for i in eachindex(p)
        p1 = copy(p); p1[i] += h
        p2 = copy(p); p2[i] -= h
        g[i] = (f(p1) - f(p2)) / (2h)
    end
    g
end

function run_one(label, μ, Σ, a, b, μ̂, Σ̂)
    n = length(μ)
    U = Matrix(cholesky(0.5 * (inv(Σ) + inv(Σ)')).U)
    p0 = make_param_vec_from_μ_U(μ, U)

    d = build_dist(p0, a, b)
    L = moment_loss(d, μ̂, Σ̂)

    g_an = vector_grad_true_loss(p0, a, b, μ̂, Σ̂)
    g_fd = fd_grad(p -> moment_loss(build_dist(p, a, b), μ̂, Σ̂), p0; h = 1e-6)

    abs_err = norm(g_an - g_fd)
    rel_err = abs_err / max(norm(g_fd), eps())

    return (label = label, n = n, L = L,
            g_an = g_an, g_fd = g_fd,
            abs_err = abs_err, rel_err = rel_err)
end

# ----- test cases ---------------------------------------------------------

function main()
    rows = []

    # 2D, off-target so L > 0 and the gradient is non-trivial
    push!(rows, run_one("2D off-target",
        [0.0, 0.0],
        [1.0 0.3; 0.3 1.0],
        [-1.0, -1.5], [1.5, 1.0],
        [0.20, -0.15],
        [0.50 0.10; 0.10 0.45]))

    # 2D bounded box (Example 4): μ̂, Σ̂ are exactly the truncated moments,
    # so L = 0 and ∇L = 0; this checks numerical noise floor.
    let ne = get_example(n = 2, index = 4)
        push!(rows, run_one("2D on-target (ex. 4)",
            collect(ne.μ), Matrix(ne.Σ),
            collect(ne.a), collect(ne.b),
            collect(ne.μ̂), Matrix(ne.Σ̂)))
    end

    # 2D off-target, larger mismatch
    push!(rows, run_one("2D big mismatch",
        [0.3, -0.2],
        [1.0 0.5; 0.5 1.5],
        [-2.0, -2.0], [2.0, 2.5],
        [0.0, 0.0],
        [1.5 0.0; 0.0 1.0]))

    # 3D off-target
    push!(rows, run_one("3D off-target",
        [0.0, 0.0, 0.0],
        [1.0 0.3 0.0; 0.3 1.0 0.2; 0.0 0.2 1.0],
        [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5],
        [0.1, -0.1, 0.05],
        [0.5 0.05 0.0; 0.05 0.45 0.05; 0.0 0.05 0.55]))

    println()
    @printf("%-22s %3s %12s %14s %14s %14s\n",
            "case", "n", "L", "‖∇L_an‖", "‖∇L_an − ∇L_FD‖", "rel. error")
    println(repeat("-", 90))
    for r in rows
        @printf("%-22s %3d %12.4e %14.4e %14.2e %14.2e\n",
                r.label, r.n, r.L, norm(r.g_an), r.abs_err, r.rel_err)
    end

    println("\nDetail (case 1, 2D off-target): analytic vs FD components")
    r = rows[1]
    @printf("%4s %16s %16s %12s\n", "i", "∇L_an", "∇L_FD", "|diff|")
    for i in eachindex(r.g_an)
        @printf("%4d %16.6e %16.6e %12.2e\n",
                i, r.g_an[i], r.g_fd[i], abs(r.g_an[i] - r.g_fd[i]))
    end

    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
