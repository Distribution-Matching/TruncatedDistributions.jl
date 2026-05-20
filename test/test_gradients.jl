# Correctness of the analytic derivative expressions used by the
# moment-matching algorithm. Each gradient is checked against central finite
# differences of the corresponding scalar loss it claims to differentiate:
#
#   ∇_{(μ,U)} L̃(·; μA, μ̂, Σ̂)   vs FD of approximate_vector_moment_loss
#       — the surrogate gradient from §3-§4 of the paper, with μA frozen.
#
#   ∇_{(μ,U)} L(μ,U; μ̂, Σ̂)     vs FD of vector_moment_loss (built on the
#       true moment_loss, μA = mean(d) recomputed at every parameter
#       perturbation).
#
# Both checks are run on three small but qualitatively distinct cases.

using LinearAlgebra
using PDMats

# Central finite-difference gradient of a scalar function of a parameter
# vector. Step size h = 1e-6 keeps truncation and round-off balanced for
# the moment integrals we evaluate here.
function _fd_grad(f, p::Vector{Float64}; h::Float64 = 1e-6)
    g = zeros(length(p))
    for i in eachindex(p)
        p_plus  = copy(p); p_plus[i]  += h
        p_minus = copy(p); p_minus[i] -= h
        g[i] = (f(p_plus) - f(p_minus)) / (2h)
    end
    return g
end

# Build the (μ, U) parameter vector at a Σ that is *not* exactly the target,
# so the gradient is non-trivial.
function _grad_test_setup(μ, Σ, a, b)
    U = Matrix(cholesky(0.5 * (inv(Σ) + inv(Σ)')).U)
    p = make_param_vec_from_μ_U(μ, U)
    return p
end

const GRAD_CASES = [
    (
        label = "2D bounded box, off-target",
        μ  = [0.0, 0.0],
        Σ  = [1.0 0.3; 0.3 1.0],
        a  = [-1.0, -1.5],
        b  = [ 1.5,  1.0],
        μ̂  = [0.20, -0.15],
        Σ̂  = [0.50 0.10; 0.10 0.45],
    ),
    (
        label = "2D large mismatch",
        μ  = [0.3, -0.2],
        Σ  = [1.0 0.5; 0.5 1.5],
        a  = [-2.0, -2.0],
        b  = [ 2.0,  2.5],
        μ̂  = [0.0, 0.0],
        Σ̂  = [1.5 0.0; 0.0 1.0],
    ),
    (
        label = "3D off-target",
        μ  = [0.0, 0.0, 0.0],
        Σ  = [1.0 0.3 0.0; 0.3 1.0 0.2; 0.0 0.2 1.0],
        a  = [-1.5, -1.5, -1.5],
        b  = [ 1.5,  1.5,  1.5],
        μ̂  = [0.1, -0.1, 0.05],
        Σ̂  = [0.5 0.05 0.0; 0.05 0.45 0.05; 0.0 0.05 0.55],
    ),
]

@testset "Surrogate gradient ∇L̃ (μA frozen)" begin
    for c in GRAD_CASES
        @testset "$(c.label)" begin
            p   = _grad_test_setup(c.μ, c.Σ, c.a, c.b)
            μA  = copy(c.μ̂)   # μA frozen at μ̂, as in the surrogate

            f(q) = approximate_vector_moment_loss(q, c.a, c.b, μA,
                                                  collect(c.μ̂), Matrix(c.Σ̂))
            g_an = vector_gradient(p, c.a, c.b, μA,
                                   collect(c.μ̂), Matrix(c.Σ̂))
            g_fd = _fd_grad(f, p)

            # Relative error scaled by ‖∇L̃_FD‖; absolute tolerance for the
            # very small components.
            rel = norm(g_an - g_fd) / max(norm(g_fd), 1e-10)
            @test rel < 1e-4
            @test norm(g_an - g_fd, Inf) < 1e-6
        end
    end
end

@testset "True-loss gradient ∇L (μA recomputed)" begin
    for c in GRAD_CASES
        @testset "$(c.label)" begin
            p = _grad_test_setup(c.μ, c.Σ, c.a, c.b)

            f(q) = vector_moment_loss(q, c.a, c.b,
                                      collect(c.μ̂), Matrix(c.Σ̂))
            g_an = vector_grad_true_loss(p, c.a, c.b,
                                         collect(c.μ̂), Matrix(c.Σ̂))
            g_fd = _fd_grad(f, p)

            rel = norm(g_an - g_fd) / max(norm(g_fd), 1e-10)
            @test rel < 1e-4
            @test norm(g_an - g_fd, Inf) < 1e-6
        end
    end
end

@testset "True-loss gradient vanishes at the true moments" begin
    # When μ̂, Σ̂ are exactly the truncated moments of d, L(μ,U) = 0 and
    # ∇L should be 0 to numerical precision.
    ne = get_example(n = 2, index = 4)
    p  = _grad_test_setup(collect(ne.μ), Matrix(ne.Σ), collect(ne.a), collect(ne.b))
    g_an = vector_grad_true_loss(p, collect(ne.a), collect(ne.b),
                                 collect(ne.μ̂), Matrix(ne.Σ̂))
    @test norm(g_an, Inf) < 1e-5
end
