function correct_to_moments_with_prima(     d::RecursiveMomentsBoxTruncatedMvNormal, 
                                            μ̂::AbstractVector{Float64},
                                            Σ̂::AbstractMatrix{Float64})
    prima_result, prima_info = newuoa((v)->vector_moment_loss(v, d.region.a, d.region.b, μ̂, Σ̂), make_param_vec_from_μ_Σ(μ̂, Σ̂); ftarget = 1e-3)
    # @show prima_info
    μ_prima, Σ_prima = make_μ_Σ_from_param_vec(prima_result)
    return RecursiveMomentsBoxTruncatedMvNormal(μ_prima, PDMat(Σ_prima),d.region.a, d.region.b)
end

function correct_to_moments_with_optim(     d::RecursiveMomentsBoxTruncatedMvNormal,
                                            μ̂::AbstractVector{Float64},
                                            Σ̂::AbstractMatrix{Float64})
    optim_result = optimize((v)->vector_moment_loss(v, d.region.a, d.region.b, μ̂, Σ̂),  #function
                            make_param_vec_from_μ_Σ(μ̂, Σ̂), #initial value
                            LBFGS(),
                            Optim.Options(show_trace = false,
                                          iterations = 50,
                                          time_limit = 60.0,
                                          callback   = s -> s.value < 1e-3))
    μ_optim, Σ_optim = make_μ_Σ_from_param_vec(optim_result.minimizer)
    return RecursiveMomentsBoxTruncatedMvNormal(μ_optim, PDMat(Σ_optim),d.region.a, d.region.b)
end

# LBFGS with the explicit *true-loss* gradient. Uses Optim's only_fg!
# interface so a single Kan–Robotti recursion serves both the loss and
# the gradient per iteration — half the recursion work of the previous
# implementation that called f and g! separately.
function correct_to_moments_with_optim_explicit_grad(
                                            d::RecursiveMomentsBoxTruncatedMvNormal,
                                            μ̂::AbstractVector{Float64},
                                            Σ̂::AbstractMatrix{Float64})
    a, b   = d.region.a, d.region.b
    μ̂v     = collect(μ̂)
    Σ̂m     = Matrix(Σ̂)
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂v, Σ̂m)
    p0 = make_param_vec_from_μ_Σ(μ̂v, Σ̂m)
    res = optimize(Optim.only_fg!(fg!), p0, LBFGS(),
                   Optim.Options(show_trace = false,
                                 iterations = 50,
                                 time_limit = 30.0,
                                 callback   = s -> s.value < 1e-3))
    μ_fit, Σ_fit = make_μ_Σ_from_param_vec(res.minimizer)
    return RecursiveMomentsBoxTruncatedMvNormal(μ_fit, PDMat(Σ_fit), a, b)
end

# LBFGS with the surrogate L̃ and its explicit gradient. μA is frozen at μ̂
# throughout the run (matched (f, ∇f) on L̃). The degeneracy of L̃ described
# in §2b means this method can converge to a spurious minimum where
# m^{(0)} → 0; a tight iteration cap and time limit are set so a failure
# on one case does not stall the benchmark.
function correct_to_moments_with_optim_surrogate_grad(
                                            d::RecursiveMomentsBoxTruncatedMvNormal,
                                            μ̂::AbstractVector{Float64},
                                            Σ̂::AbstractMatrix{Float64})
    a, b   = d.region.a, d.region.b
    μ̂v     = collect(μ̂)
    Σ̂m     = Matrix(Σ̂)
    μA     = copy(μ̂v)
    f(p)     =  approximate_vector_moment_loss(p, a, b, μA, μ̂v, Σ̂m)
    g!(g, p) = (g .= vector_gradient(p, a, b, μA, μ̂v, Σ̂m))
    p0 = make_param_vec_from_μ_Σ(μ̂v, Σ̂m)
    res = optimize(f, g!, p0, LBFGS(),
                   Optim.Options(show_trace = false,
                                 iterations = 50,
                                 time_limit = 30.0,
                                 callback   = s -> s.value < 1e-3))
    μ_fit, Σ_fit = make_μ_Σ_from_param_vec(res.minimizer)
    return RecursiveMomentsBoxTruncatedMvNormal(μ_fit, PDMat(Σ_fit), a, b)
end

function correct_to_moments_with_pair_gradient_descent( d::RecursiveMomentsBoxTruncatedMvNormal,
                                                        μ̂::AbstractVector{Float64},
                                                        Σ̂::AbstractMatrix{Float64})
    return pair_gradient_descent(μ̂, Σ̂, d.region.a, d.region.b)
end

function correct_to_moments_with_full_gradient( d::RecursiveMomentsBoxTruncatedMvNormal,
                                                μ̂::AbstractVector{Float64},
                                                Σ̂::AbstractMatrix{Float64})
    dtrunc, _ = loss_based_fit(μ̂, Matrix(Σ̂), d.region.a, d.region.b)
    return dtrunc
end

