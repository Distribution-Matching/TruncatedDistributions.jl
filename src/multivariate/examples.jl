"""
    Returns an example.
"""
function get_example(;dist_type = MvNormal, n = 2, index = 1)
    if dist_type == MvNormal
        if haskey(normal_examples, n)
            return normal_examples[n][index]
        else
            error("No examples for $dist_type with n=$n")
        end
    else
        error("Not supporting dist type $dist_type")
    end
end

"""
    Returns the number of examples for a given n.
"""
function get_num_examples(n; dist_type = MvNormal)
    if dist_type == MvNormal
        return length(normal_examples[n])
    else
        error("Not supporting dist type $dist_type")
    end
end

"""
    Returns the possible example sizes.
"""
function get_example_sizes(;dist_type = MvNormal)
    return sort(collect(keys(normal_examples)))
end

"""
    An example. The `arb_moment_to_check_index` is some tuple of indexes of a moment to check (can be nothing). 
    The associated value is `arb_moment_to_check_value`. This is for checking some single (arbitrary) moment.
"""
@with_kw struct NormalExample
    n::Int = 2
    μ::AbstractVector{Float64}
    Σ::AbstractMatrix{Float64}
    a::AbstractVector{Float64}
    b::AbstractVector{Float64}
    tp::Float64
    μ̂::AbstractVector{Float64}
    Σ̂::AbstractMatrix{Float64}
    arb_moment_to_check_index::Union{Nothing,Tuple{Vararg{Int}}} = nothing
    arb_moment_to_check_value::Union{Nothing, Float64} = nothing
end

"""
Create a distribution from an example.
"""
function dist_from_example(ne::NormalExample)
    return TruncatedMvNormal(ne.μ, ne.Σ, ne.a, ne.b)
end

#########################
## Hard coded examples ##
#########################
normal_examples = Dict()
normal_examples[2] = [
    NormalExample(  μ = [2.5, 3.5],
                    Σ = [2.0 -0.5;
                        -0.5 5.0],
                    a = [-2.3,-20.],
                    b = [4.,12.3],
                    tp = 0.8551938607791414,
                    μ̂ = [2.126229598541412, 3.5930224468784577],
                    Σ̂ = [1.0 0; 0 1.0],
                    arb_moment_to_check_index = (3, 5),
                    arb_moment_to_check_value = 58529.1327440061),
    NormalExample(  μ = [2.5, 3.5],
                    Σ = [  3.3 0.5;
                            0.5 5.0],
                    a = [-5.4,-20.],
                    b = [2.4,6.3],
                    tp = 0.43660920327458974,
                    μ̂ = [0.9734400003512856, 2.886032492774952],
                    Σ̂ = [1.0 0; 0 1.0],
                    arb_moment_to_check_index = (0, 3),
                    arb_moment_to_check_value = 52.44849808917711),
    # Manjunath & Wilhelm (2021), "Moments Calculation for the Doubly Truncated Multivariate
    # Normal Density", Example 1. Uses the true a[2] = -Inf as specified in the original
    # paper; the package's `hcubature_inf` wrapper substitutes the unbounded coordinate
    # to a finite interval, and the Kan–Robotti recursion routes its base case through the
    # same wrapper, so both code paths handle the semi-infinite domain natively.
    NormalExample(  μ = [0.5, 0.5],
                    Σ = [1.0 1.2;
                         1.2 2.0],
                    a = [-1.0, -Inf],
                    b = [ 0.5,   1.0],
                    tp = 0.398482903122761,
                    μ̂ = [-0.15163426285883586, -0.3881151019108365],
                    Σ̂ = [0.16304394651960308 0.1613370775178306;
                         0.1613370775178306  0.6062505412608046]),
    # Fully-bounded 2D example: all four box sides are within ~1.5σ of the mean,
    # so each side meaningfully shapes the truncated distribution.
    NormalExample(  μ = [0.0, 0.0],
                    Σ = [1.0 0.3;
                         0.3 1.0],
                    a = [-1.0, -1.5],
                    b = [ 1.5,  1.0],
                    tp = 0.6046944921624118,
                    μ̂ = [ 0.12213214335225087, -0.12213214335225087],
                    Σ̂ = [0.4081962800466608  0.05396791660268199;
                         0.05396791660268199 0.40819628004683206])
]
normal_examples[3] = [
    NormalExample(  μ = [3.5,2,3.5],
                    Σ = [ 7. 1 0 ;
                          1 3.3 2  ;
                          0 2 3.8 ],
                    a = [-4. ,-3 ,-1],
                    b = [7.5 ,6.5 , 6.5],
                    tp = 0.2771862142891479,
                    μ̂ = [3.1593375223480122, 1.8453845525318782, 3.3023816830081723],
                    Σ̂ =  [5.28031    0.719954  -0.0155715 ; 0.719954   2.8325     1.38021 ;-0.0155715  1.38021    2.71015],
                    arb_moment_to_check_index = (1, 1, 1),
                    arb_moment_to_check_value = 11.740894033054031),
    # Semi-infinite 3D analogue of the Manjunath–Wilhelm 2D example: the
    # third coordinate has an unbounded lower face (a[3] = -Inf), the other
    # two faces are finite. Tests that the explicit-gradient + MvNormalCDF
    # stack handles ±Inf bounds at n = 3.
    NormalExample(  μ = [0.0, 0.0, 0.0],
                    Σ = [1.0  0.3  0.0;
                         0.3  1.0  0.3;
                         0.0  0.3  1.0],
                    a = [-1.0, -1.0, -Inf],
                    b = [ 1.5,  1.5,  1.0],
                    tp = NaN,
                    μ̂  = zeros(3),
                    Σ̂  = zeros(3, 3)),
    # Genz & Bretz (2009), Computation of Multivariate Normal and t
    # Probabilities, §1.3.1 (Φ3ex). Trivariate centred normal, upper-tailed
    # semi-infinite truncation. The reported true probability is
    # 0.827984897456834.
    NormalExample(  μ = [0.0, 0.0, 0.0],
                    Σ = [1.0    3/5    1/3 ;
                         3/5    1.0   11/15;
                         1/3   11/15   1.0],
                    a = [-Inf, -Inf, -Inf],
                    b = [ 1.0,  4.0,  2.0],
                    tp = 0.827984897456834,
                    μ̂  = zeros(3),
                    Σ̂  = zeros(3, 3))
]
normal_examples[4] = [
    NormalExample(  μ = [3.5,2,3.5,3.5],
                    Σ = [ 7. 1 0 1 ; 
                            1 3.3 2 0 ;
                            0 2 3.4 0 ;
                            1 0 0 4 ],
                    a = [-8.0 ,-10, -4 ,-20],
                    b = [6.0 ,15. ,13, 10],
                    tp = 0.2806670136910537,
                    μ̂ = [2.2496049084043497, 0.7606400499117575, 1.6967554685821928, 3.321372131126033],
                    Σ̂ = [1.0 0 0 0 ; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0],
                    arb_moment_to_check_index = (2, 2, 2, 3),
                    arb_moment_to_check_value = 10550.695322644422)
]
# --------------------------------------------------------------------------
# Higher-dimensional examples (n = 5, 6, 7). One mildly-truncated and one
# heavily-truncated box per dimension. μ = 0; Σ has unit diagonal and a
# tridiagonal correlation band (Σ_{ij} = 0.3 for |i-j| = 1) so the
# distribution is correlated but trivially positive-definite. Targets
# `tp`, `μ̂`, `Σ̂` are placeholders — the benchmark computes the true
# targets from each example's distribution and discards these fields.
# --------------------------------------------------------------------------

# Σ_{ij} = 1 if i=j, 0.3 if |i-j|=1, 0 otherwise — PD by diagonal dominance.
function _tridiag_corr(n::Int, ρ::Float64 = 0.3)
    M = Matrix{Float64}(I, n, n)
    for i in 1:(n-1)
        M[i, i+1] = ρ
        M[i+1, i] = ρ
    end
    return M
end

# Light: box [-2, 2]^n  → univariate Φ(2)−Φ(−2) ≈ 0.954, mass ≈ 0.95^n
# Heavy: box [-1.5, 1.5]^n → univariate Φ(1.5)−Φ(−1.5) ≈ 0.866,
#                            mass ≈ 0.87^n. At n=8 m^(0) ≈ 0.32, low
#                            enough to be a meaningful heavy-truncation
#                            case but not so low that the correlation
#                            structure of the underlying Gaussian gets
#                            washed out by the truncation (which would
#                            make the warm-start globally optimal and
#                            leave the BCD nothing to do).
function _make_high_n_examples(n::Int)
    μ  = zeros(n)
    Σ  = _tridiag_corr(n)
    placeholders = (tp = NaN, μ̂ = zeros(n), Σ̂ = zeros(n, n))
    return [
        NormalExample(n = n, μ = μ, Σ = Σ,
                      a = fill(-2.0, n), b = fill(2.0, n);
                      placeholders...),
        NormalExample(n = n, μ = μ, Σ = Σ,
                      a = fill(-1.5, n), b = fill(1.5, n);
                      placeholders...),
    ]
end

# Σ_{ij} = min(i, j) — staircase covariance from Genz & Bretz (2009),
# §1.3.2 and §1.3.3. Positive-definite for all n (Cholesky factor is
# the lower-triangular all-ones matrix scaled per row).
function _staircase_cov(n::Int)
    M = zeros(n, n)
    for i in 1:n, j in 1:n
        M[i, j] = float(min(i, j))
    end
    return M
end

# Genz & Bretz (2009), §1.3.2: Φ5ex. Pentavariate normal with staircase
# covariance and hyper-rectangle reported in the book as a "lower"
# vector (-1,...,-5) and an "upper" vector (2,...,6). The book's
# notation pairs the leftmost integral sign with the leftmost dx
# (i.e., dx_5), so a[1] in our `x_1`-indexed convention is the
# REVERSAL of the book's listed values: a = (-5,-4,-3,-2,-1),
# b = (6,5,4,3,2). Reported true probability ≈ 0.4741284.
const _genz_phi5 = NormalExample(n = 5, μ = zeros(5),
    Σ = _staircase_cov(5),
    a = [-5.0, -4.0, -3.0, -2.0, -1.0],
    b = [ 6.0,  5.0,  4.0,  3.0,  2.0],
    tp = 0.4741284, μ̂ = zeros(5), Σ̂ = zeros(5, 5))

# Genz & Bretz (2009), §1.3.3: Φ8ex. Octavariate analogue with the
# same staircase covariance; the same bound-order convention applies,
# so a = (-8,...,-1), b = (9,...,2). Reported true
# probability ≈ 0.32395.
const _genz_phi8 = NormalExample(n = 8, μ = zeros(8),
    Σ = _staircase_cov(8),
    a = collect(-8.0:1.0:-1.0),
    b = collect( 9.0:-1.0: 2.0),
    tp = 0.32395, μ̂ = zeros(8), Σ̂ = zeros(8, 8))

normal_examples[5]  = vcat(_make_high_n_examples(5), [_genz_phi5])
normal_examples[6]  = _make_high_n_examples(6)
normal_examples[7]  = _make_high_n_examples(7)
normal_examples[8]  = vcat(_make_high_n_examples(8), [_genz_phi8])
normal_examples[9]  = _make_high_n_examples(9)
normal_examples[10] = _make_high_n_examples(10)
normal_examples[20] = []
normal_examples[50] = []
