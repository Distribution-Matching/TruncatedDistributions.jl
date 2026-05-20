module TruncatedDistributions

using Distributions
using HCubature
using LinearAlgebra
using PDMats
using ProgressMeter
using DifferentialEquations
using Parameters
using PRIMA
using Optim
using Combinatorics

import Distributions: insupport, pdf, moment
import Base: size, length, show, rand
import Statistics: mean, cov

export 
    compute_tp,
    compute_mean,
    compute_cov,
    compute_moment,
    TruncationRegion,
    TruncatedMvDistribution,
    BoxTruncatedMvNormalRecursiveMoments,#QQQQ
    BoxTruncationRegion,
    intruncationregion,
    insupport,
    BoxTruncatedMvNormal,
    moment,
    moments,
    compute_moments,
    alpha,
    pdf,
    raw_moment,
    raw_moment_from_indices,
    raw_moment_dict,
    mean,
    cov,
    rand,
    tp,
    TruncatedMvDistributionSecondOrderState,
    BasicBoxTruncatedMvNormal,
    RecursiveMomentsBoxTruncatedMvNormal,
    EllipticalTruncationRegion,
    TruncatedMvDistributionState,
    μ_gradient,
    U_gradient,
    grad_true_loss,
    moment_grad_μ,
    moment_grad_U,
    vector_grad_true_loss,
    vector_fg_true_loss,
    loss_based_fit,
    truncateDynamicFit,
    moment_loss,
    approximate_moment_loss,
    vector_moment_loss,
    approximate_vector_moment_loss,
    vector_gradient,
    make_μ_Σ_from_param_vec,
    make_param_vec_from_μ_Σ,
    make_param_vec_from_μ_U,
    n_from_param_size,
    get_example,
    get_num_examples,
    get_example_sizes,
    dist_from_example,
    correct_to_moments_with_prima,
    correct_to_moments_with_optim,
    correct_to_moments_with_optim_explicit_grad,
    correct_to_moments_with_optim_surrogate_grad,
    correct_to_moments_with_pair_gradient_descent,
    correct_to_moments_with_full_gradient,
    find_pair_with_worst_loss,
    pair_gradient_descent

include("commonTypes.jl")
include("regions.jl")
include("hcubatureInf.jl")
include("commonOperations.jl")
include("commonCompute.jl")
    include("univariate/distributionsPackageExtensions.jl")
    include("univariate/truncateDynamicFit.jl")
    include("multivariate/boxTruncatedMvNormalRecursiveMomentsState.jl")
    include("multivariate/normal.jl")
    include("multivariate/otherThanNormal.jl")
    include("multivariate/boxTruncatedMvNormalRecursiveMoments.jl")
    include("multivariate/examples.jl")

    include("parameterMatching/dynamicUnivariateFit.jl")
    include("parameterMatching/parameter_gradients.jl")
    include("parameterMatching/correct_to_moments.jl")

include("parameterMatching/lossMultivariateFit.jl")
include("parameterMatching/parameter_gradients_true_loss.jl")

end #module