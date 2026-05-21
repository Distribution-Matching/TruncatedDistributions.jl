module TruncatedDistributions

using Distributions
using HCubature
using LinearAlgebra
using PDMats
using Parameters
using Optim
using Combinatorics
using MvNormalCDF
using Printf
using Random

import Distributions: insupport, pdf, logpdf, moment, params
import Base: size, length, show, rand
import Statistics: mean, cov, var, std, cor

export
    # core types
    TruncationRegion,
    BoxTruncationRegion,
    EllipticalTruncationRegion,
    TruncatedMvDistribution,
    TruncatedMvDistributionState,
    TruncatedMvDistributionSecondOrderState,
    BasicBoxTruncatedMvNormal,
    RecursiveMomentsBoxTruncatedMvNormal,
    TruncatedMvNormal,

    # distribution queries
    intruncationregion,
    insupport,
    pdf,
    logpdf,
    rand,
    mean,
    cov,
    var,
    std,
    cor,
    params,
    moment,
    moments,
    tp,
    raw_moment,
    raw_moment_dict,
    raw_moment_from_indices,
    compute_tp,
    compute_mean,
    compute_cov,
    compute_moment,
    compute_moments,
    update_distribution!,
    outer_dist_from_state,

    # Kan–Robotti backend toggle
    set_kr_base_backend!,
    get_kr_base_backend,

    # integration helper
    hcubature_inf,

    # parameter fitting — public front door
    fit_mvnormal,

    # parameter fitting — lower-level building blocks
    warm_start_diagonal,
    block_coord_descent,
    moment_loss,
    vector_moment_loss,

    # explicit-gradient internals (advanced)
    vector_fg_true_loss,
    vector_fg_true_loss!,
    vector_grad_true_loss,
    grad_true_loss,
    moment_grad_μ,
    moment_grad_U,
    make_param_vec_from_μ_Σ,
    make_μ_Σ_from_param_vec,
    make_param_vec_from_μ_U,
    n_from_param_size,

    # bundled examples (used by tests / benchmarks)
    get_example,
    get_num_examples,
    get_example_sizes,
    dist_from_example

include("commonTypes.jl")
include("regions.jl")
include("hcubatureInf.jl")
include("commonOperations.jl")
include("commonCompute.jl")
include("univariate/distributionsPackageExtensions.jl")
include("multivariate/boxTruncatedMvNormalRecursiveMomentsState.jl")
include("multivariate/normal.jl")
include("multivariate/boxTruncatedMvNormalRecursiveMoments.jl")
include("multivariate/examples.jl")
include("parameterMatching/lossMultivariateFit.jl")
include("parameterMatching/parameter_gradients_true_loss.jl")
include("parameterMatching/warm_start.jl")
include("parameterMatching/block_coord_descent.jl")
include("parameterMatching/fit.jl")

end # module
