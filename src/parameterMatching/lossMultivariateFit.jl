function loss_based_fit(μ̂::Vector{Float64}, Σ̂::Matrix{Float64}, a::Vector{Float64}, b::Vector{Float64}; 
                        μ_init::Vector{Float64} = μ̂, 
                        Σ_init::Matrix{Float64} = Σ̂,
                        α = 0.01,
                        min_grad_norm = 1e-2,
                        max_iter = Inf)
    n = length(μ̂)
    n1, m1 = size(Σ̂)
    (n != n1 || n != m1) && error("Mismatch of dimensions")  
    
    std_devs = sqrt.(diag(Σ̂))
    μ̂0 = zeros(n);
    Σ̂0 =  PDMat(Σ̂ ./ (std_devs * std_devs'))
    a0 = (a - μ̂) ./ std_devs
    b0 = (b - μ̂) ./ std_devs
    !all(a0 .< 0.0) && error("a needs to be less than μ̂")
    !all(b0 .> 0.0) && error("b needs to be less than μ̂")

    #QQQQ check if sigma_i is too high for a given [a,b] then flag it is not possible - error 

    μ_init0 = (μ_init- μ̂) ./ std_devs
    Σ_init0 = PDMat(Σ_init ./ (std_devs * std_devs'))


    losses_μ = []
    losses_Σ = []
    dists = []
    losses_total = []
    μ = μ_init0
    Σ = Σ_init0
    U = cholesky(0.5*(inv(Σ) + inv(Σ)')).U
    dtrunc = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a0, b0; max_moment_levels = 4);

    
    grad_norm_sum = Inf
    i = 1
    while grad_norm_sum > min_grad_norm && i < max_iter
        push!(dists,  RecursiveMomentsBoxTruncatedMvNormal(dtrunc.untruncated.μ .*std_devs + μ̂, PDMat(dtrunc.untruncated.Σ .* (std_devs * std_devs')), a, b; max_moment_levels = 2))
        dtrunc = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a0, b0; max_moment_levels = 4);
        μA = mean(dtrunc)
        μ_grad = μ_gradient(dtrunc, μA, μ̂0, Σ̂0)'
        U_grad = U_gradient(dtrunc, μA, μ̂0, Σ̂0) #QQQQ pass in U
        μ = μ - 10*α*μ_grad #gradient based update
        U = U - α*U_grad #gradient based update
        Ui = inv(U)
        Σ = PDMat(Ui*Ui')
        loss_μ = norm(mean(dtrunc) - μ̂0)
        loss_Σ = norm(cov(dtrunc) - Σ̂0)
        loss_total = loss_μ + loss_Σ
        grad_norm_sum = norm(μ_grad) + norm(U_grad)
        # @show i, loss_total, grad_norm_sum#, (loss_μ, loss_Σ)
        push!(losses_total, loss_total)
        push!(losses_μ, loss_μ)
        push!(losses_Σ, loss_Σ)
        i+=1
    end
    Σ = PDMat(Σ .* (std_devs * std_devs')) #back to original coordinates
    μ = μ .*std_devs + μ̂
    return RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 4), 
                (losses_total = losses_total, 
                losses_μ = losses_μ,
                losses_Σ = losses_Σ,
                dists = dists)
end

function moment_loss(dist::RecursiveMomentsBoxTruncatedMvNormal, μ̂::AbstractVector{Float64}, Σ̂::AbstractMatrix{Float64})
    # Use the cached Kan–Robotti primitive moments directly. Calling
    # mean(dist) / cov(dist) instead would re-integrate the dist via
    # HCubature on x f(x) and x x^T f(x) — bypassing the recursion that's
    # already populated in dist.state — and a single moment_loss call
    # would then dominate the cost of the full f + ∇f pair (~96% of
    # benchmark time, ~44M allocations and 1.4 GiB per call at n=3).
    n  = length(dist)
    m0 = raw_moment_from_indices(dist, Int[])
    # Guard against the LBFGS line search wandering into a region where
    # the truncation probability m^{(0)} → 0; μA = m^{(1)}/m^{(0)} then
    # overflows and the optimizer would freeze at L = Inf. Return a large
    # finite penalty so the line search backs off instead.
    if !isfinite(m0) || m0 < eps(Float64)
        return prevfloat(Inf) / 4
    end
    m1 = [raw_moment_from_indices(dist, [i])    for i in 1:n]
    m2 = [raw_moment_from_indices(dist, [i, j]) for i in 1:n, j in 1:n]
    μA = m1 ./ m0
    ΣA = m2 ./ m0 .- μA * μA'
    L = 0.5 * (sum(abs2, μA .- μ̂) + sum(abs2, ΣA .- Σ̂))
    return isfinite(L) ? L : prevfloat(Inf) / 4
end

function approximate_moment_loss(d::RecursiveMomentsBoxTruncatedMvNormal,
                                μA::Vector{Float64},
                                μ̂::AbstractVector{Float64},  
                                Σ̂::AbstractMatrix{Float64})
    n = length(d)
    m(inds) = raw_moment_from_indices(d, inds)
    m0 = m(Int[])
    term1 = sum(abs2, m([i]) - m0*μ̂[i] for i in 1:n)
    term2 = sum(abs2, [m([i,j]) - m([i])*μA[j] - m([j])*μA[i] + m0*(μA[i]*μA[j] - Σ̂[i,j])  for i in 1:n, j in 1:n])
    # @show term1, term2
    return (term1 + term2)/2
end

function vector_moment_loss(param_vec::Vector{Float64},
                            a,
                            b,
                            μ̂::AbstractVector{Float64},
                            Σ̂::AbstractMatrix{Float64})
    μ, Σ = make_μ_Σ_from_param_vec(param_vec)
    # Σ comes from U^{-1} U^{-T}; round-off during LBFGS line search can make
    # it slightly non-PD. Symmetrize and add a tiny jitter so the Cholesky
    # inside PDMat does not throw and abort the optimization.
    Σsym = 0.5 .* (Σ .+ Σ')
    Σsym .+= eps(Float64) * (tr(Σsym) + 1.0) * Matrix{Float64}(I, size(Σsym))
    dist = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σsym), a, b)
    return moment_loss(dist, μ̂, Σ̂)
end


function approximate_vector_moment_loss(param_vec::Vector{Float64},
                            a,
                            b,
                            μA::Vector{Float64},
                            μ̂::AbstractVector{Float64},
                            Σ̂::AbstractMatrix{Float64})
    μ, Σ = make_μ_Σ_from_param_vec(param_vec)
    Σsym = 0.5 .* (Σ .+ Σ')
    Σsym .+= eps(Float64) * (tr(Σsym) + 1.0) * Matrix{Float64}(I, size(Σsym))
    dist = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σsym), a, b)
    return approximate_moment_loss(dist, μA, μ̂, Σ̂)
end

function vector_gradient(   param_vec::Vector{Float64},
                            a,
                            b,
                            μA::AbstractVector{Float64},
                            μ̂::AbstractVector{Float64},
                            Σ̂::AbstractMatrix{Float64})
    μ, Σ = make_μ_Σ_from_param_vec(param_vec)
    Σsym = 0.5 .* (Σ .+ Σ')
    Σsym .+= eps(Float64) * (tr(Σsym) + 1.0) * Matrix{Float64}(I, size(Σsym))
    dist = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σsym), a, b; max_moment_levels = 4)
    μ_grad = μ_gradient(dist, μA, μ̂, PDMat(Σ̂))'
    U_grad = U_gradient(dist, μA, μ̂, PDMat(Σ̂))
    make_param_vec_from_μ_U(μ_grad, U_grad)
end

function make_μ_Σ_from_param_vec(param_vec)
    n = n_from_param_size(length(param_vec))
    μ = param_vec[1:n]
    inds = [CartesianIndex(i,j) for i=1:n for j=i:n]
    U = zeros(n,n)
    U[inds] = param_vec[(n+1):end]
    U = UpperTriangular(U)
    Ui = inv(U)
    Σ = Ui*Ui'  #retrieve the covaranice from the upper triangular matrix
    return μ, Σ
end

function make_param_vec_from_μ_Σ(μ, Σ)
    # inv(Σ) can have asymmetric round-off even when Σ is symmetric PD;
    # cholesky requires exact symmetry. Symmetrize before factorizing.
    Σi = inv(Σ)
    F = cholesky(0.5 .* (Σi .+ Σi'))
    U = F.U
    n = size(U)[1]
    inds = [CartesianIndex(i,j) for i=1:n for j=i:n]
    vcat(μ, U[inds]) #first two coordinates are the mean and the remaining coordinates are the factorized covariance
end

function make_param_vec_from_μ_U(μ, U)
    n = size(U)[1]
    inds = [CartesianIndex(i,j) for i=1:n for j=i:n]
    vcat(μ,U[inds]) #first two coordinates are the mean and the remaining coordinates are the factorized covariance
end

#n + n*(n+1)/2 = T
# 3n/2 + n^2/2  = T
# 3n + n^2 = 2T
# n^2 + 3n - 2T
# n = (-3 + sqrt(9 +8T))/2
function n_from_param_size(param_size::Integer)
    return Int((-3 + sqrt(9+8param_size))/2)
end

function find_pair_with_worst_loss(μ, Σ, μ̂, Σ̂, a, b)
    n = length(μ)
    pair_sets = collect(combinations(1:n, 2))
    pair_losses = zeros(length(pair_sets))
    for (i,s) in enumerate(pair_sets)
        μ_s = μ[s]
        Σ_s = Σ[s,s]
        μ̂_s = μ̂[s]
        Σ̂_s = Σ̂[s,s]
        a_s = a[s]
        b_s = b[s];
        dtrunc = RecursiveMomentsBoxTruncatedMvNormal(μ_s, PDMat(Σ_s), a_s, b_s)
        pair_losses[i] = moment_loss(dtrunc, μ̂_s, Σ̂_s)
    end
    return pair_sets[argmax(pair_losses)]
end


function pair_gradient_descent(μ̂, Σ̂, a, b)    
    μ = copy(μ̂) 
    Σ = copy(Σ̂)

    total_loss = []
    dtrunc_all_coords = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ),a,b)
    for i in 1:500
        current_pair = find_pair_with_worst_loss(μ, Σ, μ̂, Σ̂, a, b)
        @info "Starting iteration $i on set $current_pair"

        # #create the 2x2 problem
        μ̂_current = μ̂[current_pair]
        Σ̂_current = Σ̂[current_pair, current_pair]
        a_current = a[current_pair]
        b_current = b[current_pair];

        #do a few gradient descent steps on the 2x2 problem
        dtrunc, logs = loss_based_fit(μ̂_current, Σ̂_current, a_current, b_current; 
                                        μ_init = μ[current_pair],
                                        Σ_init = Σ[current_pair, current_pair],
                                        max_iter=max(3,round(Int,300/i^2)), α = 0.05);        
        μ[current_pair] = dtrunc.untruncated.μ
        Σ[current_pair, current_pair] = dtrunc.untruncated.Σ


        dtrunc_all_coords = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σ),a,b)
        all_coordinates_loss = moment_loss(dtrunc_all_coords, μ̂, Σ̂)
        @show all_coordinates_loss
        if all_coordinates_loss < 1e-3
            break
        end
        push!(total_loss, all_coordinates_loss)
    end
    return dtrunc_all_coords
end
