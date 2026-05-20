function μ_gradient(    d::RecursiveMomentsBoxTruncatedMvNormal, 
                        μA::Vector{Float64}, 
                        μ̂::Vector{Float64}, 
                        Σ̂::PDMat)
    μ = d.untruncated.μ
    Σ = d.untruncated.Σ
    n = length(d)
    m(inds) = raw_moment_from_indices(d, inds)
    m0 = m(Int[])

    ii(k) = ((k-1) % n) + 1
    jj(k) = floor(Int,(k-1)/n) + 1
    I1 = [m([i]) - m0*μ̂[i] for i in 1:n]
    I2 = [m([i,j]) - m([i])*μ[j] - m([j])*μ̂[i] + m0*μ̂[i]*μ[j] for i in 1:n, j in 1:n]
    I3 = [(
        m([ii(k), jj(k)]) - m([ii(k)])*μA[jj(k)] - m([jj(k)])*μA[ii(k)] 
        + m0*(μA[ii(k)]*μA[jj(k)] - Σ̂[ii(k),jj(k)])
        )
        for k in 1:n^2]
    # Paper eq (eq:I-mu-4): 8 terms. Earlier revisions of this file were
    # missing the m^{(1)}_{i(k)} μA_{j(k)} μ_ℓ term and the μ_ℓ factor on
    # the closing m^{(0)} term, both restored here.
    I4 = [(
        m([ii(k),jj(k),l]) - m([ii(k),l])*μA[jj(k)] - m([jj(k),l])*μA[ii(k)] - m([ii(k),jj(k)])*μ[l]
        + m([l])*(μA[ii(k)]*μA[jj(k)]-Σ̂[ii(k),jj(k)])
        + m([jj(k)])*μA[ii(k)]*μ[l]
        + m([ii(k)])*μA[jj(k)]*μ[l]
        - m0*μ[l]*(μA[ii(k)]*μA[jj(k)] - Σ̂[ii(k),jj(k)])
        )
        for k in 1:n^2, l in 1:n]
    return (I1'*I2 + I3'*I4)*inv(Matrix(Σ)) #note it returns a row (just like the paper)
end

function U_gradient(    d::RecursiveMomentsBoxTruncatedMvNormal, 
                        μA::Vector{Float64}, 
                        μ̂::Vector{Float64}, 
                        Σ̂::PDMat;
                        U::Union{Nothing, Matrix{Float64}} = nothing) # U is optional
    μ = d.untruncated.μ
    Σ = d.untruncated.Σ
    if isnothing(U)
        U = cholesky(0.5*(inv(Matrix(Σ)) + inv(Matrix(Σ))')).U
    end
    iU = inv(U) #QQQQ see if to pass in as optional arg?
    n = length(d)

    m(inds) = raw_moment_from_indices(d, inds)
    m0 = m(Int[])

    #First matrix
    I1(i) = fill(m([i]) - m0*μ̂[i], n, n)

    #Second matrix
    Ĩ2(k,l,i) = - sum(U[k,s]*(m([i,s,l]) - m([i,s])*μ[l] - m([l,s])*μ̂[i] - m([i,l])*μ[s] + m([i])*μ[l]*μ[s] + m([s])*μ̂[i]*μ[l] + m([l])*μ̂[i]*μ[s] - m0*μ̂[i]*μ[s]*μ[l])  for s in k:n) 
    Ĩ2diag(k,i) = -iU[k,k]*(m0*μ̂[i] - m([i]))
    I2(i) = [(if k < l
                    Ĩ2(k,l,i)
                elseif k == l
                    Ĩ2diag(k,i) + Ĩ2(k,k,i)
                else
                    0
                end) for k in 1:n, l in 1:n]
       
    #Third matrix     
    I3(i,j) = fill(m([i,j]) - m([i])*μA[j] - μA[i]*m([j]) + (μA[i]*μA[j] - Σ̂[i,j])*m0, n, n)
          
    #Fourth matrix     
    Ĩ4(k, l, i, j) = -sum(U[k, s]*( m([i, j, s, l])
                                    - m([j, s, l]) * μA[i]
                                    - m([i, j, l]) * μ[s]
                                    - m([i, s, l]) * μA[j]
                                    - m([i, j, s]) * μ[l]
                                    + m([j, l]) * μA[i] * μ[s]
                                    + m([s, l]) * μA[i] * μA[j]
                                    - m([s, l]) * Σ̂[i, j]
                                    + m([i, l]) * μA[j] * μ[s]
                                    + m([j, s]) * μ[l] * μA[i]
                                    + m([i, j]) * μ[l] * μ[s]
                                    + m([i, s]) * μ[l] * μA[j]
                                    - m([l]) * μA[i] * μA[j] * μ[s]
                                    + m([l]) * Σ̂[i, j] * μ[s]
                                    - m([j]) * μA[i] * μ[s] * μ[l]
                                    - m([s]) * μA[i] * μA[j] * μ[l]
                                    + m([s]) * Σ̂[i, j] * μ[l]
                                    - m([i]) * μA[j] * μ[s] * μ[l]
                                    + m0 * (μA[i] * μA[j] * μ[s] * μ[l] - Σ̂[i, j] * μ[s] * μ[l])) for s in k:n)
    # Paper eq (eq:I-U-4) k=ℓ case: m0 multiplies (μA_i μA_j − Σ̂_{i,j}),
    # not only μA_i μA_j. Earlier revision had Σ̂ outside the m0 factor.
    Ĩ4diag(k,i,j) =  iU[k,k]*(m([i,j]) -m([j])*μA[i]-m([i])*μA[j]+ m0 * (μA[i]*μA[j] - Σ̂[i, j]))
    I4(i,j) = [(if k < l
                    Ĩ4(k,l,i,j)
                    elseif k == l
                        Ĩ4diag(k,i,j) + Ĩ4(k,k,i,j)
                    else
                        0
                    end) for k in 1:n, l in 1:n]

    #Fifth matrix     
    I5(i) = fill(m([i,i]) - 2m([i])*μA[i]  + (μA[i]^2 - Σ̂[i,i])*m0, n, n)

    #Sixth matrix     
    Ĩ6(k, l, i) = -sum(U[k, s] * (  m([i, i, s, l])
                                    - 2 * m([i, s, l]) * μA[i]
                                    -     m([i, i, l]) * μ[s]
                                    -     m([i, i, s]) * μ[l]
                                    +     m([i, i]) * μ[s] * μ[l]
                                    + 2 * m([i, l]) * μA[i] * μ[s]
                                    + 2 * m([i, s]) * μ[l] * μA[i]
                                    +     m([s, l]) * μA[i]^2
                                    -     m([s, l]) * Σ̂[i, i]
                                    +     m([s]) * Σ̂[i, i] * μ[l]
                                    +     m([l]) * Σ̂[i, i] * μ[s]
                                    -     m([l]) * μ[s] * μA[i]^2
                                    - 2 * m([i]) * μA[i] * μ[s] * μ[l]
                                    -     m([s]) * μA[i]^2 * μ[l]
                                    +     m0 * (μA[i]^2 * μ[s] * μ[l] - μ[s] * μ[l] * Σ̂[i, i])) for s in k:n)
    Ĩ6diag(k,i) =  iU[k,k]*(m([i,i]) -2m([i])*μA[i] + m0 * (μA[i]^2- Σ̂[i, i]))
    I6(i) = [(if k < l
                    Ĩ6(k,l,i)
                elseif k == l
                    Ĩ6diag(k,i) + Ĩ6(k,k,i)
                else
                    0
                end) for k in 1:n, l in 1:n]

    total34 = zeros(n,n)
    for i in 1:n
        for j in (i+1):n
            total34 += I3(i,j) .* I4(i,j)
        end
    end                
    return sum([I1(i) .* I2(i) + I5(i) .* I6(i) for i in 1:n]) + 2*total34
end