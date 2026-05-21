function compute_tp(d::TruncatedMvDistribution{D,R}; 
                    tol::Float64 = 10e-4, 
                    tol_step::Int = 10^5,
                    alg::Symbol = :hc) where {D,R}
    if alg == :mc
        i, n = 0, 0
        err = 1.0
        tp = 0.0
        while err ≥ tol
            candidate = rand(d.untruncated)
            i += intruncationregion(d.region,candidate)
            n += 1
            tp = i/n
            if n % tol_step == 0 
                err = 3*√(tp*(1-tp)/n)
            end
        end
        d.state.tp = tp
    elseif alg == :hc
        d.state.tp, d.state.tp_err = hcubature_inf((x)->pdf(d.untruncated,x),d.region.a, d.region.b)
    else
        error("Unknown algorithm $(alg)")
    end
    nothing
end

function compute_mean(d::TruncatedMvDistribution{D,R}; 
                    tol::Float64 = 10e-4, 
                    tol_step::Int = 10^5,
                    alg::Symbol = :hc) where {D,R}
    if alg == :mc
        @error("still not implemented")
    elseif alg == :hc
        d.state.μ, d.state.μ_err = hcubature_inf((x)->pdf(d,x)*x,d.region.a, d.region.b; maxevals = 10^6)
    else
        error("Unknown algorithm $(alg)")
    end
    nothing
end

function compute_cov(d::TruncatedMvDistribution{D,R}; 
                    tol::Float64 = 10e-4, 
                    tol_step::Int = 10^5,
                    alg::Symbol = :hc) where {D,R}
    if alg == :mc
        @error("still not implemented")
    elseif alg == :hc
        μ = mean(d)
        res = hcubature_inf((x)->pdf(d,x)*(x-μ)*(x-μ)',d.region.a, d.region.b; maxevals = 10^6)
        d.state.Σ, d.state.Σ_err = PDMat(0.5*(res[1] + res[1]')), res[2]
    else
        error("Unknown algorithm $(alg)")
    end
    nothing
end

function compute_moment(d::TruncatedMvDistribution{D,R}, k::Vector{Int};
                        alg::Symbol = :hc) where {D,R}

    if alg == :mc
        @error("still not implemented")
    elseif alg == :hc
        return hcubature_inf((x)->pdf(d,x)*prod(x.^k),d.region.a, d.region.b; maxevals = 10^6)[1]
    else
        error("Unknown algorithm $(alg)")
    end
    nothing
end