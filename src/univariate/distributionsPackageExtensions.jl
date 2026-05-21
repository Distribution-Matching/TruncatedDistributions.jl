"
Return the k'th moment of a truncated normal distribution.
"
function moment(d::Truncated{Normal{T},Continuous}, k::Int) where T
    k == 0 ? one(T) : last(moments(d,k))
end

"
Compute the 1'st to k'th moment of a truncated normal distribution. Uses a recursive formula.
"
function moments(d::Truncated{Normal{T},Continuous}, k::Int) where T
    k == 0 && return 1
    m = Array{T}(undef,k+2) #Array of moments with index i being the i-2's moment 
                            #(treating the -1's moment as 0 and 0'ths moment as 1)
    m[1], m[2] = 0, 1 
    pars = params(d)
    μ, σ, σ² =pars[1], pars[2], pars[2]^2
    L, U = pars[3], pars[4]
    zL, zU = (L - μ)/σ, (U - μ)/σ 
    ϕL, ϕU = pdf.(Normal(),(zL, zU))
    ΦUL = cdf(Normal(),zU) - cdf(Normal(),zL)
    for i in 3:(k+2)
        kk = i-2
        # Boundary contributions: U^{kk-1} ϕU and L^{kk-1} ϕL. If a bound is
        # at ±∞ the Gaussian factor vanishes faster than the polynomial, so
        # the limit is 0, but the naïve product evaluates as Inf * 0 = NaN.
        # Skip the term when the corresponding ϕ is exactly zero.
        Uterm = iszero(ϕU) ? zero(T) : U^(kk-1) * ϕU
        Lterm = iszero(ϕL) ? zero(T) : L^(kk-1) * ϕL
        #recursive formula for kk'th moment as a function of the previous two moments (if kk=-1 it uses 0)
        m[i] = (kk-1)*σ²*m[i-2] + μ*m[i-1] - σ*(Uterm - Lterm)/ΦUL
    end
    return m[3:(k+2)]
end
