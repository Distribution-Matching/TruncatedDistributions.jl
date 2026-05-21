"""
A Box Truncated normal distribution with a naive implementation and representation in the state of the mean and covariance. Works well for very low dimensions (e.g. 2,3,4).
"""
const BasicBoxTruncatedMvNormal = TruncatedMvDistribution{MvNormal,BoxTruncationRegion,TruncatedMvDistributionSecondOrderState}

function BasicBoxTruncatedMvNormal( μₑ::Vector{Float64},
                                    Σₑ::PDMat, 
                                    a::Vector{Float64}, 
                                    b::Vector{Float64})
    d = MvNormal(μₑ,Σₑ)
    r = BoxTruncationRegion(a,b)
    TruncatedMvDistribution{MvNormal,BoxTruncationRegion,TruncatedMvDistributionSecondOrderState}(d,r)
end

"""
A Box Truncated normal distribution with a recursive moment computation implementation.
"""
const RecursiveMomentsBoxTruncatedMvNormal = TruncatedMvDistribution{MvNormal,BoxTruncationRegion,BoxTruncatedMvNormalRecursiveMomentsState}

function Base.show(io::IO, d::RecursiveMomentsBoxTruncatedMvNormal) 
    println(io, "Truncated Multivariate Normal, n=$(length(d))")
    println(io, "μ:\n $(d.untruncated.μ)")
    println(io, "Σ:")
        show(io, "text/plain", d.untruncated.Σ)
    println(io)
    println(io, "a:\n $(d.region.a)")
    println(io, "b:\n $(d.region.b)")
    println(io, "tp:\n $(round(tp(d), digits = 5))")
    println(io, "Mean:\n $(round.(mean(d), digits = 5))")
    println(io, "Cov:")
        show(io, "text/plain", round.(cov(d), digits = 5))

end



function RecursiveMomentsBoxTruncatedMvNormal(  μₑ::Vector{Float64},
                                                Σₑ::PDMat, 
                                                a::Vector{Float64}, 
                                                b::Vector{Float64};
                                                max_moment_levels::Int = 2)
    d = MvNormal(μₑ,Σₑ)
    r = BoxTruncationRegion(a,b)
    s = BoxTruncatedMvNormalRecursiveMomentsState(d,r,max_moment_levels)
    TruncatedMvDistribution{MvNormal,BoxTruncationRegion,BoxTruncatedMvNormalRecursiveMomentsState}(d,r,s)
end

