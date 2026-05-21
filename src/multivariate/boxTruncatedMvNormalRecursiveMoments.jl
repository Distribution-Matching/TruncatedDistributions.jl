function raw_moment(d::RecursiveMomentsBoxTruncatedMvNormal, k::Vector{Int})
    raw_moment(d.state, k)
end

function raw_moment_dict(d::RecursiveMomentsBoxTruncatedMvNormal)
    raw_moment_dict(d.state)
end

"""
    raw_moment_from_indices(d, indices)

Convenience accessor: convert a list of axis indices into a multi-index
`κ` (counting repeats) and return the corresponding raw moment.
`raw_moment_from_indices(d, [1, 2])` is equivalent to
`raw_moment(d, [1, 1, 0, …])` at `n ≥ 2`.
"""
function raw_moment_from_indices(d::RecursiveMomentsBoxTruncatedMvNormal, indices::Vector{Int})
    kappa = zeros(Int, length(d))
    for i in indices
        kappa[i] += 1
    end
    raw_moment(d, kappa)
end