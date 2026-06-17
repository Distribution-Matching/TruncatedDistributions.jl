function raw_moment(d::RecursiveMomentsBoxTruncatedMvNormal, k::Vector{Int})
    # Validate against the cache contract before the lookup so callers get an
    # actionable message instead of a bare `KeyError`. The cache only holds
    # multi-indices with non-negative entries, length n, and total order
    # ≤ max_moment_levels (see `init_dicts`). This guard is on the user-facing
    # distribution method only; the hot `c_vector` path calls `raw_moment` on
    # child *states* directly and is left untouched.
    n = length(d)
    length(k) == n ||
        throw(DimensionMismatch("multi-index k has length $(length(k)); expected n = $n"))
    any(<(0), k) &&
        throw(ArgumentError("multi-index k must have non-negative entries; got k = $k"))
    L = d.state.max_moment_levels
    sum(k) ≤ L ||
        throw(ArgumentError(
            "raw_moment: requested multi-index k = $k has total order $(sum(k)), " *
            "which exceeds max_moment_levels = $L. Reconstruct the distribution " *
            "with `max_moment_levels = $(sum(k))` (or higher) to cache it via the " *
            "Kan–Robotti recursion, or call `moment(d, k)` for a direct (uncached) " *
            "cubature integral of E[∏ xᵢ^kᵢ]."))
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