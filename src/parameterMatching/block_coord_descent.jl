"""
    block_coord_descent(μ̂, Σ̂, a, b; ...) -> (μ, Σ, hist)

Marginal-loss block coordinate descent for truncated MVN moment matching.
At each iteration we enumerate every candidate subset `S ⊂ {1,…,n}` whose
size is in `block_sizes` (default `[1, 2, 3]`), score it by the
*per-target* marginal residual

    score(S) = moment_loss(2D-truncN(μ_S, Σ_SS), μ̂_S, Σ̂_SS) / |S|(|S|+3)/2 ,

and pick the highest-scoring subset. We run a small EXP-KR-MVN sub-problem
at size `|S|` to fit the block to its 2D/3D marginal target, then write
the in-block `(μ_S, Σ_SS)` back into the full iterate while preserving the
correlations of the unchanged off-block coordinates (off-block `Σ_{i,k}`,
`k ∉ S` rescales by `σ_i_new / σ_i_old`).

The mix of `k = 1, 2, 3` updates emerges from the heuristic: a `k = 3`
block carries 9 targets, a `k = 1` block carries 2 targets, but they are
compared after dividing by `|S|(|S|+3)/2`, so the size that the current
iterate is most off-target in (per-target) gets picked. Concretely:
when the diagonals are roughly matched but a single triple has a strong
3-way correlation residual, `k = 3` wins; when one coordinate is far off
even after the warm-start, `k = 1` wins.

Arguments
---------
* `μ̂, Σ̂`                : target moments
* `a, b`                  : box bounds
* `μ_init, Σ_init`        : starting point (defaults to `(μ̂, Σ̂)`)
* `block_sizes`           : allowed sub-problem sizes; default `[1, 2, 3]`
* `max_iters`             : maximum number of single-block updates
* `inner_iters`           : LBFGS iterations inside each sub-problem
* `ftarget`               : early exit when monitored loss < this
* `monitor_full_loss`     : compute full n-dim loss after each update.
                            Set `false` for large n; the per-iter
                            sum-of-pair-residual is logged instead
* `monitor_every`         : only re-evaluate the monitor every k updates
                            (saves the n-dim KR call between updates)
* `verbose`               : print one line per update
* `exclude_recent`        : avoid re-picking a subset that was just
                            chosen within the last N iterations; prevents
                            getting stuck on one block
* `accept_by`             : `:marginal` accepts an update whenever the
                            block's own marginal residual goes down
                            (cheap; can oscillate at small n).
                            `:full` accepts only when the full n-dim
                            loss strictly improves (one extra full-loss
                            call per iter — fine at small n, expensive
                            at large n)
* `selection`             : `:greedy` picks the highest-scoring block;
                            `:softmax` samples `P(S) ∝ exp(score(S)/T)`
                            with temperature `softmax_T`. Softmax mode
                            avoids the "stuck on the same rejected block"
                            failure mode without needing the
                            tried-since-accept set
* `softmax_T`             : temperature for softmax sampling. Smaller
                            = more greedy; larger = more uniform
* `rng`                   : random number generator for softmax mode
* `proximal_λ`            : proximal regularization on the sub-problem.
                            Adds `λ * ||p - p₀||²` to the block's
                            objective, where `p₀` is the current iterate's
                            block params. Damps how far a single block
                            update can drag the joint iterate — useful
                            when marginal targets are biased (because
                            out-of-block truncation distorts marginals)
                            and exact marginal-matching overshoots the
                            true joint optimum. `0.0` (default) disables

Returns `(μ, Σ, hist, picks)` where `picks` is a vector of
`(k, S, accepted::Bool, score)` tuples — one per outer iteration —
documenting what the block-selection heuristic actually did.
"""
function block_coord_descent(μ̂::AbstractVector, Σ̂::AbstractMatrix,
                              a::AbstractVector, b::AbstractVector;
                              μ_init::Union{Nothing, AbstractVector} = nothing,
                              Σ_init::Union{Nothing, AbstractMatrix} = nothing,
                              block_sizes::Vector{Int} = [1, 2, 3],
                              max_iters::Int = 30,
                              inner_iters::Int = 15,
                              ftarget::Float64 = 1e-3,
                              monitor_full_loss::Bool = true,
                              monitor_every::Int = 1,
                              verbose::Bool = true,
                              exclude_recent::Int = 0,
                              accept_by::Symbol = :marginal,
                              selection::Symbol = :greedy,
                              softmax_T::Float64 = 1.0,
                              rng::AbstractRNG = Random.default_rng(),
                              proximal_λ::Float64 = 0.0)
    accept_by in (:marginal, :full) ||
        throw(ArgumentError("accept_by must be :marginal or :full"))
    selection in (:greedy, :softmax) ||
        throw(ArgumentError("selection must be :greedy or :softmax"))
    n = length(μ̂)
    μ = μ_init === nothing ? Vector{Float64}(μ̂) : Vector{Float64}(μ_init)
    Σ = Σ_init === nothing ? Matrix{Float64}(Σ̂) : Matrix{Float64}(Σ_init)

    # Build the candidate subset pool once. Silently skip sizes outside
    # 1..n so the caller can use a uniform `block_sizes = [1,2,3]` across
    # all dimensions without special-casing small n.
    pool = Vector{Vector{Int}}()
    for k in block_sizes
        (1 <= k <= n) || continue
        for S in combinations(1:n, k)
            push!(pool, collect(S))
        end
    end

    hist  = Float64[]
    picks = Tuple{Int, Vector{Int}, Bool, Float64}[]
    L0 = _monitor_loss(μ, Σ, a, b, μ̂, Σ̂, monitor_full_loss, pool)
    push!(hist, L0)
    verbose && @info @sprintf("[BCD start] n=%d  candidates=%d  loss=%.3e  (%s)",
                              n, length(pool), L0,
                              monitor_full_loss ? "full" : "Σmarg")
    if L0 < ftarget
        return μ, Σ, hist, picks
    end

    # `tried_since_accept` accumulates blocks that have been rejected since
    # the last successful update; we don't reconsider them until something
    # else moves the iterate. If every candidate ends up in here we've hit
    # a multi-block local minimum and stop.
    tried_since_accept = Set{Vector{Int}}()
    recent = Vector{Vector{Int}}()
    t_start = time()

    for iter in 1:max_iters
        S, score = _pick_block(μ, Σ, a, b, μ̂, Σ̂, pool,
                                recent, tried_since_accept,
                                selection, softmax_T, rng)
        if S === nothing
            verbose && @info "[BCD] all blocks rejected since last accept — local min"
            break
        end

        accepted = _block_update_marginal!(μ, Σ, S, a, b, μ̂, Σ̂,
                                            inner_iters, accept_by,
                                            proximal_λ)
        push!(picks, (length(S), copy(S), accepted, score))
        if accepted
            empty!(tried_since_accept)
            push!(recent, S)
            if length(recent) > exclude_recent
                popfirst!(recent)
            end
        else
            push!(tried_since_accept, S)
        end

        if iter % monitor_every == 0 || iter == max_iters
            L = _monitor_loss(μ, Σ, a, b, μ̂, Σ̂, monitor_full_loss, pool)
            push!(hist, L)
            if verbose
                el = time() - t_start
                @info @sprintf("[BCD iter %2d] k=%d S=%s  score=%.2e  loss=%.3e  %s  %.2fs",
                               iter, length(S), _Sshort(S), score, L,
                               accepted ? "accept" : "reject", el)
            end
            if L < ftarget
                break
            end
        else
            if verbose
                @info @sprintf("[BCD iter %2d] k=%d S=%s  score=%.2e  %s",
                               iter, length(S), _Sshort(S), score,
                               accepted ? "accept" : "reject")
            end
        end
    end
    return μ, Σ, hist, picks
end

_Sshort(S::Vector{Int}) = "[" * join(S, ",") * "]"

# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

function _pick_block(μ, Σ, a, b, μ̂, Σ̂,
                      pool::Vector{Vector{Int}},
                      recent::Vector{Vector{Int}},
                      tried::Set{Vector{Int}},
                      selection::Symbol,
                      softmax_T::Float64,
                      rng::AbstractRNG)
    if selection === :greedy
        best_score = -Inf
        best_S     = nothing
        for S in pool
            S in recent && continue
            S in tried  && continue
            s = _block_marginal_score(μ, Σ, S, a, b, μ̂, Σ̂)
            if s > best_score
                best_score = s
                best_S = S
            end
        end
        return best_S, best_score
    else  # :softmax
        eligible   = Vector{Vector{Int}}()
        raw_scores = Float64[]
        for S in pool
            S in recent && continue
            S in tried  && continue
            push!(eligible, S)
            push!(raw_scores, _block_marginal_score(μ, Σ, S, a, b, μ̂, Σ̂))
        end
        isempty(eligible) && return nothing, NaN
        # Stabilised softmax in log-space.
        smax = maximum(raw_scores)
        w = exp.((raw_scores .- smax) ./ softmax_T)
        w ./= sum(w)
        idx = _sample_categorical(rng, w)
        return eligible[idx], raw_scores[idx]
    end
end

function _sample_categorical(rng::AbstractRNG, p::AbstractVector{Float64})
    u = rand(rng)
    acc = 0.0
    for (i, pi) in enumerate(p)
        acc += pi
        if u <= acc
            return i
        end
    end
    return length(p)
end

# Per-target marginal residual: marginal moment loss of the k-D truncated
# normal with the current iterate's (μ_S, Σ_SS), divided by k(k+3)/2 so
# that scores across different block sizes are comparable.
function _block_marginal_score(μ, Σ, S::Vector{Int}, a, b, μ̂, Σ̂)
    L = _marginal_loss_k(μ, Σ, S, a, b, μ̂, Σ̂)
    k = length(S)
    return L / (k * (k + 3) / 2)
end

function _marginal_loss_k(μ, Σ, S::Vector{Int}, a, b, μ̂, Σ̂)
    μ_S  = μ[S]
    Σ_SS = _symmetrize_with_jitter(Σ[S, S])
    d = RecursiveMomentsBoxTruncatedMvNormal(μ_S, PDMat(Σ_SS), a[S], b[S])
    return moment_loss(d, μ̂[S], Σ̂[S, S])
end

function _symmetrize_with_jitter(A::AbstractMatrix)
    M = 0.5 .* (Matrix(A) .+ Matrix(A)')
    M .+= eps(Float64) * (tr(M) + 1.0) * Matrix{Float64}(I, size(M))
    return M
end

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

function _monitor_loss(μ, Σ, a, b, μ̂, Σ̂, monitor_full::Bool,
                       pool::Vector{Vector{Int}})
    if monitor_full
        return _full_moment_loss(μ, Σ, a, b, μ̂, Σ̂)
    else
        L = 0.0
        for S in pool
            L += _marginal_loss_k(μ, Σ, S, a, b, μ̂, Σ̂)
        end
        return L
    end
end

function _full_moment_loss(μ, Σ, a, b, μ̂, Σ̂)
    Σsym = _symmetrize_with_jitter(Σ)
    d = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σsym), a, b)
    return moment_loss(d, μ̂, Σ̂)
end

# ---------------------------------------------------------------------------
# Block sub-problem
# ---------------------------------------------------------------------------

# Solve the k-D sub-problem on block S via EXP-KR-MVN with the existing
# vector_fg_true_loss pipeline. Write back to (μ, Σ) only if the marginal
# residual on S strictly improves.
function _block_update_marginal!(μ::Vector{Float64}, Σ::Matrix{Float64},
                                  S::Vector{Int}, a, b,
                                  μ̂::AbstractVector, Σ̂::AbstractMatrix,
                                  inner_iters::Int,
                                  accept_by::Symbol,
                                  proximal_λ::Float64)
    σ_old = sqrt.(diag(Σ))
    μ_S0  = μ[S]
    Σ_SS0 = _symmetrize_with_jitter(Σ[S, S])

    a_S = a[S]; b_S = b[S]
    μ̂_S = Vector{Float64}(μ̂[S])
    Σ̂_SS = Matrix{Float64}(Σ̂[S, S])

    # Build candidate (μ_new, Σ_new) for the block.
    local μ_new, Σ_new, L_marg_after
    if length(S) == 1
        μ_new1, σ²_new1 = _block_update_1d(μ_S0[1], Σ_SS0[1, 1],
                                            a_S[1], b_S[1],
                                            μ̂_S[1], Σ̂_SS[1, 1],
                                            inner_iters)
        μ_new = [μ_new1]
        Σ_new = reshape([σ²_new1], 1, 1)
        L_marg_after = _try_1d_loss(μ_new1, σ²_new1, a_S[1], b_S[1],
                                     μ̂_S[1], Σ̂_SS[1, 1])
    else
        p0 = make_param_vec_from_μ_Σ(μ_S0, Σ_SS0)
        fg! = if proximal_λ > 0
            p_anchor = copy(p0)
            (F, G, p) -> _fg_with_proximal(F, G, p, a_S, b_S, μ̂_S, Σ̂_SS,
                                            p_anchor, proximal_λ)
        else
            (F, G, p) -> vector_fg_true_loss(F, G, p, a_S, b_S, μ̂_S, Σ̂_SS)
        end
        opts = Optim.Options(iterations = inner_iters,
                             time_limit = 30.0,
                             show_trace = false)
        res = try
            optimize(Optim.only_fg!(fg!), p0, LBFGS(), opts)
        catch
            return false
        end
        μ_new, Σ_new = make_μ_Σ_from_param_vec(res.minimizer)
        # Strip the proximal term back out so acceptance compares pure loss.
        L_marg_after = if proximal_λ > 0
            res.minimum - proximal_λ * sum(abs2, res.minimizer .- p0)
        else
            res.minimum
        end
    end

    # Acceptance test.
    if accept_by === :marginal
        L_marg_before = _marginal_loss_k(μ, Σ, S, a, b, μ̂, Σ̂)
        if !(isfinite(L_marg_after) && L_marg_after < L_marg_before)
            return false
        end
    else  # :full
        L_full_before = _full_moment_loss(μ, Σ, a, b, μ̂, Σ̂)
        # Apply tentatively into a scratch copy.
        μ_try = copy(μ); Σ_try = copy(Σ)
        σ_try = copy(σ_old)
        for (idx, i) in enumerate(S)
            σ_try[i] = sqrt(max(Σ_new[idx, idx], eps()))
        end
        _write_block!(μ_try, Σ_try, μ_new, Σ_new, S, σ_old, σ_try)
        L_full_after = _full_moment_loss(μ_try, Σ_try, a, b, μ̂, Σ̂)
        if !(isfinite(L_full_after) && L_full_after < L_full_before)
            return false
        end
    end

    # Commit.
    σ_new = copy(σ_old)
    for (idx, i) in enumerate(S)
        σ_new[i] = sqrt(max(Σ_new[idx, idx], eps()))
    end
    _write_block!(μ, Σ, μ_new, Σ_new, S, σ_old, σ_new)
    return true
end

# 1D LBFGS fit: same as warm_start_diagonal but for a single axis given
# a current iterate.
function _block_update_1d(μ0::Real, σ²0::Real,
                          ai::Real, bi::Real,
                          μ̂i::Real, Σ̂ii::Real,
                          iters::Int)
    function f(p)
        μ_i  = p[1]
        σ²_i = exp(p[2])
        d_i  = truncated(Normal(μ_i, sqrt(σ²_i)), ai, bi)
        m    = moments(d_i, 2)
        return (m[1] - μ̂i)^2 + (m[2] - m[1]^2 - Σ̂ii)^2
    end
    p0  = [Float64(μ0), log(max(σ²0, eps()))]
    res = optimize(f, p0, LBFGS(),
                   Optim.Options(iterations = iters);
                   autodiff = :forward)
    return res.minimizer[1], exp(res.minimizer[2])
end

function _try_1d_loss(μ::Real, σ²::Real, a::Real, b::Real,
                       μ̂::Real, Σ̂ii::Real)
    d = truncated(Normal(μ, sqrt(max(σ², eps()))), a, b)
    m = moments(d, 2)
    return (m[1] - μ̂)^2 + (m[2] - m[1]^2 - Σ̂ii)^2
end

# `vector_fg_true_loss` plus a proximal `λ ‖p - p_anchor‖²` term.
function _fg_with_proximal(F, G, p::Vector{Float64},
                            a, b,
                            μ̂_S::Vector{Float64}, Σ̂_SS::Matrix{Float64},
                            p_anchor::Vector{Float64}, λ::Float64)
    val = vector_fg_true_loss(F, G, p, a, b, μ̂_S, Σ̂_SS)
    if G !== nothing
        @inbounds for i in eachindex(G)
            G[i] += 2λ * (p[i] - p_anchor[i])
        end
    end
    if F !== nothing
        return val + λ * sum(abs2, p .- p_anchor)
    end
    return nothing
end

function _write_block!(μ::Vector{Float64}, Σ::Matrix{Float64},
                       μ_new::AbstractVector, Σ_new::AbstractMatrix,
                       S::Vector{Int},
                       σ_old::AbstractVector,
                       σ_new::AbstractVector)
    n = length(μ)
    in_S = falses(n); in_S[S] .= true
    @inbounds for (idx, i) in enumerate(S)
        μ[i] = μ_new[idx]
    end
    @inbounds for ai in 1:length(S), aj in 1:length(S)
        i, j = S[ai], S[aj]
        Σ[i, j] = Σ_new[ai, aj]
    end
    @inbounds for ai in 1:length(S)
        i = S[ai]
        scale = σ_new[i] / (σ_old[i] + eps())
        for k in 1:n
            if !in_S[k]
                Σ[i, k] *= scale
                Σ[k, i] = Σ[i, k]
            end
        end
    end
    return
end
