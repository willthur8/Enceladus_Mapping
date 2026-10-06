is_valuable(pomdp::SatellitePOMDP, v) = v > pomdp.value_threshold

function prob_valuable(pomdp::SatellitePOMDP, μ_post, ν_post, c::Int)
    q = pomdp.query_index[c]
    return 1 - cdf(Normal(), (pomdp.value_threshold - μ_post[q])/sqrt(ν_post[q]))
end

function POMDPs.reward(pomdp::SatellitePOMDP, s::SatelliteState, a::Symbol, sp::SatelliteState)
    new_cells = setdiff(sp.mapped, s.mapped)
    return Float64(count(c -> is_valuable(pomdp, s.location_states[c]), new_cells))
end

function belief_reward(pomdp::SatellitePOMDP, b::SatelliteBelief, a::Symbol, bp::SatelliteBelief)
    r = 0.0

    if isterminal(pomdp, b)
        return r
    end

    μ_post, ν_post, S_post = location_posterior(bp)

    new_cells = setdiff(bp.mapped, b.mapped)
    r += sum((prob_valuable(pomdp, μ_post, ν_post, c) for c in new_cells), init=0.0)

    if pomdp.info_weight > 0
        μ_init, ν_init, S_init = location_posterior(b)
        variance_reduction = (sum(ν_init) - sum(ν_post))
        r += pomdp.info_weight*variance_reduction
    end

    return r
end
