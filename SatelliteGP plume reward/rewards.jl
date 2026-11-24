is_valuable(pomdp::SatellitePOMDP, v) = v > pomdp.value_threshold

cell_value(pomdp::SatellitePOMDP, c::Int, v) = pomdp.weights[c] * max(v, 0.0)

# E[w max(f, 0)] for f ~ N(μ, ν)
function expected_cell_value(pomdp::SatellitePOMDP, μ_post, ν_post, c::Int)
    q = pomdp.query_index[c]
    σ = sqrt(ν_post[q])
    z = μ_post[q]/σ
    return pomdp.weights[c] * (μ_post[q]*cdf(Normal(), z) + σ*pdf(Normal(), z))
end

function POMDPs.reward(pomdp::SatellitePOMDP, s::SatelliteState, a::Symbol, sp::SatelliteState)
    new_cells = setdiff(sp.mapped, s.mapped)
    return sum((cell_value(pomdp, c, s.location_states[c]) for c in new_cells), init=0.0)
end

function belief_reward(pomdp::SatellitePOMDP, b::SatelliteBelief, a::Symbol, bp::SatelliteBelief)
    r = 0.0

    if isterminal(pomdp, b)
        return r
    end

    # score with the belief before the observation: same expectation, no observation noise
    μ_init, ν_init, S_init = location_posterior(b)

    new_cells = setdiff(bp.mapped, b.mapped)
    r += sum((expected_cell_value(pomdp, μ_init, ν_init, c) for c in new_cells), init=0.0)

    if pomdp.info_weight > 0
        μ_post, ν_post, S_post = location_posterior(bp)
        variance_reduction = (sum(ν_init) - sum(ν_post))
        r += pomdp.info_weight*variance_reduction
    end

    return r
end
