is_valuable(pomdp::SatellitePOMDP, v) = v > pomdp.value_threshold

# only deposition above reward_floor counts (fallout too thin for sampling is worth nothing)
cell_value(pomdp::SatellitePOMDP, c::Int, v) = max(v - pomdp.reward_floor, 0.0)

# E[max(f - t, 0)] for f ~ N(μ, ν) and t = reward_floor: (μ - t) Φ(z) + σ φ(z) with z = (μ - t)/σ
function expected_cell_value(pomdp::SatellitePOMDP, μ_post, ν_post, c::Int)
    q = pomdp.query_index[c]
    σ = sqrt(ν_post[q])
    m = μ_post[q] - pomdp.reward_floor
    z = m/σ
    return m*cdf(Normal(), z) + σ*pdf(Normal(), z)
end

# expected_cell_value for every cell at once, indexed by linear cell index (0 outside the map)
function expected_cell_values(pomdp::SatellitePOMDP, μ_post, ν_post)
    ev = zeros(length(pomdp.in_map))
    for c in findall(vec(pomdp.in_map))
        ev[c] = expected_cell_value(pomdp, μ_post, ν_post, c)
    end
    return ev
end

# expected reward of flying the pass at (offset, angle) next, counting only cells not already mapped
swath_gain(pomdp::SatellitePOMDP, ev, mapped, offset::Int, angle::Int) =
    sum((ev[c] for c in swath(pomdp, offset, angle) if !(c in mapped)), init=0.0)

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
    μ_init, ν_init = location_mean_var(pomdp, b)

    new_cells = setdiff(bp.mapped, b.mapped)
    r += sum((expected_cell_value(pomdp, μ_init, ν_init, c) for c in new_cells), init=0.0)

    if pomdp.info_weight > 0
        ν_post = location_variance(pomdp, bp)
        variance_reduction = (sum(ν_init) - sum(ν_post))
        r += pomdp.info_weight*variance_reduction
    end

    return r
end
