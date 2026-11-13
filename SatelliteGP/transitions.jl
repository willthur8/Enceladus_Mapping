function next_pose(pomdp::SatellitePOMDP, offset::Int, angle::Int, a::Symbol)
    n = num_offsets(pomdp)
    offset = clamp(offset + dir[a], 1, n)
    angle = angle + turn[a]

    # heading θ + 180° is the same track as heading θ with d flipped to -d
    if angle < 1 || angle > num_angles(pomdp)
        angle = mod1(angle, num_angles(pomdp))
        offset = n + 1 - offset
    end

    return offset, angle
end

function generate_s(pomdp::SatellitePOMDP, s::SatelliteState, a::Symbol, rng::RNG) where {RNG <: AbstractRNG}

    if isterminal(pomdp, s)
        return s
    end

    offset, angle = next_pose(pomdp, s.offset, s.angle, a)
    new_mapped = union(s.mapped, swath(pomdp, offset, angle))
    new_cost_expended = s.cost_expended + action_cost(pomdp, a)

    return SatelliteState(offset, angle, s.location_states, new_cost_expended, new_mapped)
end
