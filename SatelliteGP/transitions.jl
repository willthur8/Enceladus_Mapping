function next_pose(pomdp::SatellitePOMDP, offset::Int, horizontal::Bool, a::Symbol)
    if a == :switch
        return offset, !horizontal
    end

    n = horizontal ? pomdp.map_size[1] : pomdp.map_size[2]
    return clamp(offset + dir[a], 1, n), horizontal
end

function generate_s(pomdp::SatellitePOMDP, s::SatelliteState, a::Symbol, rng::RNG) where {RNG <: AbstractRNG}

    if isterminal(pomdp, s)
        return s
    end

    offset, horizontal = next_pose(pomdp, s.offset, s.horizontal, a)
    new_mapped = union(s.mapped, swath(pomdp, offset, horizontal))
    new_cost_expended = s.cost_expended + action_cost(pomdp, a)

    return SatelliteState(offset, horizontal, s.location_states, new_cost_expended, new_mapped)
end
