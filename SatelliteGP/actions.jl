# Actions

const dir = Dict(:stay => 0, :down => -1, :up => 1, :switch => 0)

POMDPs.actions(pomdp::SatellitePOMDP) = [:stay, :down, :up, :switch]
POMDPs.actions(pomdp::SatellitePOMDP, s::SatelliteState) = actions_possible_from_current(pomdp, s.offset, s.horizontal)
POMDPs.actions(pomdp::SatellitePOMDP, b::SatelliteBelief) = actions_possible_from_current(pomdp, b.offset, b.horizontal)

function actions_possible_from_current(pomdp::SatellitePOMDP, offset::Int, horizontal::Bool)
    actions = [:stay, :down, :up, :switch]
    n = horizontal ? pomdp.map_size[1] : pomdp.map_size[2]

    if offset == 1
        deleteat!(actions, actions .== :down)
    elseif offset == n
        deleteat!(actions, actions .== :up)
    end

    return actions
end

action_cost(pomdp::SatellitePOMDP, a::Symbol) = 1.0
