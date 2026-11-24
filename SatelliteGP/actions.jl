# Actions

const dir = Dict(:down => -1, :up => 1, :angle_down => 0, :angle_up => 0, :stay => 0)
const turn = Dict(:down => 0, :up => 0, :angle_down => -1, :angle_up => 1, :stay => 0)

POMDPs.actions(pomdp::SatellitePOMDP) = pomdp.allow_stay ? [:up, :down, :angle_up, :angle_down, :stay] : [:up, :down, :angle_up, :angle_down]
POMDPs.actions(pomdp::SatellitePOMDP, s::SatelliteState) = actions_possible_from_current(pomdp, s.offset, s.angle)
POMDPs.actions(pomdp::SatellitePOMDP, b::SatelliteBelief) = actions_possible_from_current(pomdp, b.offset, b.angle)

function actions_possible_from_current(pomdp::SatellitePOMDP, offset::Int, angle::Int)
    actions = [:up, :down, :angle_up, :angle_down]

    # the offset stops at the map edge, the angle wraps around so it is never removed
    if offset == 1
        deleteat!(actions, actions .== :down)
    elseif offset == num_offsets(pomdp)
        deleteat!(actions, actions .== :up)
    end
    pomdp.allow_stay && push!(actions, :stay)

    return actions
end

action_cost(pomdp::SatellitePOMDP, a::Symbol) = 1.0
