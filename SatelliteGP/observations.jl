# Observations

function generate_o(pomdp::SatellitePOMDP, s::SatelliteState, action::Symbol, sp::SatelliteState, rng::AbstractRNG)
    # observation at sp
    cells = swath(pomdp, sp.offset, sp.angle)
    return [sp.location_states[c] + pomdp.σ_obs*randn(rng) for c in cells]
end
