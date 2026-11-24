# Observations

function generate_o(pomdp::SatellitePOMDP, s::SatelliteState, action::Symbol, sp::SatelliteState, rng::AbstractRNG)
    # Remember you make the observation at sp NOT s
    cells = swath(pomdp, sp.offset, sp.horizontal)
    return [sp.location_states[c] + pomdp.σ_obs*randn(rng) for c in cells]
end
