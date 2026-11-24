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
    cells = swath(pomdp, offset, angle)
    new_mapped = union(s.mapped, cells)
    new_cost_expended = s.cost_expended + action_cost(pomdp, a)

    t = s.t + 1
    last_seen = copy(s.last_seen)
    last_seen[cells] .= t

    # the jets change before the pass Within the precomputed path every policy sees the same jets. past it (or without one) they are stepped with rng.
    location_states, jets = s.location_states, s.jets

    if pomdp.jet_map !== nothing && !isempty(jets)

        if t + 1 <= length(pomdp.map_path)
            jets,location_states = pomdp.jet_path[t + 1], pomdp.map_path[t+ 1]

        else
            jets = step_jets(pomdp, jets, rng)
            location_states =jet_field(pomdp.jet_map, jets)

        end
    end

    return SatelliteState(offset, angle, location_states, new_cost_expended, new_mapped, t, last_seen, jets)
end

# one AR(1) step of the jet log-strengths, see JetMap in plume_map.jl
function ar1_step(z::Vector{Float64}, jet_T::Real, jet_σ::Real, rng::AbstractRNG)
    φ = exp(-1/jet_T)
    return φ .* z .+ (jet_σ*sqrt(1 - φ^2)) .* randn(rng, length(z))
end
step_jets(pomdp::SatellitePOMDP, z::Vector{Float64}, rng::AbstractRNG) = ar1_step(z, pomdp.jet_T, pomdp.jet_map.jet_σ, rng)
