struct SatelliteBeliefUpdater{P<:POMDPs.POMDP} <: Updater
    pomdp::P
end

function location_posterior(b::SatelliteBelief)
    return b.location_belief.X == [] ? query_no_data(b.location_belief) : query(b.location_belief)
end

function Base.rand(rng::AbstractRNG, pomdp::SatellitePOMDP, b::SatelliteBelief)
    μ_post, ν_post, S_post = location_posterior(b)

    location_states = fill(NaN, pomdp.map_size)
    location_states[pomdp.in_map] = rand(rng, b.location_belief, μ_post, S_post)

    return SatelliteState(b.offset, b.horizontal, location_states, b.cost_expended, b.mapped)
end


function POMDPs.update(updater::SatelliteBeliefUpdater, b::SatelliteBelief, a::Symbol, o::Vector{Float64})
    ub = update_belief(updater.pomdp, b, a, o, updater.pomdp.rng)
    return ub
end


function update_belief(pomdp::P, b::SatelliteBelief, a::Symbol, o::Vector{Float64}, rng::RNG) where {P <: POMDPs.POMDP, RNG <: AbstractRNG}
    if isterminal(pomdp, b)
        return b
    end

    offset, horizontal = next_pose(pomdp, b.offset, b.horizontal, a)
    cells = swath(pomdp, offset, horizontal)

    # NOTE: σ_obs is the stddev whereas σ²_n is the variance. Julia uses σ_obs
    # for normal dist whereas our GP setup uses σ²_n
    σ²_n = pomdp.σ_obs^2
    X_samp = [convert_idx_2_coord(pomdp, c) for c in cells]
    f_posterior = posterior(b.location_belief, X_samp, o, fill(σ²_n, length(cells)))

    new_cost_expended = b.cost_expended + action_cost(pomdp, a)
    new_mapped = union(b.mapped, cells)

    return SatelliteBelief(offset, horizontal, f_posterior, new_cost_expended, new_mapped)
end


function POMDPs.initialize_belief(updater::SatelliteBeliefUpdater, d)
    return initial_belief_state(updater.pomdp, updater.pomdp.rng)
end

function POMDPs.initialize_belief(updater::SatelliteBeliefUpdater, d, rng::RNG) where {RNG <: AbstractRNG}
    return initial_belief_state(updater.pomdp, rng)
end

function initial_belief_state(pomdp::SatellitePOMDP, rng::RNG) where {RNG <: AbstractRNG}
    location_belief = pomdp.f_prior
    cost_expended = 0.0
    mapped = Set{Int}()

    return SatelliteBelief(pomdp.init_offset, pomdp.init_horizontal, location_belief, cost_expended, mapped)
end
