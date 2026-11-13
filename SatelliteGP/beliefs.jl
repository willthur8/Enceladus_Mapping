struct SatelliteBeliefUpdater{P<:POMDPs.POMDP} <: Updater
    pomdp::P
end

# full posterior (μ, ν, S); S is Nq × Nq, so planning code uses location_mean_var instead
function location_posterior(b::SatelliteBelief)
    return query(b.location_belief)
end

# Posterior variance at every cell. 
function location_variance(pomdp::SatellitePOMDP, b::SatelliteBelief)
    gp = b.location_belief
    ν = get(pomdp.var_cache, gp.X, nothing)
    if ν === nothing
        ν = query_var(gp)
        entry_bytes = 8.0*(length(ν) + length(gp.X))
        if pomdp.var_cache_used + entry_bytes > pomdp.var_cache_bytes
            empty!(pomdp.var_cache)
            pomdp.var_cache_used = 0.0
        end
        pomdp.var_cache[gp.X] = ν
        pomdp.var_cache_used += entry_bytes
    end
    return ν
end

# posterior mean and variance at every cell, indexed by pomdp.query_index
location_mean_var(pomdp::SatellitePOMDP, b::SatelliteBelief) = (query_mean(b.location_belief), location_variance(pomdp, b))

# One draw of the noisy measurements a pass over cells returns, from the belief's posterior predictive
# N(μ_c, S_cc + σ_obs² I). 
function sample_observation(rng::AbstractRNG, pomdp::SatellitePOMDP, b::SatelliteBelief, cells::Vector{Int})
    isempty(cells) && return Float64[]
    μ_c, ν_c, S_c = query_cells(b.location_belief, pomdp.query_index[cells])
    return rand(rng, MvNormal(μ_c, Symmetric(S_c) + (pomdp.σ_obs^2 + 1e-6)*I)) # 1e-6 matches mvnrand's inflation
end

function Base.rand(rng::AbstractRNG, pomdp::SatellitePOMDP, b::SatelliteBelief)
    μ_post, ν_post, S_post = location_posterior(b)

    location_states = fill(NaN, pomdp.map_size)
    location_states[pomdp.in_map] = rand(rng, b.location_belief, μ_post, S_post)

    return SatelliteState(b.offset, b.angle, location_states, b.cost_expended, b.mapped)
end


function POMDPs.update(updater::SatelliteBeliefUpdater, b::SatelliteBelief, a::Symbol, o::Vector{Float64})
    ub = update_belief(updater.pomdp, b, a, o, updater.pomdp.rng)
    return ub
end


function update_belief(pomdp::P, b::SatelliteBelief, a::Symbol, o::Vector{Float64}, rng::RNG) where {P <: POMDPs.POMDP, RNG <: AbstractRNG}
    if isterminal(pomdp, b)
        return b
    end

    offset, angle = next_pose(pomdp, b.offset, b.angle, a)
    cells = swath(pomdp, offset, angle)

    
    σ²_n = pomdp.σ_obs^2
    X_samp = [convert_idx_2_coord(pomdp, c) for c in cells]
    f_posterior = posterior(b.location_belief, X_samp, o, fill(σ²_n, length(cells)))

    new_cost_expended = b.cost_expended + action_cost(pomdp, a)
    new_mapped = union(b.mapped, cells)

    return SatelliteBelief(offset, angle, f_posterior, new_cost_expended, new_mapped)
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

    return SatelliteBelief(pomdp.init_offset, pomdp.init_angle, location_belief, cost_expended, mapped)
end
