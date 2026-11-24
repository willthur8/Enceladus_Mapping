struct SatelliteBeliefUpdater{P<:POMDPs.POMDP} <: Updater
    pomdp::P
end

# full posterior (μ, ν, S); S is Nq × Nq, so planning code uses location_mean_var instead
function location_posterior(b::SatelliteBelief)
    return query(b.location_belief; tq=b.t)
end

function cache_put!(pomdp::SatellitePOMDP, key, value, entry_bytes)

    if pomdp.var_cache_used + entry_bytes > pomdp.var_cache_bytes
        empty!(pomdp.var_cache)

        pomdp.var_cache_used = 0.0
    end

    pomdp.var_cache[key] = value
    pomdp.var_cache_used += entry_bytes
end

# Posterior variance at every cell. It depends on where and when the GP was observed, not.
# on values, so it is cached by shared passes
function location_variance(pomdp::SatellitePOMDP, b::SatelliteBelief)
    gp = b.location_belief
    gp.time === nothing || return location_var_terms(pomdp, b).ν
    ν = get(pomdp.var_cache, gp.X, nothing)
    if ν === nothing
        ν = query_var(gp)
        cache_put!(pomdp, gp.X, ν, 8.0*(length(ν) + length(gp.X)))
    end
    return ν
end

# variance and forecast terms at the belief's pass (query_var_terms), cached like location_variance
function location_var_terms(pomdp::SatellitePOMDP, b::SatelliteBelief)
    gp = b.location_belief
    gp.time === nothing && return (ν=location_variance(pomdp, b), Qss=Float64[], Qst=Float64[])
    key = (gp.X, gp.t_obs, b.t)
    terms = get(pomdp.var_cache, key, nothing)
    if terms === nothing
        terms = query_var_terms(gp; tq=b.t)
        cache_put!(pomdp, key, terms, 8.0*(length(terms.ν) + length(terms.Qss) + length(terms.Qst) + 2length(gp.X)))
    end
    return terms
end

# posterior mean and variance at every cell at the belief's pass, indexed by pomdp.query_index
location_mean_var(pomdp::SatellitePOMDP, b::SatelliteBelief) = (query_mean(b.location_belief; tq=b.t), location_variance(pomdp, b))




# Belief forecast for the changing jet plumes: mean and variance at each cell k passes after the belief's pass, with no new
# readings. g(Δt) = a + (1 - a) exp(-|Δt|/T). With e = exp(-k/T):
# μ_k = m + a u + e (μ - m - a u)
# ν_k = s² - [a² Qss + 2 a e (Qst - a Qss) + e² (Qtt - 2 a Qst + a² Qss)],   Qtt = s² - ν

struct Forecast
    μ::Vector{Float64}
    ν::Vector{Float64}
    m::Vector{Float64}
    s2::Vector{Float64}
    u::Vector{Float64}
    Qss::Vector{Float64}
    Qst::Vector{Float64}
    a::Float64
    T::Float64
end

function location_forecast(pomdp::SatellitePOMDP, b::SatelliteBelief)
    gp = b.location_belief
    terms = location_var_terms(pomdp, b)
    μ = query_mean(gp; tq=b.t)
    tk = gp.time
    tk === nothing && return Forecast(μ, terms.ν, gp.mXq, Float64[], Float64[], Float64[], Float64[], 0.0, Inf)
    u = tk.a > 0 ? query_mean_space(gp) : Float64[]
    return Forecast(μ, terms.ν, gp.mXq, diag(gp.KXqXq), u, terms.Qss, terms.Qst, tk.a, tk.T)
end

# forecast (mean, variance) of query cell q, k passes ahead
function forecast_cell(f::Forecast, q::Int, k::Int)
    isinf(f.T) && return f.μ[q], f.ν[q]
    e, a = exp(-k/f.T), f.a
    a == 0 && return f.m[q] + e*(f.μ[q] - f.m[q]), f.s2[q] - e^2*(f.s2[q] - f.ν[q])
    au = a*f.u[q]
    Qtt, Qss, Qst = f.s2[q] - f.ν[q], f.Qss[q], f.Qst[q]
    νk = f.s2[q] - (a^2*Qss + 2a*e*(Qst - a*Qss) + e^2*(Qtt - 2a*Qst + a^2*Qss))
    return f.m[q] + au + e*(f.μ[q] - f.m[q] - au), max(νk, eps())
end

# One draw of the noisy measurements a pass over cells returns
# N(μ_c, S_cc + σ_obs² I). 
function sample_observation(rng::AbstractRNG, pomdp::SatellitePOMDP, b::SatelliteBelief, cells::Vector{Int})
    isempty(cells) && return Float64[]
    μ_c, ν_c, S_c = query_cells(b.location_belief, pomdp.query_index[cells]; tq=b.t + 1) 
    return rand(rng, MvNormal(μ_c, Symmetric(S_c) + (pomdp.σ_obs^2 + 1e-6)*I)) 
end 

function Base.rand(rng::AbstractRNG, pomdp::SatellitePOMDP, b::SatelliteBelief)
    μ_post, ν_post, S_post = location_posterior(b)

    location_states = fill(NaN, pomdp.map_size)
    location_states[pomdp.in_map] = rand(rng, b.location_belief, μ_post, S_post)

    # sampled field has no jets, so it stays fixed if  stepped
    return SatelliteState(b.offset, b.angle, location_states, b.cost_expended, b.mapped, b.t, b.last_seen, Float64[])
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
    t = b.t + 1

    X_samp = [convert_idx_2_coord(pomdp, c) for c in cells]
    f_posterior = posterior(b.location_belief, X_samp, o, fill(σ²_n, length(cells)); t_samp=fill(t, length(cells)))

    new_cost_expended = b.cost_expended + action_cost(pomdp, a)
    new_mapped = union(b.mapped, cells)

    last_seen = copy(b.last_seen)
    last_seen[cells] .= t

    return SatelliteBelief(offset, angle, f_posterior, new_cost_expended, new_mapped, t, last_seen)
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

    return SatelliteBelief(pomdp.init_offset, pomdp.init_angle, location_belief, cost_expended, mapped, 0, zeros(Int, length(pomdp.in_map)))
end
