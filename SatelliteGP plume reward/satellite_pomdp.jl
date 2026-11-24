using POMDPs
using POMDPTools: Deterministic
using Parameters, Random, Distributions, LinearAlgebra
using KernelFunctions


export
    SatellitePOMDP,
    SatelliteState,
    SatelliteBelief

struct SatelliteState
    offset::Int
    horizontal::Bool
    location_states::Matrix{Float64}
    cost_expended::Float64
    mapped::Set{Int}
end

struct SatelliteBelief
    offset::Int
    horizontal::Bool
    location_belief::GaussianProcess
    cost_expended::Float64
    mapped::Set{Int}
end


@with_kw mutable struct SatellitePOMDP <: POMDP{SatelliteState, Symbol, Vector{Float64}} # POMDP{State, Action, Observation}
    true_map::Matrix{Float64}
    f_prior::GaussianProcess
    in_map::BitMatrix                      = trues(size(true_map))
    weights::Matrix{Float64}               = ones(size(true_map)) # sun and Earth constraints, known in advance
    sun_azimuth::Float64                   = 0.0 # only used for plot labels
    map_size::Tuple{Int, Int}              = size(true_map)
    query_index::Matrix{Int}               = build_query_index(in_map)
    init_offset::Int                       = 1
    init_horizontal::Bool                  = true

    discount::Float64                      = 1.0
    σ_obs::Float64                         = 0.05 # stddev of the deposition measurement
    value_threshold::Float64               = 0.66 # plume area boundary, used for plotting
    cost_budget::Float64                   = 3.0
    info_weight::Float64                   = 0.0 # weight on variance reduction in belief_reward
    rng::AbstractRNG
end

POMDPs.isterminal(pomdp::SatellitePOMDP, s::SatelliteState) = s.cost_expended >= pomdp.cost_budget
POMDPs.isterminal(pomdp::SatellitePOMDP, b::SatelliteBelief) = b.cost_expended >= pomdp.cost_budget

function POMDPs.gen(pomdp::SatellitePOMDP, s::SatelliteState, a::Symbol, rng::RNG) where {RNG <: AbstractRNG}
    sp = generate_s(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp, rng)
    r = reward(pomdp, s, a, sp)

    return (sp=sp, o=o, r=r)
end

# discount
POMDPs.discount(pomdp::SatellitePOMDP) = pomdp.discount


include("states.jl")
include("actions.jl")
include("observations.jl")
include("beliefs.jl")
include("transitions.jl")
include("rewards.jl")
