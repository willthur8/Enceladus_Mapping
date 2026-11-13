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
    angle::Int
    location_states::Matrix{Float64}
    cost_expended::Float64
    mapped::Set{Int}
end

struct SatelliteBelief
    offset::Int
    angle::Int
    location_belief::GaussianProcess
    cost_expended::Float64
    mapped::Set{Int}
end


@with_kw mutable struct SatellitePOMDP <: POMDP{SatelliteState, Symbol, Vector{Float64}} # POMDP{State, Action, Observation}
    true_map::Matrix{Float64}
    f_prior::GaussianProcess
    in_map::BitMatrix = trues(size(true_map))
    map_size::Tuple{Int, Int}= size(true_map)
    query_index::Matrix{Int} = build_query_index(in_map)
    angle_step::Float64 = 30.0 # deg between neighboring track headings, must divide 180
    swath_width::Float64 = 1.0 # cells across the track
    swaths::Matrix{Vector{Int}} = build_swaths(in_map, angle_step, swath_width) # [offset, angle], rebuild if the two above change
    init_offset::Int = 1
    init_angle::Int = 1 # heading 0°

    discount::Float64= 1.0
    σ_obs::Float64 = 0.05 # stddev of the measurement
    value_threshold::Float64 = 0.66 # plume area boundary, used for plotting
    reward_floor::Float64 = 0.0 # deposition below this earns nothing: reward is max(f - reward_floor, 0)
    cost_budget::Float64 = 3.0
    info_weight::Float64 = 0.0 # weight on variance reduction in belief_reward
    var_cache::Dict{Any, Vector{Float64}} = Dict{Any, Vector{Float64}}() # posterior variance by observed points, see location_variance
    var_cache_bytes::Float64 = 0.5e9 # approximate memory the cache may use before it is cleared (per trial)
    var_cache_used::Float64  = 0.0 # approximate memory the cache uses now, see location_variance
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
