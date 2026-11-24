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
    t::Int # passes flown so far
    last_seen::Vector{Int} # pass each cell was last observed, by linear cell index (0 = never)
    jets::Vector{Float64} # jet log-strengths when the jets change (empty otherwise)
end

struct SatelliteBelief
    offset::Int
    angle::Int
    location_belief::GaussianProcess
    cost_expended::Float64
    mapped::Set{Int}
    t::Int
    last_seen::Vector{Int}
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
    info_weight::Float64 = 0.0 # weight on variance reduction in belief_reward (only when the jets do not change)
    jet_T::Float64 = Inf # passes for a jet's log-strength correlation to fall to 1/e; Inf = jets never change (v3)
    jet_map = nothing # JetMap (plume_map.jl) when jet_T is finite
    jet_path::Vector{Vector{Float64}} = Vector{Float64}[] # jet log-strengths at passes 0, 1, 2, ..., drawn once per trial
    map_path::Vector{Matrix{Float64}} = Matrix{Float64}[] # map at each of those passes, so every policy sees the same jets
    allow_stay::Bool = false # adds :stay, which flies the current track again
    var_cache::Dict{Any, Any} = Dict{Any, Any}() # posterior variance by observed points, see location_variance
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
