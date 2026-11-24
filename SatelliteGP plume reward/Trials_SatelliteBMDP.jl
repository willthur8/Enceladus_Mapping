include("CustomGP.jl")
include("satellite_pomdp.jl")
include("belief_mdp.jl")
include("plume_map.jl")
include("plot_SatelliteBMDP.jl")
using Random
using POMDPs
using Statistics
using Distributions
using KernelFunctions
using MCTS
using Printf


################################################################################
# Problem Setup
################################################################################
function build_f_prior(in_map::BitMatrix, stats; informed_prior=true)
    s_map = informed_prior ? stats.std_map : fill(stats.field_std, size(in_map))
    mean_map = informed_prior ? stats.mean_map : fill(stats.field_mean, size(in_map))

    k = ScaledFieldKernel(with_lengthscale(SqExponentialKernel(), stats.ℓ), s_map)
    m(x) = mean_map[x[1], x[2]]
    X_query = build_query_points(in_map)
    KXqXq = K(X_query, X_query, k)
    return GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq)
end

function build_trial(rng::RNG, in_map::BitMatrix, p::PlumeParams, stats, f_prior; sun_azimuth=rand(rng)*360, use_earth=true,
                     w_min=0.2, σ_obs=0.05, total_budget=6.0, init_offset=1) where {RNG<:AbstractRNG}
    true_map = build_map(rng, in_map, p, stats.norm_const)
    weights = constraint_weights(in_map, p; sun_azimuth=sun_azimuth, use_earth=use_earth, w_min=w_min)
    return SatellitePOMDP(true_map=true_map, f_prior=f_prior, in_map=in_map, weights=weights, sun_azimuth=sun_azimuth,
                          init_offset=init_offset, σ_obs=σ_obs, value_threshold=stats.τ, cost_budget=total_budget, rng=rng)
end

available_reward(pomdp::SatellitePOMDP) = sum(cell_value(pomdp, c, pomdp.true_map[c]) for c in findall(pomdp.in_map))

function print_map(pomdp::SatellitePOMDP, M)
    for i in 1:pomdp.map_size[1]
        println(join((pomdp.in_map[i, j] ? string(Int(is_valuable(pomdp, M[i, j]))) : "." for j in 1:pomdp.map_size[2]), " "))
    end
end


################################################################################
# Policies
################################################################################
function get_gp_bmdp_policy(bmdp, rng, max_depth=5, queries=200)
    planner = solve(MCTS.DPWSolver(depth=max_depth, n_iterations=queries, rng=rng, k_state=0.5, k_action=10000.0, alpha_state=0.5), bmdp)
    return b -> action(planner, b)
end

function raster_policy(pomdp::SatellitePOMDP, b::SatelliteBelief)
    if b.cost_expended == 0
        return :stay
    end
    return :up in actions(pomdp, b) ? :up : :switch
end

function greedy_policy(pomdp::SatellitePOMDP, b::SatelliteBelief)
    μ_post, ν_post, S_post = location_posterior(b)
    function expected_gain(a)
        cells = swath(pomdp, next_pose(pomdp, b.offset, b.horizontal, a)...)
        return sum((expected_cell_value(pomdp, μ_post, ν_post, c) for c in cells if !(c in b.mapped)), init=0.0)
    end
    return argmax(expected_gain, actions(pomdp, b))
end


################################################################################
# Trials
################################################################################
function run_satellite_bmdp(rng::RNG, pomdp::SatellitePOMDP, updater::SatelliteBeliefUpdater, policy) where {RNG<:AbstractRNG}

    # s carries the true map, so observations and rewards here are real; the policy only sees b
    s = rand(rng, initialstate(pomdp))
    b = initialize_belief(updater, initialstate(pomdp))

    state_hist = [deepcopy(s)]
    gp_hist = [deepcopy(b.location_belief)]
    action_hist = Symbol[]
    reward_hist = Float64[]
    total_planning_time = 0.0

    total_reward = 0.0
    while !isterminal(pomdp, s)
        a, t = @timed policy(b)
        total_planning_time += t

        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
        b = update(updater, b, a, o)
        s = sp

        total_reward += r
        state_hist = vcat(state_hist, deepcopy(s))
        gp_hist = vcat(gp_hist, deepcopy(b.location_belief))
        action_hist = vcat(action_hist, a)
        reward_hist = vcat(reward_hist, r)
    end

    return total_reward, state_hist, gp_hist, action_hist, reward_hist, total_planning_time, length(reward_hist)
end

function solver_test_SatelliteBMDP(; n::Int=24, p::PlumeParams=PlumeParams(), informed_prior=true, use_earth=true, total_budget=6.0,
                                     σ_obs=0.05, init_offset::Int=1, seed::Int64=1234, num_trials=20, depth=5, queries=200)

    in_map = circle_mask(n)
    stats = field_statistics(MersenneTwister(seed), in_map, p)
    @printf("τ = %.2f  ℓ = %.2f cells  (%.1f km per cell)\n", stats.τ, stats.ℓ, 2p.R_map/n)
    f_prior = build_f_prior(in_map, stats; informed_prior=informed_prior)

    policy_names = ["random", "raster", "greedy", "gp_mcts_dpw"]
    rewards = Dict(name => Float64[] for name in policy_names)
    planning_times = Dict(name => 0.0 for name in policy_names)
    available = Float64[]

    for i in 1:num_trials
        rng = MersenneTwister(seed + i)

        pomdp = build_trial(rng, in_map, p, stats, f_prior; use_earth=use_earth, σ_obs=σ_obs, total_budget=total_budget, init_offset=init_offset)
        up = SatelliteBeliefUpdater(pomdp)
        bmdp = BeliefMDP(pomdp, up, belief_reward)

        policies = Dict("random" => b -> rand(rng, actions(pomdp, b)),
                        "raster" => b -> raster_policy(pomdp, b),
                        "greedy" => b -> greedy_policy(pomdp, b),
                        "gp_mcts_dpw" => get_gp_bmdp_policy(bmdp, rng, depth, queries))

        push!(available, available_reward(pomdp))
        for name in policy_names
            total_reward, state_hist, gp_hist, action_hist, reward_hist, planning_time, num_plans = run_satellite_bmdp(rng, pomdp, up, policies[name])
            push!(rewards[name], total_reward)
            planning_times[name] += planning_time/num_plans
        end
    end

    @printf("%d cells in map, %.1f reward available on average, budget %.0f sweeps\n", count(in_map), mean(available), total_budget)
    for name in policy_names
        R = rewards[name]
        @printf(" %-12s mean = %.2f ± %.2f   (%.3f s per action)\n", name, mean(R), std(R)/sqrt(length(R)), planning_times[name]/num_trials)
    end
    d = rewards["greedy"] .- rewards["raster"]
    @printf(" greedy − raster, paired: %.2f ± %.2f\n", mean(d), std(d)/sqrt(length(d)))

    return rewards
end

function plot_demo(; policy_name="greedy", n::Int=24, p::PlumeParams=PlumeParams(), informed_prior=true, use_earth=true, sun_azimuth=30.0,
                     total_budget=6.0, σ_obs=0.05, init_offset::Int=6, seed::Int64=1234, trial::Int=1, depth=5, queries=200, dir="figures")

    in_map = circle_mask(n)
    stats = field_statistics(MersenneTwister(seed), in_map, p)
    f_prior = build_f_prior(in_map, stats; informed_prior=informed_prior)

    rng = MersenneTwister(seed + trial)
    pomdp = build_trial(rng, in_map, p, stats, f_prior; sun_azimuth=sun_azimuth, use_earth=use_earth, σ_obs=σ_obs,
                        total_budget=total_budget, init_offset=init_offset)
    up = SatelliteBeliefUpdater(pomdp)

    policies = Dict("random" => b -> rand(rng, actions(pomdp, b)),
                    "raster" => b -> raster_policy(pomdp, b),
                    "greedy" => b -> greedy_policy(pomdp, b),
                    "gp_mcts_dpw" => get_gp_bmdp_policy(BeliefMDP(pomdp, up, belief_reward), rng, depth, queries))

    total_reward, state_hist, gp_hist, action_hist, reward_hist, planning_time, num_plans = run_satellite_bmdp(rng, pomdp, up, policies[policy_name])
    @printf("%s on trial %d: reward %.1f of %.1f available, actions %s\n", policy_name, trial, total_reward, available_reward(pomdp), join(action_hist, ", "))

    plot_trial(pomdp, state_hist, gp_hist, action_hist, reward_hist, trial, policy_name; dir=dir)
end


println("24 cell diameter south polar map:")
solver_test_SatelliteBMDP()

println("\nPlotting:")
plot_demo(policy_name="greedy", trial=1)
plot_demo(policy_name="raster", trial=1)
