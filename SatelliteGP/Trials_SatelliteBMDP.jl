include("CustomGP.jl")
include("satellite_pomdp.jl")
include("belief_mdp.jl")
include("plume_map.jl")
# define SATELLITE_NO_PLOTS before including this file to skip loading Plots (cluster runs do not plot)
isdefined(Main, :SATELLITE_NO_PLOTS) || include("plot_SatelliteBMDP.jl")
using Random
using POMDPs
using Statistics
using Distributions
using KernelFunctions
using MCTS
using POMDPTools: RandomSolver
using LinearAlgebra
using Printf


function build_f_prior(in_map::BitMatrix, stats; informed_prior=true)
    s_map = informed_prior ? stats.std_map : fill(stats.field_std, size(in_map))
    mean_map = informed_prior ? stats.mean_map : fill(stats.field_mean, size(in_map))

    k = ScaledFieldKernel(with_lengthscale(SqExponentialKernel(), stats.ℓ), s_map)
    m(x) = mean_map[x[1], x[2]]
    X_query = build_query_points(in_map)
    KXqXq = K(X_query, X_query, k)
    return GaussianProcess(m, μ(X_query, m), k, X_query, KXqXq)
end

# Reward per newly mapped cell is max(f - t, 0).
# threshold_reward=true (default): t = τ, the plume threshold, so only deposition above it counts. With t = 0 the
#   background fallout makes every pass about equally valuable and a raster is already near optimal.
# angle_step: degrees between neighboring headings, must divide 180. :angle_up/:angle_down still turn one step per action.

function build_trial(rng::RNG, in_map::BitMatrix, p::PlumeParams, stats, f_prior; threshold_reward=true, σ_obs=0.05,
                     total_budget=10.0, init_offset=1, angle_step=30.0, var_cache_bytes=0.5e9) where {RNG<:AbstractRNG}
    isinteger(180/angle_step) || error("angle_step = $angle_step does not divide 180")
    true_map = build_map(rng, in_map, p, stats.norm_const)
    return SatellitePOMDP(true_map=true_map, f_prior=f_prior, in_map=in_map, angle_step=Float64(angle_step),
                          init_offset=init_offset, σ_obs=σ_obs, value_threshold=stats.τ, reward_floor=threshold_reward ? stats.τ : 0.0,
                          cost_budget=total_budget, var_cache_bytes=Float64(var_cache_bytes), rng=rng)
end

# budget that keeps the share of the map one run can cover the same as v3 (10 passes on the 24-cell map, 1-cell swath)
scaled_budget(n::Int) = round(Int, 10*n/24)

available_reward(pomdp::SatellitePOMDP) = sum(cell_value(pomdp, c, pomdp.true_map[c]) for c in findall(vec(pomdp.in_map)))

function print_map(pomdp::SatellitePOMDP, M)
    for i in 1:pomdp.map_size[1]
        println(join((pomdp.in_map[i, j] ? string(Int(is_valuable(pomdp, M[i, j]))) : "." for j in 1:pomdp.map_size[2]), " "))
    end
end



# leaf- how MCTS values a new node
#   :greedy  greedy_rollout_value, deterministic and cheap (default)
#   :random  MCTS.jl's default random rollout to the end of the budget, one GP sample per step
#   :zero    no lookahead past the tree
# enable_action_pw=false gives every node all legal actions at once; with only 4 actions there is nothing to widen
# exploration_constant=20: per-pass rewards are ~5 and returns ~50, so MCTS.jl's default of 1 barely explores.
# 20-map sweep (seed 1234): c = 1, 5, 20 gave MCTS − greedy = +0.8, +1.9, +2.5 at 200 iterations (sweep_mcts.jl)
function get_gp_bmdp_policy(bmdp, rng, max_depth=5, queries=200; leaf::Symbol=:greedy, exploration_constant=20.0,
                            k_state=0.5, alpha_state=0.5, enable_action_pw=false)
    estimate = leaf == :greedy ? (m, b, d) -> greedy_rollout_value(m.pomdp, b) :
               leaf == :random ? RolloutEstimator(RandomSolver(rng)) :
               leaf == :zero   ? 0.0 : error("leaf must be :greedy, :random, or :zero")
    planner = solve(MCTS.DPWSolver(depth=max_depth, n_iterations=queries, rng=rng, exploration_constant=exploration_constant,
                                   k_state=k_state, alpha_state=alpha_state, k_action=10000.0, enable_action_pw=enable_action_pw,
                                   estimate_value=estimate), bmdp)
    return b -> action(planner, b)
end

# sweep the offsets at a fixed heading, rotate once the edge is reached
function raster_policy(pomdp::SatellitePOMDP, b::SatelliteBelief)
    return :up in actions(pomdp, b) ? :up : :angle_up
end

function greedy_policy(pomdp::SatellitePOMDP, b::SatelliteBelief)
    ev = expected_cell_values(pomdp, location_mean_var(pomdp, b)...)
    return argmax(a -> swath_gain(pomdp, ev, b.mapped, next_pose(pomdp, b.offset, b.angle, a)...), actions(pomdp, b))
end

# MCTS leaf value: expected reward of flying greedy passes until the budget runs out
function greedy_rollout_value(pomdp::SatellitePOMDP, b::SatelliteBelief)
    isterminal(pomdp, b) && return 0.0
    ev = expected_cell_values(pomdp, location_mean_var(pomdp, b)...)
    offset, angle, cost, mapped = b.offset, b.angle, b.cost_expended, copy(b.mapped)
    γ, value, t = discount(pomdp), 0.0, 0
    while cost < pomdp.cost_budget
        a = argmax(a -> swath_gain(pomdp, ev, mapped, next_pose(pomdp, offset, angle, a)...),
                   actions_possible_from_current(pomdp, offset, angle))
        offset, angle = next_pose(pomdp, offset, angle, a)
        value += γ^t * swath_gain(pomdp, ev, mapped, offset, angle)
        union!(mapped, swath(pomdp, offset, angle))
        cost += action_cost(pomdp, a)
        t += 1
    end
    return value
end



# record_history=false skips the copies of every state and belief, which only plotting needs. At large n, belief
# copies  memory (about 1 GB per run at n = 72); state_hist and gp_hist are then returned empty.
function run_satellite_bmdp(rng::RNG, pomdp::SatellitePOMDP, updater::SatelliteBeliefUpdater, policy; record_history=true) where {RNG<:AbstractRNG}

    # s carries the true map, so observations and rewards here are real; the policy only sees b
    s = rand(rng, initialstate(pomdp))
    b = initialize_belief(updater, initialstate(pomdp))

    state_hist = record_history ? [deepcopy(s)] : SatelliteState[]
    gp_hist = record_history ? [deepcopy(b.location_belief)] : GaussianProcess[]
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
        if record_history
            push!(state_hist, deepcopy(s))
            push!(gp_hist, deepcopy(b.location_belief))
        end
        push!(action_hist, a)
        push!(reward_hist, r)
    end

    return total_reward, state_hist, gp_hist, action_hist, reward_hist, total_planning_time, length(reward_hist)
end

# policies: any of "random", "raster", "greedy", "gp_mcts_dpw" (leaf from mcts_leaf), "gp_mcts_random"

function solver_test_SatelliteBMDP(; n::Int=24, p::PlumeParams=PlumeParams(), informed_prior=true, 
    threshold_reward=true, total_budget=10.0,
    σ_obs=0.05, init_offset::Int=1, seed::Int64=1234, num_trials=20,
    depth=5, queries=200,
    policy_names=["random", "raster", "greedy", "gp_mcts_dpw"],
    mcts_leaf::Symbol=:greedy,
    exploration_constant=20.0, angle_step=30.0, 
    max_parallel::Int=Threads.nthreads(),
    var_cache_bytes=0.5e9, stats_cache_dir="cache", on_trial=nothing, verbose=true)

    in_map = circle_mask(n)
    stats = cached_field_statistics(seed, in_map, p; dir=stats_cache_dir)
    verbose && @printf("τ = %.2f  ℓ = %.2f cells  (%.1f km per cell)\n", stats.τ, stats.ℓ, 2p.R_map/n)
    f_prior = build_f_prior(in_map, stats; informed_prior=informed_prior)

    R = zeros(num_trials, length(policy_names))# total reward per trial and policy
    T = zeros(num_trials, length(policy_names))# planning seconds per action
    W = zeros(num_trials, length(policy_names))# wall seconds per run (planning plus belief updates)
    available = zeros(num_trials)

    Threads.nthreads() > 1 && BLAS.set_num_threads(1) # one BLAS thread per trial thread
    slots = Base.Semaphore(max(max_parallel, 1))
    @sync for i in 1:num_trials
        Threads.@spawn begin
            Base.acquire(slots)
            try
                rng = MersenneTwister(seed + i)
                rand(rng) # v3 used one random number here before building the map; drawing it keeps each seed's maps identical to v3

                pomdp = build_trial(rng, in_map, p, stats, f_prior; threshold_reward=threshold_reward, σ_obs=σ_obs, total_budget=total_budget,
                                    init_offset=init_offset, angle_step=angle_step, var_cache_bytes=var_cache_bytes)
                up = SatelliteBeliefUpdater(pomdp)
                bmdp = BeliefMDP(pomdp, up, belief_reward)
                mcts(leaf) = get_gp_bmdp_policy(bmdp, rng, depth, queries; leaf=leaf, exploration_constant=exploration_constant)

                policies = Dict("random" => b -> rand(rng, actions(pomdp, b)),
                                "raster" => b -> raster_policy(pomdp, b),
                                "greedy" => b -> greedy_policy(pomdp, b),
                                "gp_mcts_dpw" => mcts(mcts_leaf),
                                "gp_mcts_random" => mcts(:random))

                available[i] = available_reward(pomdp)
                for (j, name) in enumerate(policy_names)
                    W[i, j] = @elapsed begin
                        total_reward, state_hist, gp_hist, action_hist, reward_hist, planning_time, num_plans =
                            run_satellite_bmdp(rng, pomdp, up, policies[name]; record_history=false)
                    end
                    R[i, j] = total_reward
                    T[i, j] = planning_time/num_plans
                end
                on_trial === nothing || on_trial((trial=i, available=available[i], rewards=R[i, :], sec_per_action=T[i, :], wall_seconds=W[i, :]))
            finally
                Base.release(slots)
            end
        end
    end

    if verbose
        @printf("%d cells in map, %.1f reward available on average, budget %.0f passes, %g° heading steps\n",
                count(in_map), mean(available), total_budget, angle_step)
        for (j, name) in enumerate(policy_names)
            frac = R[:, j] ./ available
            @printf(" %-15s mean = %.2f ± %.2f   fraction of available = %.3f ± %.3f   (%.3f s per action)\n", name,
                    mean(R[:, j]), std(R[:, j])/sqrt(num_trials), mean(frac), std(frac)/sqrt(num_trials), mean(T[:, j]))
        end
        # same maps for every policy, so compare per map
        col(name) = R[:, findfirst(==(name), policy_names)]
        for (x, y) in [("greedy", "raster"), ("gp_mcts_dpw", "greedy"), ("gp_mcts_random", "greedy"), ("gp_mcts_dpw", "gp_mcts_random")]
            if x in policy_names && y in policy_names
                d = col(x) .- col(y)
                @printf(" %s − %s, paired: %.2f ± %.2f  (wins/ties/losses %d/%d/%d)\n", x, y, mean(d), std(d)/sqrt(num_trials),
                        count(>(1e-9), d), count(v -> abs(v) <= 1e-9, d), count(<(-1e-9), d))
            end
        end
    end

    return Dict(name => R[:, j] for (j, name) in enumerate(policy_names))
end

function plot_demo(; policy_name="greedy", n::Int=24, p::PlumeParams=PlumeParams(), informed_prior=true, threshold_reward=true,
                     total_budget=10.0, σ_obs=0.05, init_offset::Int=6, seed::Int64=1234, trial::Int=1, depth=5, queries=200, dir="figures",
                     mcts_leaf::Symbol=:greedy, exploration_constant=20.0, angle_step=30.0, stats_cache_dir="cache")

    in_map = circle_mask(n)
    stats = cached_field_statistics(seed, in_map, p; dir=stats_cache_dir)
    f_prior = build_f_prior(in_map, stats; informed_prior=informed_prior)

    rng = MersenneTwister(seed + trial)
    pomdp = build_trial(rng, in_map, p, stats, f_prior; threshold_reward=threshold_reward, σ_obs=σ_obs,
                        total_budget=total_budget, init_offset=init_offset, angle_step=angle_step)
    up = SatelliteBeliefUpdater(pomdp)

    policies = Dict("random" => b -> rand(rng, actions(pomdp, b)),
                    "raster" => b -> raster_policy(pomdp, b),
                    "greedy" => b -> greedy_policy(pomdp, b),
                    "gp_mcts_dpw" => get_gp_bmdp_policy(BeliefMDP(pomdp, up, belief_reward), rng, depth, queries;
                                                        leaf=mcts_leaf, exploration_constant=exploration_constant))

    total_reward, state_hist, gp_hist, action_hist, reward_hist, planning_time, num_plans = run_satellite_bmdp(rng, pomdp, up, policies[policy_name])
    @printf("%s on trial %d: reward %.1f of %.1f available, actions %s\n", policy_name, trial, total_reward, available_reward(pomdp), join(action_hist, ", "))

    plot_trial(pomdp, state_hist, gp_hist, action_hist, reward_hist, trial, policy_name; dir=dir)
end


# define SATELLITE_NO_MAIN before including this file to load the functions without running the experiments
if !isdefined(Main, :SATELLITE_NO_MAIN)

println("24 cell diameter south polar map:")
solver_test_SatelliteBMDP()

println("\nPlotting:")
plot_demo(policy_name="greedy", trial=1)
plot_demo(policy_name="gp_mcts_dpw", trial=1, depth= 10)

end
