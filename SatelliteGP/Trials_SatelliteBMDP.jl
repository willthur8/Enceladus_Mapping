include("CustomGP.jl")
include("satellite_pomdp.jl")
include("belief_mdp.jl")
using Random
using POMDPs
using Statistics
using Distributions
using KernelFunctions
using MCTS
using Printf


################################################################################
# Map Building
################################################################################
fade(t) = t^3 * (t * (6t - 15) + 10)

function perlin(x, y, G)
    i, j = floor(Int, x), floor(Int, y)
    u, v = x - i, y - j
    g(a, b) = G[i+a+1, j+b+1][1] * (u - a) + G[i+a+1, j+b+1][2] * (v - b)
    lo = g(0, 0) + fade(u) * (g(1, 0) - g(0, 0))
    hi = g(0, 1) + fade(u) * (g(1, 1) - g(0, 1))
    return lo + fade(v) * (hi - lo)
end

function raw_field(rng::RNG, map_size::Tuple{Int, Int}, scale::Float64) where {RNG<:AbstractRNG}
    L = ceil(Int, maximum(map_size) * scale) + 3
    G = [(cos(θ), sin(θ)) for θ in 2π .* rand(rng, L, L)]
    ox, oy = rand(rng, 2) # random lattice offset keeps the field stationary
    return [perlin(i * scale + ox, j * scale + oy, G) for i in 1:map_size[1], j in 1:map_size[2]]
end

function build_map(rng::RNG, in_map::BitMatrix, scale::Float64, σ_field::Float64) where {RNG<:AbstractRNG}
    true_map = raw_field(rng, size(in_map), scale) ./ σ_field
    true_map[.!in_map] .= NaN
    return true_map
end

# empirical field stddev, valuable threshold, and squared exponential length scale
function field_statistics(rng::RNG, in_map::BitMatrix, scale::Float64; n_maps=500, valuable_frac=0.25) where {RNG<:AbstractRNG}
    F = reduce(hcat, [raw_field(rng, size(in_map), scale)[in_map] for _ in 1:n_maps])
    σ_field = std(F)
    Z = F ./ σ_field
    τ = quantile(vec(Z), 1 - valuable_frac)

    X = build_query_points(in_map)
    D = [norm(x - x′) for x in X, x′ in X]
    C = cov(Z')
    ℓ = argmin(l -> sum(abs2, C - exp.(-D.^2 ./ (2l^2))), 0.3:0.05:8.0)

    return σ_field, τ, ℓ
end

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
        return sum((prob_valuable(pomdp, μ_post, ν_post, c) for c in cells if !(c in b.mapped)), init=0.0)
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

function solver_test_SatelliteBMDP(; n::Int=16, circle::Bool=true, scale::Float64=0.2, total_budget=6.0, σ_obs=0.1, init_offset::Int=1,
                                     seed::Int64=1234, num_trials=20, depth=5, queries=200)

    in_map = circle ? circle_mask(n) : trues(n, n)
    σ_field, τ, ℓ = field_statistics(MersenneTwister(seed), in_map, scale)
    @printf("σ_field = %.3f  τ = %.2f  ℓ = %.2f\n", σ_field, τ, ℓ)

    k = with_lengthscale(SqExponentialKernel(), ℓ)
    m(x) = 0.0
    X_query = build_query_points(in_map)
    KXqXq = K(X_query, X_query, k)
    GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq)
    f_prior = GP

    policy_names = ["random", "raster", "greedy", "gp_mcts_dpw"]
    rewards = Dict(p => Float64[] for p in policy_names)
    planning_times = Dict(p => 0.0 for p in policy_names)
    n_valuable = Float64[]

    for i in 1:num_trials
        rng = MersenneTwister(seed + i)

        true_map = build_map(rng, in_map, scale, σ_field)
        pomdp = SatellitePOMDP(true_map=true_map, f_prior=f_prior, in_map=in_map, init_offset=init_offset,
                               σ_obs=σ_obs, value_threshold=τ, cost_budget=total_budget, rng=rng)
        up = SatelliteBeliefUpdater(pomdp)
        bmdp = BeliefMDP(pomdp, up, belief_reward)

        policies = Dict("random" => b -> rand(rng, actions(pomdp, b)),
                        "raster" => b -> raster_policy(pomdp, b),
                        "greedy" => b -> greedy_policy(pomdp, b),
                        "gp_mcts_dpw" => get_gp_bmdp_policy(bmdp, rng, depth, queries))

        push!(n_valuable, count(c -> is_valuable(pomdp, true_map[c]), findall(in_map)))
        for p in policy_names
            total_reward, state_hist, gp_hist, action_hist, reward_hist, planning_time, num_plans = run_satellite_bmdp(rng, pomdp, up, policies[p])
            push!(rewards[p], total_reward)
            planning_times[p] += planning_time/num_plans
        end
    end

    @printf("%d cells in map, %.1f valuable on average, budget %.0f sweeps\n", count(in_map), mean(n_valuable), total_budget)
    for p in policy_names
        R = rewards[p]
        @printf(" %-12s mean = %.2f ± %.2f   (%.3f s per action)\n", p, mean(R), std(R)/sqrt(length(R)), planning_times[p]/num_trials)
    end

    return rewards
end


println("6x6 square:")
solver_test_SatelliteBMDP(n=6, circle=false, scale=0.35, total_budget=3.0)

println("\n16 cell diameter circle:")
solver_test_SatelliteBMDP(n=16, circle=true, scale=0.2, total_budget=6.0)
