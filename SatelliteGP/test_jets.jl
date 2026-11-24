
const SATELLITE_NO_MAIN = true
const SATELLITE_NO_PLOTS = true
include("Trials_SatelliteBMDP.jl")
using Test

in_map = circle_mask(24)
p = PlumeParams()
stats = cached_field_statistics(1234, in_map, p)

@testset "defaults reproduce v3" begin
    lines = readlines(joinpath("results", "threshold_heldout.csv"))
    names = String.(split(lines[1], ",")[2:end])
    v3 = [parse.(Float64, split(l, ",")) for l in lines[2:end]]
    R = solver_test_SatelliteBMDP(seed=5678, num_trials=length(v3), policy_names=names, mcts_leaf=:greedy,
                                  exploration_constant=10.0, queries=1000, verbose=false)
    for (j, name) in enumerate(names)
        @test maximum(abs(R[name][i] - v3[i][j+1]) for i in eachindex(v3)) < 1e-9
    end
end

@testset "first jet map matches build_map" begin
    r1, r2 = MersenneTwister(9), MersenneTwister(9)
    M1 = build_map(r1, in_map, p, stats.norm_const)
    jm = build_jet_map(r2, in_map, p, stats.norm_const)
    M2 = jet_field(jm, jm.z0)
    @test isnan.(M1) == isnan.(M2)
    @test maximum(abs, M1[in_map] .- M2[in_map]) < 1e-10
    @test rand(r1) == rand(r2) # both used the same number of random draws
end

@testset "jet AR(1)" begin
    rng = MersenneTwister(3)
    pomdp = build_trial(rng, in_map, p, stats, build_f_prior(in_map, stats); jet_T=5.0)
    z = copy(pomdp.jet_map.z0)
    Z = zeros(50_000, length(z))
    for i in 1:size(Z, 1)
        z = step_jets(pomdp, z, rng)
        Z[i, :] = z
    end
    @test abs(std(vec(Z)) - p.jet_σ) < 0.02
    @test abs(mean(vec(Z))) < 0.02
    lag(k) = mean(cor(Z[1:end-k, j], Z[1+k:end, j]) for j in 1:size(Z, 2))
    for k in (1, 5)
        @test abs(lag(k) - exp(-k/5)) < 0.02
    end
end

function fly(pomdp, up, acts, rng)
    s = rand(rng, initialstate(pomdp))
    b = initialize_belief(up, initialstate(pomdp))
    for a in acts
        s, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
        b = update(up, b, a, o)
    end
    return b, s
end

for a_share in (0.0, 0.3)
    @testset "space-time GP and forecast, fixed share $a_share" begin
        f_prior = build_f_prior(in_map, stats; belief_T=4.0, fixed_share=a_share)
        pomdp = build_trial(MersenneTwister(5), in_map, p, stats, f_prior; jet_T=4.0, allow_stay=true, init_offset=10)
        up = SatelliteBeliefUpdater(pomdp)
        b, _ = fly(pomdp, up, [:up, :stay, :angle_up, :up, :stay], MersenneTwister(1))
        gp = b.location_belief
        @test b.t == 5 && gp.t_obs[end] == 5

        A = K(gp.X, gp.X, gp.k) .* time_corr(gp, gp.t_obs, gp.t_obs) + Diagonal(gp.ν)
        f = location_forecast(pomdp, b)
        for k in (0, 1, 3, 10)
            tq = b.t + k
            C = K(gp.X_query, gp.X, gp.k) .* time_corr(gp, tq, gp.t_obs)
            μ_ref = gp.mXq + C*(A \ (gp.y - μ(gp.X, gp.m)))
            ν_ref = diag(gp.KXqXq) - vec(sum(C .* transpose(A \ C'), dims=2))
            @test maximum(abs, query_mean(gp; tq=tq) - μ_ref) < 1e-6
            @test maximum(abs, query_var(gp; tq=tq) - ν_ref) < 1e-6
            fc = [forecast_cell(f, q, k) for q in eachindex(gp.mXq)]
            @test maximum(abs, first.(fc) - μ_ref) < 1e-6
            @test maximum(abs, last.(fc) - ν_ref) < 1e-6
        end
    end
end

@testset "refresh reward and :stay" begin
    pomdp = build_trial(MersenneTwister(5), in_map, p, stats, build_f_prior(in_map, stats); jet_T=4.0, allow_stay=true, init_offset=10)
    rng = MersenneTwister(2)
    s0 = rand(rng, initialstate(pomdp))
    s1, _, r1 = @gen(:sp, :o, :r)(pomdp, s0, :up, rng)
    s2, _, r2 = @gen(:sp, :o, :r)(pomdp, s1, :stay, rng)
    cells = swath(pomdp, s1.offset, s1.angle)
    @test (s2.offset, s2.angle) == (s1.offset, s1.angle)
    @test s2.location_states[in_map] != s1.location_states[in_map] # the jets changed
    @test r1 ≈ sum(cell_value(pomdp, c, s1.location_states[c]) for c in cells)
    @test r2 ≈ sum(cell_value(pomdp, c, s2.location_states[c]) for c in cells)*(1 - exp(-1/4))
end

@testset "same jets for every policy, stay and repeat counts" begin
    pomdp = build_trial(MersenneTwister(5), in_map, p, stats, build_f_prior(in_map, stats); jet_T=2.0, allow_stay=true,
                        init_offset=10, total_budget=6.0, jet_rng=MersenneTwister(8))
    up = SatelliteBeliefUpdater(pomdp)
    @test length(pomdp.map_path) == 7
    run_up = run_satellite_bmdp(MersenneTwister(1), pomdp, up, b -> :up)
    run_stay = run_satellite_bmdp(MersenneTwister(2), pomdp, up, b -> :stay)
    h_up, h_stay = run_up[2], run_stay[2]
    @test all(h_up[k].location_states[in_map] == h_stay[k].location_states[in_map] for k in eachindex(h_up))
    @test run_stay[8] == 6 && run_up[8] == 0
    @test run_stay[9] ≈ 5/6 
end

println("\nshort changing-jets comparison, 2 maps:")
R = solver_test_SatelliteBMDP(num_trials=2, queries=50, policy_names=["raster", "greedy", "gp_mcts_dpw"],
                              jet_T=5.0, belief_T=5.0, fixed_share=0.3, allow_stay=true)
@test all(isfinite, R["gp_mcts_dpw"])
