
# julia --project=. -t 4 size_sweeps.jl

const SATELLITE_NO_MAIN = true
const SATELLITE_NO_PLOTS = true
include("Trials_SatelliteBMDP.jl")

const NS = [24, 36, 48, 60]
const ANGLE_STEPS = [30.0, 10.0]# 10° avoids  uneven swaths at 45°
const SEED = 1234
const NUM_TRIALS = 20
const POLICIES = ["raster", "greedy", "gp_mcts_dpw"]
const QUERIES = 200 # MCTS iterations per action
const DEPTH = 5
const C_24 = 10.0# MCTS exploration constant at n = 24, scaled by (n/24)^2 because returns grow like n^2
const MAX_PARALLEL = Dict(24 => 4, 36 => 4, 48 => 3, 60 => 1)  # trials at once, to stay under about 20 GB
const CSV_PATH = joinpath("results", "size_sweeps.csv")

mkpath("results")
const io = open(CSV_PATH, "w")
println(io, "n,angle_step,budget,c,trial,policy,reward,available,reward_frac,sec_per_action,wall_seconds")
const write_lock = ReentrantLock()

for n in NS, angle_step in ANGLE_STEPS
    budget, c = scaled_budget(n), C_24*(n/24)^2
    @printf("\nn = %d, %g° steps, budget %d, c = %.1f\n", n, angle_step, budget, c)

    
    write_trial = r -> lock(write_lock) do
        for (j, policy) in enumerate(POLICIES)
            println(io, join((n, angle_step, budget, c, r.trial, policy, r.rewards[j], r.available,
                              r.rewards[j]/r.available, r.sec_per_action[j], r.wall_seconds[j]), ","))
        end
        flush(io)
    end

    solver_test_SatelliteBMDP(n=n, angle_step=angle_step, total_budget=budget, seed=SEED, num_trials=NUM_TRIALS,
                              policy_names=POLICIES, depth=DEPTH, queries=QUERIES, exploration_constant=c,
                              max_parallel=MAX_PARALLEL[n], on_trial=write_trial)
    @printf("peak memory so far: %.1f GB\n", Sys.maxrss()/1e9)
end

close(io)
println("\ndone: ", CSV_PATH)
