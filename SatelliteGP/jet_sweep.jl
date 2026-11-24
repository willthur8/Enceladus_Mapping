# julia --project=. -t 4 jet_sweep.jl

const SATELLITE_NO_MAIN = true
const SATELLITE_NO_PLOTS = true
include("Trials_SatelliteBMDP.jl")

const N = 24
const ANGLE_STEP = 30.0
const BUDGETS = [5, 10, 20] # scaled_budget(24) × {0.5, 1, 2}; the constant budgets 10 and 20 are already included
const SEED = 1234
const NUM_TRIALS = 20
const POLICIES = ["raster", "greedy", "gp_mcts_dpw"]
const QUERIES = 200 # MCTS iterations per action
const DEPTH = 5
const C_24 = 10.0 # MCTS exploration constant at n = 24
const MAX_PARALLEL = 4
const JET_TS = [Inf, 20.0, 5.0, 1.25] # jet time scale T_z in passes, the same for every budget (TODO: physical value)
const BELIEF_T_PER_JET_T = [0.5, 1.0, 2.0] # belief T as a multiple of T_z, plus T = Inf (belief assumes nothing changes)
const FIXED_SHARES = [0.0, 0.3, 0.6]
const CSV_PATH = joinpath("results", "jet_sweep.csv")

# (jet_T, belief_T, fixed_share) settings; the unchanging field only gets the unchanging belief
function jet_settings()
    settings = Tuple{Float64, Float64, Float64}[]
    for jet_T in JET_TS
        push!(settings, (jet_T, Inf, 0.0))
        isinf(jet_T) && continue
        for m in BELIEF_T_PER_JET_T, a in FIXED_SHARES
            push!(settings, (jet_T, m*jet_T, a))
        end
    end
    return settings
end

mkpath("results")
const io = open(CSV_PATH, "w")
println(io, "n,angle_step,budget,c,jet_T,belief_T,fixed_share,trial,policy,reward,available,stays,repeat_frac,sec_per_action,wall_seconds")
const write_lock = ReentrantLock()

const EXPLORATION = C_24*(N/24)^2
for budget in BUDGETS, (jet_T, belief_T, a) in jet_settings()
    @printf("\nn = %d, %g° steps, budget %d, c = %.1f, jet_T = %g, belief_T = %g, fixed share = %g\n",
            N, ANGLE_STEP, budget, EXPLORATION, jet_T, belief_T, a)

    write_trial = r -> lock(write_lock) do
        for (j, policy) in enumerate(POLICIES)
            println(io, join((N, ANGLE_STEP, budget, EXPLORATION, jet_T, belief_T, a, r.trial, policy, r.rewards[j], r.available,
                              r.stays[j], r.repeat_frac[j], r.sec_per_action[j], r.wall_seconds[j]), ","))
        end
        flush(io)
    end

    solver_test_SatelliteBMDP(n=N, angle_step=ANGLE_STEP, total_budget=budget, seed=SEED, num_trials=NUM_TRIALS,
                              policy_names=POLICIES, depth=DEPTH, queries=QUERIES, exploration_constant=EXPLORATION,
                              max_parallel=MAX_PARALLEL, jet_T=jet_T, belief_T=belief_T, fixed_share=a, allow_stay=true,
                              on_trial=write_trial)
end

close(io)
println("\ndone: ", CSV_PATH)
