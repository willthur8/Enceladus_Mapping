#satellite toy problem.
using Pkg
using POMDPs
using SARSOP # load a  POMDP Solver

using QuickPOMDPs
using POMDPTools

const N = 5 # map size 
const X = 6 # sweep budget
const ACTIONS = [:stay, :down, :up, :switch]

BLOCK_H, BLOCK_W = 3, 3

function make_map(n, r0, c0, h, w)
    M = falses(n, n)

    M[r0:r0+h-1, c0:c0+w-1] .= true

    return M
end

const MAPS = [make_map(N, r, c, BLOCK_H, BLOCK_W) for c in 1:N-BLOCK_W+1 for r in 1:N-BLOCK_H+1]

const VALUABLE_TOTALS = count.(MAPS)

struct SatState
    offset :: Int
    horizontal :: Bool 
    map :: Int # hidden, which grid is being mapped
    t :: Int # how many sweeps have been done
    rows :: NTuple{N, Bool}
    cols :: NTuple{N, Bool}
end

is_mapped(s::SatState, i, j) = s.rows[i] || s.cols[j]

function valuable_mapped(s::SatState)
    M = MAPS[s.map]
    return count(M[i, j] && is_mapped(s, i, j) for i in 1:N, j in 1:N)
end

function transition(s::SatState, a::Symbol)
    h, k = s.horizontal, s.offset
    if a == :down
        k = max(k - 1, 1)
    elseif a == :up
        k = min(k + 1, 5)
    elseif a == :switch
        h = !h
    end

    rows = h ? Base.setindex(s.rows, true, k) : s.rows
    cols = h ? s.cols : Base.setindex(s.cols, true, k)

    return SatState(k, h, s.map, s.t + 1, rows, cols)

end

# observations
function line_values(s::SatState)
    M = MAPS[s.map]
    #return a tuple of booleans corresponding to the passed line.
    return s.horizontal ? Tuple(M[s.offset, :]) : Tuple(M[:, s.offset])
end

const ALL_OBS = vec(collect(Iterators.product(ntuple(_ -> (false, true), N)...)))

# initial state
const EMPTY = ntuple(_ -> false, N)
const INITIAL_STATES = [SatState(1, true, map, 0, EMPTY, EMPTY) for map in eachindex(MAPS)]


m = QuickPOMDP(
    actions = ACTIONS,
    observations = ALL_OBS,
    obstype = NTuple{N,Bool},
    discount = 1.0, 
    transition = (s, a) -> Deterministic(transition(s, a)),
    observation = (a, sp) -> Deterministic(line_values(sp)),
    reward = (s, a) -> valuable_mapped(transition(s, a)) - valuable_mapped(s),
    initialstate = Uniform(INITIAL_STATES),
    isterminal = s -> s.t >= X
)

println("Random policy:")
for (s, a, o, r) in stepthrough(m, RandomPolicy(m), "s,a,o,r", max_steps = X)
println(" t=$(s.t) line=$(s.horizontal ? "row" : "col") $(s.offset) a=$a obs=$o r=$r")
end

using BasicPOMCP
planner = solve(POMCPSolver(tree_queries = 5_000, max_depth = X), m)
println("\nPOMCP policy:")
hist = simulate(HistoryRecorder(max_steps = X), m, planner)
for step in eachstep(hist, "s,a,r")
println(" t=$(step.s.t) a=$(step.a) r=$(step.r)")
end
s0 = first(state_hist(hist))
println(MAPS[s0.map])
println("Valuable cells mapped: ", undiscounted_reward(hist), " / ", count(MAPS[s0.map]))

