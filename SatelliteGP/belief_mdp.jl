#adapted from sisl/SBO_AIPPMS Rover/GP_BMDP_Rover/belief_mdp.jl
"""
    BeliefMDP(pomdp, updater, belief_reward)
Create a belief MDP corresponding to POMDP `pomdp` with belief updates performed by `updater`.
"""
struct BeliefMDP{P<:POMDP, U<:Updater, B, A} <: MDP{B, A}
    pomdp::P
    updater::U
    belief_reward
end

function BeliefMDP(pomdp::P, up::U, belief_reward) where {P<:POMDP, U<:Updater}
    # XXX hack to determine belief type
    b0 = initialize_belief(up, initialstate(pomdp))
    BeliefMDP{P, U, typeof(b0), actiontype(pomdp)}(pomdp, up, belief_reward)
end

function POMDPs.gen(bmdp::BeliefMDP, b, a, rng::AbstractRNG)
    if isterminal(bmdp.pomdp, b)
        return (sp=b, r=0.0)
    end

    s = rand(rng, bmdp.pomdp, b)
    sp, o, r = @gen(:sp, :o, :r)(bmdp.pomdp, s, a, rng)
    bp = update(bmdp.updater, b, a, o)

    r = bmdp.belief_reward(bmdp.pomdp, b, a, bp)
    return (sp=bp, r=r)
end

POMDPs.actions(bmdp::BeliefMDP{P,U,B,A}, b::B) where {P,U,B,A} = actions(bmdp.pomdp, b)
POMDPs.actions(bmdp::BeliefMDP) = actions(bmdp.pomdp)

POMDPs.isterminal(bmdp::BeliefMDP, b) = isterminal(bmdp.pomdp, b)

POMDPs.discount(bmdp::BeliefMDP) = discount(bmdp.pomdp)

function POMDPs.initialstate(bmdp::BeliefMDP)
    return Deterministic(initialize_belief(bmdp.updater, initialstate(bmdp.pomdp)))
end
