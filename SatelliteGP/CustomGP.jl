# Adapted from sisl/SBO_AIPPMS Rover/GP_BMDP_Rover/CustomGP.jl 
#
# Changed for  speed:
#  posterior keeps the lower Cholesky factor L of K(X,X) + diag(ν) instead of K(X,X). one pass adds one block,
#  so the factor is extended (cost n^2*s for n old and s new points) instead of re-solving the whole thing every time.
#  posterior mean is mXq + K(Xq,X) α with α cached, so the mean never needs full covariance.
# query_cells and query_var return only what a caller needs. query still returns the full (μ, ν, S) for plots.
using Random, LinearAlgebra, Distributions, KernelFunctions

const GP_JITTER =1e-8 # added to K(X,X) + diag(ν) so repeated points with σ_obs = 0 stay factorizable

mutable struct GaussianProcess
    m # mean
    mXq # mean function at query points
    k # covariance function
    X # design points
    X_query # query points (assuming these always stay the same)
    y # objective values
    ν # noise variance
    L # lower Cholesky factor of K(X,X) + diag(ν) (replaces KXX)
    KXqX # K(Xq,X) the points we are querying and we have measured
    KXqXq
    α # (K(X,X) + diag(ν)) \ (y - m(X)), so the posterior mean is mXq + KXqX α
    time # nothing for a field that does not change between passes, otherwise a TimeKernel
    t_obs # pass of each design point (empty when time is nothing). KXqX is spatial the time factor is applied per query
end

# prior with no observations
GaussianProcess(m, mXq, k, X_query, KXqXq; time=nothing) = GaussianProcess(m, mXq, k, [], X_query, [], [], [], [], KXqXq, [], time, Int[])

# Changing field: k((x,t),(x′,t′)) = k(x,x′) g(t - t′) with g(Δt) = a + (1 - a) exp(-|Δt|/T).
# a = share of the variance that never changes, T = decay time of the rest, in passes. a = 0 is one plain exponential.
struct TimeKernel
    a::Float64
    T::Float64
end
time_corr(tk::TimeKernel, Δt) = tk.a + (1 - tk.a)*exp(-abs(Δt)/tk.T)
time_corr(GP::GaussianProcess, ta, tb) = time_corr.(Ref(GP.time), ta .- transpose(tb))

last_pass(GP::GaussianProcess) = isempty(GP.t_obs) ? 0 : GP.t_obs[end]

# nonstationary kernel k(x,x′) = s(x) s(x′) k₀(x,x′), where s is a per-cell prior std map
struct ScaledFieldKernel{Kₒ<:Kernel} <: Kernel
    k::Kₒ
    s::Matrix{Float64}
end
(κ::ScaledFieldKernel)(x, x′) = κ.s[x[1], x[2]] * κ.s[x′[1], x′[2]] * κ.k(x, x′)

μ(X, m) = [m(x) for x in X]
Σ(X, k) = kernelmatrix(k, X, X)
K(X, X′, k) = kernelmatrix(k, X, X′)

function mvnrand(rng, μ, Σ, inflation=1e-6)
    N = MvNormal(μ, Symmetric(Σ) + inflation*I)
    return rand(rng, N)
end
Base.rand(rng::AbstractRNG, GP::GaussianProcess, X) = mvnrand(rng, μ(X, GP.m), Σ(X, GP.k))
Base.rand(rng::AbstractRNG, GP::GaussianProcess, μ_calc, Σ_calc) = mvnrand(rng, μ_calc, Σ_calc)

# Cholesky factor of a symmetric matrix that should be positive definite, adding jitter only if rounding made it fail
function chol_lower(A::AbstractMatrix)
    C = cholesky(Symmetric(A), check=false)
    jitter = 1e-10
    while !issuccess(C)
        jitter *= 10
        jitter > 1e-3 && error("covariance block is not positive definite even with jitter 1e-3")
        C = cholesky(Symmetric(A + jitter*I), check=false)
    end
    return Matrix(C.L)
end

has_data(GP::GaussianProcess) = GP.X != []

# K((Xq[q], tq), X): rows q of the cross covariance with the query points at pass tq (q = : for all, no copy)
function cross_cov(GP::GaussianProcess, q, tq)
    KqX = q === Colon() ? GP.KXqX : GP.KXqX[q, :]
    GP.time === nothing && return KqX
    return KqX .* time_corr(GP, tq, GP.t_obs)
end


whiten(GP::GaussianProcess, q, tq) = LowerTriangular(GP.L) \ Matrix(transpose(cross_cov(GP, q, tq)))

function query_no_data(GP::GaussianProcess)
    μₚ = GP.mXq
    S = GP.KXqXq
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ, S)
end

# full posterior mean, variance, and covariance at every query point 
# tq is the pass the posterior is wanted at 
function query(GP::GaussianProcess; tq=last_pass(GP))

    has_data(GP) || return query_no_data(GP)
    V = whiten(GP, 1:size(GP.KXqX, 1), tq)

    μₚ = GP.mXq + cross_cov(GP, :, tq)*GP.α
    S = GP.KXqXq - V'V

    νₚ = diag(S) .+ eps() 
    return (μₚ, νₚ, S)
end

# posterior mean at every query point, O(Nq n)
query_mean(GP::GaussianProcess; tq=last_pass(GP)) = has_data(GP) ? GP.mXq + cross_cov(GP, :, tq)*GP.α : GP.mXq

# posterior variance at every query point without forming the Nq × Nq covariance. It depends only on X, not on y.
function query_var(GP::GaussianProcess; tq=last_pass(GP))
    has_data(GP) || return diag(GP.KXqXq) .+ eps()
    V = whiten(GP, 1:size(GP.KXqX, 1), tq)
    return diag(GP.KXqXq) .- vec(sum(abs2, V, dims=1)) .+ eps()
end

# posterior mean, variance, and covariance at the query indices q only
function query_cells(GP::GaussianProcess, q; tq=last_pass(GP))
    has_data(GP) || return (GP.mXq[q], diag(GP.KXqXq)[q] .+ eps(), GP.KXqXq[q, q])
    V = whiten(GP, q, tq)
    μq = GP.mXq[q] + cross_cov(GP, q, tq)*GP.α
    Sq = GP.KXqXq[q, q] - V'V
    νq = diag(Sq) .+ eps()
    return (μq, νq, Sq)
end


function posterior(GP::GaussianProcess, X_samp, y_samp, ν_samp; t_samp=Int[])
    isempty(X_samp) && return GP

    Kss = K(X_samp, X_samp, GP.k)
    GP.time === nothing || (Kss = Kss .* time_corr(GP, t_samp, t_samp))
    Kss = Kss + Diagonal(ν_samp) + GP_JITTER*I
    KXqXs = K(GP.X_query, X_samp, GP.k)

    if !has_data(GP)
        X, y, ν = X_samp, y_samp, ν_samp
        L = chol_lower(Kss)
        KXqX = KXqXs
    else
        # a whole swath arrives at once, so extend the factor by one block:
        # [L 0; B' C] with B = L \ K(X, X_samp) and C C' = Kss - B'B
        KXXs = K(GP.X, X_samp, GP.k)
        GP.time === nothing || (KXXs = KXXs .* time_corr(GP, GP.t_obs, t_samp))
        B = LowerTriangular(GP.L) \ KXXs
        C = chol_lower(Kss - B'B)
        X, y, ν = [GP.X; X_samp], [GP.y; y_samp], [GP.ν; ν_samp]
        L = [GP.L zeros(size(GP.L, 1), length(X_samp)); B' C]
        KXqX = [GP.KXqX KXqXs]
    end

    Lt = LowerTriangular(L)
    α = Lt' \ (Lt \ (y - μ(X, GP.m)))
    t_obs = GP.time === nothing ? GP.t_obs : [GP.t_obs; t_samp]

    return GaussianProcess(GP.m, GP.mXq, GP.k, X, GP.X_query, y, ν, L, KXqX, GP.KXqXq, α, GP.time, t_obs)
end

# Pieces of the forecast in beliefs.jl, at pass tq: the variance ν, and when the fixed share a > 0 also
# Q_ss = diag(Ks A⁻¹ Ks') and Q_st = diag(Kt A⁻¹ Ks'), where Ks is the spatial cross covariance (no time decay),
# Kt the cross covariance at tq, and A = K(X,X) + diag(ν).
function query_var_terms(GP::GaussianProcess; tq=last_pass(GP))
    s2 = diag(GP.KXqXq)
    
    has_data(GP) || return (ν=s2 .+ eps(), Qss=zeros(length(s2)), Qst=zeros(length(s2)))
    Vt = whiten(GP, :, tq)

    ν = s2 .- vec(sum(abs2, Vt, dims=1)) .+ eps()
    (GP.time === nothing || GP.time.a == 0) && return (ν=ν, Qss=Float64[], Qst=Float64[])

    Vs = LowerTriangular(GP.L) \ Matrix(transpose(GP.KXqX))
    return (ν=ν, Qss=vec(sum(abs2, Vs, dims=1)), Qst=vec(sum(Vt .* Vs, dims=1)))
end

# Ks α: the part of the posterior mean shift computed with spatial covariances only (used by the forecast when a > 0)
query_mean_space(GP::GaussianProcess) = has_data(GP) ? GP.KXqX*GP.α : zeros(length(GP.mXq))
