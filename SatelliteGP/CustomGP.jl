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
end

# prior with no observations
GaussianProcess(m, mXq, k, X_query, KXqXq) = GaussianProcess(m, mXq, k, [], X_query, [], [], [], [], KXqXq, [])

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

# L \ K(X, Xq[q]): whitened cross covariance, one column per query index in q
whiten(GP::GaussianProcess, q) = LowerTriangular(GP.L) \ Matrix(transpose(GP.KXqX[q, :]))

function query_no_data(GP::GaussianProcess)
    μₚ = GP.mXq
    S = GP.KXqXq
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ, S)
end

# full posterior mean, variance, and covariance at every query point (Nq × Nq, so keep it out of planning loops)
function query(GP::GaussianProcess)
    has_data(GP) || return query_no_data(GP)
    V = whiten(GP, 1:size(GP.KXqX, 1))
    μₚ = GP.mXq + GP.KXqX*GP.α
    S = GP.KXqXq - V'V
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ, S)
end

# posterior mean at every query point, O(Nq n)
query_mean(GP::GaussianProcess) = has_data(GP) ? GP.mXq + GP.KXqX*GP.α : GP.mXq

# posterior variance at every query point without forming the Nq × Nq covariance. It depends only on X, not on y.
function query_var(GP::GaussianProcess)
    has_data(GP) || return diag(GP.KXqXq) .+ eps()
    V = whiten(GP, 1:size(GP.KXqX, 1))
    return diag(GP.KXqXq) .- vec(sum(abs2, V, dims=1)) .+ eps()
end

# posterior mean, variance, and covariance at the query indices q only
function query_cells(GP::GaussianProcess, q)
    has_data(GP) || return (GP.mXq[q], diag(GP.KXqXq)[q] .+ eps(), GP.KXqXq[q, q])
    V = whiten(GP, q)
    μq = GP.mXq[q] + GP.KXqX[q, :]*GP.α
    Sq = GP.KXqXq[q, q] - V'V
    νq = diag(Sq) .+ eps()
    return (μq, νq, Sq)
end

function posterior(GP::GaussianProcess, X_samp, y_samp, ν_samp)
    isempty(X_samp) && return GP

    Kss = K(X_samp, X_samp, GP.k) + Diagonal(ν_samp) + GP_JITTER*I
    KXqXs = K(GP.X_query, X_samp, GP.k)

    if !has_data(GP)
        X, y, ν = X_samp, y_samp, ν_samp
        L = chol_lower(Kss)
        KXqX = KXqXs
    else
        # a whole swath arrives at once, so extend the factor by one block:
        # [L 0; B' C] with B = L \ K(X, X_samp) and C C' = Kss - B'B
        B = LowerTriangular(GP.L) \ K(GP.X, X_samp, GP.k)
        C = chol_lower(Kss - B'B)
        X, y, ν = [GP.X; X_samp], [GP.y; y_samp], [GP.ν; ν_samp]
        L = [GP.L zeros(size(GP.L, 1), length(X_samp)); B' C]
        KXqX = [GP.KXqX KXqXs]
    end

    Lt = LowerTriangular(L)
    α = Lt' \ (Lt \ (y - μ(X, GP.m)))

    return GaussianProcess(GP.m, GP.mXq, GP.k, X, GP.X_query, y, ν, L, KXqX, GP.KXqXq, α)
end
