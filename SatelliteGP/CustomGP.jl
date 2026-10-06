# Adapted from sisl/SBO_AIPPMS Rover/GP_BMDP_Rover/CustomGP.jl (MIT License)
using Random, LinearAlgebra, Distributions, KernelFunctions

mutable struct GaussianProcess
    m # mean
    mXq # mean function at query points
    k # covariance function
    X # design points
    X_query # query points (assuming these always stay the same)
    y # objective values
    ν # noise variance
    KXX # K(X,X) the points we have measured
    KXqX # K(Xq,X) the points we are querying and we have measured
    KXqXq
end

μ(X, m) = [m(x) for x in X]
Σ(X, k) = kernelmatrix(k, X, X)
K(X, X′, k) = kernelmatrix(k, X, X′)

function mvnrand(rng, μ, Σ, inflation=1e-6)
    N = MvNormal(μ, Symmetric(Σ) + inflation*I)
    return rand(rng, N)
end
Base.rand(rng::AbstractRNG, GP::GaussianProcess, X) = mvnrand(rng, μ(X, GP.m), Σ(X, GP.k))
Base.rand(rng::AbstractRNG, GP::GaussianProcess, μ_calc, Σ_calc) = mvnrand(rng, μ_calc, Σ_calc)

function query_no_data(GP::GaussianProcess)
    μₚ = GP.mXq
    S = GP.KXqXq
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ, S)
end

function query(GP::GaussianProcess)
    tmp = GP.KXqX / (GP.KXX + Diagonal(GP.ν))
    μₚ = GP.mXq + tmp*(GP.y - μ(GP.X, GP.m))
    S = GP.KXqXq - tmp*GP.KXqX'
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ, S)
end

function posterior(GP::GaussianProcess, X_samp, y_samp, ν_samp)
    if GP.X == []
        KXX = K(X_samp, X_samp, GP.k)
        KXqX = K(GP.X_query, X_samp, GP.k)

        return GaussianProcess(GP.m, GP.mXq, GP.k, X_samp, GP.X_query, y_samp, ν_samp, KXX, KXqX, GP.KXqXq)
    else
        a = K(GP.X, X_samp, GP.k)
        KXX = [GP.KXX a; a' K(X_samp, X_samp, GP.k)] # a whole swath arrives at once, not one point
        KXqX = [GP.KXqX K(GP.X_query, X_samp, GP.k)]

        return GaussianProcess(GP.m, GP.mXq, GP.k, [GP.X; X_samp], GP.X_query, [GP.y; y_samp], [GP.ν; ν_samp], KXX, KXqX, GP.KXqXq)
    end
end
