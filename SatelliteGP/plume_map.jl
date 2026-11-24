using Random, Statistics, LinearAlgebra, Parameters, Serialization

const PERLIN_STD = 0.216 #  std of raw_field, used to normalize texture

@with_kw struct PlumeParams
    R_map::Float64= 100.0 # km from the south pole to the map edge
    stripe_len::Float64= 130.0 # km, tiger stripe length
    stripe_gap::Float64= 35.0  # km between neighboring stripes
    stripe_angle::Float64 = 40.0  # deg, stripe orientation in the map frame
    n_jets::Int= 6     # active jets per stripe, positions and strengths random per map
    curtain_amp::Float64= 0.3   # total curtain emission per stripe, relative to the jets
    jet_σ::Float64= 0.75  # lognormal spread of jet strengths
    w_near::Float64= 6.0   # km, near-vent fallout width (large grains)
    w_far::Float64= 40.0  # km, broad fallout width (small grains)
    far_frac::Float64= 0.2   # weight of the broad fallout term
    texture::Float64 = 0.05  # amplitude of the Perlin terrain texture
    scale::Float64= 0.2   # Perlin frequency
end


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



# cell centers in km from the pole: x toward increasing column, y toward decreasing row
function cell_km(n::Int, R_map::Float64)
    c = (n + 1)/2
    km_per_cell = 2R_map/n

    X = [(j - c)*km_per_cell for i in 1:n, j in 1:n]
    Y = [-(i - c)*km_per_cell for i in 1:n, j in 1:n]

    return X, Y, km_per_cell
end

function stripe_axes(p::PlumeParams)
    a = deg2rad(p.stripe_angle)
    u = [cos(a), sin(a)]
    v = [-sin(a), cos(a)]
    return [(o*v, u) for o in ((0:3) .- 1.5) .* p.stripe_gap]
end

function raw_deposition(rng::RNG, n::Int, p::PlumeParams) where {RNG<:AbstractRNG}
    X, Y, km_per_cell = cell_km(n, p.R_map)

    sources = Vector{Vector{Float64}}()

    strengths = Float64[]


    for (c0, u) in stripe_axes(p)

        ts = -p.stripe_len/2:5.0:p.stripe_len/2 # curtain sources every 5 km along the stripe

        for t in ts

            push!(sources, c0 + t*u)
            push!(strengths, p.curtain_amp/length(ts))
        end

        for t in p.stripe_len .* (rand(rng, p.n_jets) .- 0.5)
            push!(sources, c0 + t*u)
            push!(strengths, exp(p.jet_σ*randn(rng))/p.n_jets)
        end

    end

    D = zeros(n, n)
    sub = ((1:4) .- 0.5)./4 .- 0.5 # average 4x4 points per cell so narrow fallout is not aliased


    for (s, a) in zip(sources, strengths), du in sub, dv in sub
        d2 = (X .+ du*km_per_cell .- s[1]).^2 .+ (Y .+ dv*km_per_cell .- s[2]).^2
        D .+= a .* (exp.(-d2 ./ (2p.w_near^2)) .+ p.far_frac .* exp.(-d2 ./ (2p.w_far^2))) ./ 16

    end

    texture = raw_field(rng, (n, n), p.scale) ./ PERLIN_STD
    return D, texture


end

function build_map(rng::RNG, in_map::BitMatrix, p::PlumeParams, norm_const::Float64) where {RNG<:AbstractRNG}
    D, texture = raw_deposition(rng, size(in_map, 1), p)
    true_map = D ./ norm_const .+ p.texture .* texture
    true_map[.!in_map] .= NaN

    return true_map
end

# prior mean and std per cell, plume threshold, and squared exponential length scale of the residuals
function field_statistics(rng::RNG, in_map::BitMatrix, p::PlumeParams; n_maps=500, valuable_frac=0.25) where {RNG<:AbstractRNG}
    n = size(in_map, 1)
    raws = [raw_deposition(rng, n, p) for _ in 1:n_maps]
    norm_const = maximum(mean(r[1] for r in raws)[in_map])
    F = reduce(hcat, [(D ./ norm_const .+ p.texture .* texture)[in_map] for (D, texture) in raws])

    mean_map = fill(NaN, size(in_map))
    std_map = fill(NaN, size(in_map))

    mean_map[in_map] = vec(mean(F, dims=2))
    std_map[in_map] = vec(std(F, dims=2))
    τ = quantile(vec(F), 1 - valuable_frac)

    X = build_query_points(in_map)
    D = [norm(x - x′) for x in X, x′ in X]

    C = cor(F')
    ℓ = argmin(l -> sum(abs2, C - exp.(-D.^2 ./ (2l^2))), 0.3:0.05:8.0)

    return (norm_const=norm_const, mean_map=mean_map, std_map=std_map, τ=τ, ℓ=ℓ, field_mean=mean(F), field_std=std(F))


end

# each jet's fallout at strength 1, so  new map is a weighted sum. Jet j has strength exp(z_j)/n_jets as
# in raw_deposition, and its log-strength is an AR(1) process between passes (step_jets in transitions.jl):
# z_j(t+1) = φ z_j(t) + σ_η ε
# φ = exp(-1/jet_T)
# σ_η = jet_σ sqrt(1 - φ²)
# ε ~ N(0, 1)
# so z keeps v3's spread (mean 0, std jet_σ) and its correlation k passes apart is φ^k = exp(-k/jet_T).
struct JetMap
    fixed::Matrix{Float64} # curtain fallout/norm_const + texture, NaN outside the map
    footprints::Vector{Matrix{Float64}} # each jet's fallout at strength 1, divided by norm_const
    z0::Vector{Float64} # log-strengths of the first map
    n_jets::Int
    jet_σ::Float64
end

# fallout at every cell from a source at s with strength 1 averaged over 4x4 points per cell like in raw_deposition to stpo aliasisng
function unit_fallout(X, Y, s, km_per_cell, p::PlumeParams)
    F = zeros(size(X))
    sub = ((1:4) .- 0.5)./4 .- 0.5
    for du in sub, dv in sub
        d2 = (X .+ du*km_per_cell .- s[1]).^2 .+ (Y .+ dv*km_per_cell .- s[2]).^2
        F .+= (exp.(-d2 ./ (2p.w_near^2)) .+ p.far_frac .* exp.(-d2 ./ (2p.w_far^2))) ./ 16
    end
    return F
end

# draws random numbers in the same order as raw_deposition, so a seed's first map matches build_map
function build_jet_map(rng::RNG, in_map::BitMatrix, p::PlumeParams, norm_const::Float64) where {RNG<:AbstractRNG}
    n = size(in_map, 1)
    X, Y, km_per_cell = cell_km(n, p.R_map)
    curtain = zeros(n, n)

    footprints = Matrix{Float64}[]
    z0 = Float64[]


    for (c0, u) in stripe_axes(p)
        ts = -p.stripe_len/2:5.0:p.stripe_len/2
        for t in ts
            curtain .+= (p.curtain_amp/length(ts)) .* unit_fallout(X, Y, c0 + t*u, km_per_cell, p)
        end

        for t in p.stripe_len .* (rand(rng, p.n_jets) .- 0.5)
            push!(footprints, unit_fallout(X, Y, c0 + t*u, km_per_cell, p) ./ norm_const)
            push!(z0, p.jet_σ*randn(rng))
        end
    end

    texture = raw_field(rng, (n, n), p.scale) ./ PERLIN_STD
    fixed = curtain ./ norm_const .+ p.texture .* texture
    fixed[.!in_map] .= NaN
    return JetMap(fixed, footprints, z0, p.n_jets, p.jet_σ)
end

# map for jet log-strengths z
function jet_field(jm::JetMap, z)
    M = copy(jm.fixed)
    for (F, zj) in zip(jm.footprints, z)
        M .+= (exp(zj)/jm.n_jets) .* F
    end
    return M
end

function cached_field_statistics(seed::Int, in_map::BitMatrix, p::PlumeParams; n_maps=500, valuable_frac=0.25, dir="cache")
    compute() = field_statistics(MersenneTwister(seed), in_map, p; n_maps=n_maps, valuable_frac=valuable_frac)
    dir === nothing && return compute()

    n = size(in_map, 1)
    path = joinpath(dir, "field_stats_n$(n)_seed$(seed)_maps$(n_maps).jls")
    inputs = (n=n, p=p, seed=seed, n_maps=n_maps, valuable_frac=valuable_frac, in_map=in_map)
    if isfile(path)
        saved = deserialize(path)
        saved.inputs == inputs && return saved.stats
        @warn "field statistics in $path were made with different inputs, recomputing"
    end

    stats = compute()
    mkpath(dir)
    tmp = tempname(dir; cleanup=false)
    serialize(tmp, (inputs=inputs, stats=stats))
    mv(tmp, path; force=true)
    return stats
end
