using Random, Statistics, LinearAlgebra, Parameters

const R_ENCELADUS = 252.1 # km
const PERLIN_STD = 0.216 # empirical std of raw_field, used to normalize the texture

@with_kw struct PlumeParams
    R_map::Float64 = 100.0 # km from the south pole to the map edge
    stripe_len::Float64 = 130.0 # km, tiger stripe length
    stripe_gap::Float64 = 35.0  # km between neighboring stripes
    stripe_angle::Float64 = 40.0  # deg, stripe orientation in the map frame
    n_jets::Int = 6     # active jets per stripe, positions and strengths random per map
    curtain_amp::Float64  = 0.3   # total curtain emission per stripe, relative to the jets
    jet_σ::Float64 = 0.75  # lognormal spread of jet strengths
    w_near::Float64 = 6.0   # km, near-vent fallout width (large grains)
    w_far::Float64  = 40.0  # km, broad fallout width (small grains)
    far_frac::Float64 = 0.2   # weight of the broad fallout term
    texture::Float64 = 0.05  # amplitude of the Perlin terrain texture
    scale::Float64= 0.2   # Perlin frequency
end


################################################################################
# Perlin texture
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


################################################################################
# Plume fallout
################################################################################
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


################################################################################
# Sun and Earth constraints
################################################################################
# weight in [w_min, 1] from the body's elevation above each cell's local horizon
function elevation_weight(n::Int, p::PlumeParams; azimuth, elev_pole, elev_full=20.0, w_min=0.2)
    X, Y, km_per_cell = cell_km(n, p.R_map)
    colat = rad2deg.(hypot.(X, Y) ./ R_ENCELADUS)
    elev = elev_pole .+ colat .* cos.(atan.(Y, X) .- deg2rad(azimuth))
    return w_min .+ (1 - w_min) .* clamp.(elev ./ elev_full, 0, 1)
end

# sun lighting for imaging, Earth in view for landing site comms
function constraint_weights(in_map::BitMatrix, p::PlumeParams; sun_azimuth=30.0, sun_elev_pole=10.0, use_earth=true,
                            earth_offset=5.0, w_min=0.2)
    n = size(in_map, 1)
    W = elevation_weight(n, p; azimuth=sun_azimuth, elev_pole=sun_elev_pole, w_min=w_min)
    if use_earth
        # seen from Saturn, Earth stays within about 6 degrees of the Sun
        W .*= elevation_weight(n, p; azimuth=sun_azimuth + earth_offset, elev_pole=sun_elev_pole, w_min=w_min)
    end
    W[.!in_map] .= NaN
    return W
end
