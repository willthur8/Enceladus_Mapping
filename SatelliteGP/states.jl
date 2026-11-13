function convert_idx_2_coord(pomdp::SatellitePOMDP, idx::Int)
    return collect(CartesianIndices(pomdp.map_size)[idx].I)
end

function circle_mask(n::Int)
    c = (n + 1)/2
    return BitMatrix([(i - c)^2 + (j - c)^2 <= (n/2)^2 for i in 1:n, j in 1:n])
end

build_query_points(in_map::BitMatrix) = [[c[1], c[2]] for c in findall(in_map)]

function build_query_index(in_map::BitMatrix)
    Q = zeros(Int, size(in_map))
    Q[in_map] .= 1:count(in_map)
    return Q
end

num_offsets(pomdp::SatellitePOMDP) = maximum(pomdp.map_size)
num_angles(pomdp::SatellitePOMDP) = round(Int, 180/pomdp.angle_step)
heading(pomdp::SatellitePOMDP, angle::Int) = (angle - 1)*pomdp.angle_step

# A pass is a straight track at heading θ (deg counterclockwise from +x, same frame as stripe_axes in plume_map.jl)
# a signed distance d from the map center. A cell is in the swath when its center is within half the swath width
# of the track: |-x sin θ + y cos θ - d| <= w/2, all in cells. Offset k is d = k - (n+1)/2, so at heading 0
# offset 1 is the bottom row, and :up moves the track toward its left side.
function build_swaths(in_map::BitMatrix, angle_step::Float64, swath_width::Float64)
    n1, n2 = size(in_map)
    n = max(n1, n2)
    L = LinearIndices(in_map)
    x = [j - (n2 + 1)/2 for i in 1:n1, j in 1:n2]
    y = [-(i - (n1 + 1)/2) for i in 1:n1, j in 1:n2]

    swaths = Matrix{Vector{Int}}(undef, n, round(Int, 180/angle_step))
    for offset in 1:n, angle in 1:size(swaths, 2)
        θ = (angle - 1)*angle_step
        d = offset - (n + 1)/2
        on_track = abs.(-x .* sind(θ) .+ y .* cosd(θ) .- d) .<= swath_width/2
        swaths[offset, angle] = L[in_map .& on_track]
    end
    return swaths
end

# cells observed by one pass, clipped to the map
swath(pomdp::SatellitePOMDP, offset::Int, angle::Int) = pomdp.swaths[offset, angle]

function POMDPs.initialstate(pomdp::SatellitePOMDP)
    return Deterministic(SatelliteState(pomdp.init_offset, pomdp.init_angle, pomdp.true_map, 0.0, Set{Int}()))
end
