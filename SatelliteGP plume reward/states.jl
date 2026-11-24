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

# cells observed by one pass, clipped to the map
function swath(pomdp::SatellitePOMDP, offset::Int, horizontal::Bool)
    L = LinearIndices(pomdp.map_size)
    cells = horizontal ? L[offset, :] : L[:, offset]
    return [c for c in cells if pomdp.in_map[c]]
end

function POMDPs.initialstate(pomdp::SatellitePOMDP)
    return Deterministic(SatelliteState(pomdp.init_offset, pomdp.init_horizontal, pomdp.true_map, 0.0, Set{Int}()))
end
