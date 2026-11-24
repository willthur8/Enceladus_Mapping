using Plots

const FIELD_COLORS = cgrad(["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
const STD_COLORS = cgrad(["#fdeee6", "#f6b08f", "#eb6834", "#b4471c", "#7a2c0e"])
const PLUME_COLOR = "#0b0b0b"
const MAPPED_COLOR = "#52514e"
const PASS_COLOR = "#d55181"
const EDGE_COLOR = "#a8a79f"
const LINE_COLOR = "#2a78d6"


gp_posterior(gp::GaussianProcess) = gp.X == [] ? query_no_data(gp) : query(gp)

function to_map(pomdp::SatellitePOMDP, v)
    M = fill(NaN, pomdp.map_size)
    M[pomdp.in_map] = v
    return M
end

function cell_mask(pomdp::SatellitePOMDP, cells)
    M = falses(pomdp.map_size)
    M[collect(cells)] .= true
    return M
end

function region_outline(mask::AbstractMatrix{Bool})
    xs, ys = Float64[], Float64[]
    n1, n2 = size(mask)
    inside(i, j) = 1 <= i <= n1 && 1 <= j <= n2 && mask[i, j]
    for j in 1:n2, i in 1:n1
        mask[i, j] || continue
        if !inside(i-1, j); append!(xs, [j-0.5, j+0.5, NaN]); append!(ys, [i-0.5, i-0.5, NaN]); end
        if !inside(i+1, j); append!(xs, [j-0.5, j+0.5, NaN]); append!(ys, [i+0.5, i+0.5, NaN]); end
        if !inside(i, j-1); append!(xs, [j-0.5, j-0.5, NaN]); append!(ys, [i-0.5, i+0.5, NaN]); end
        if !inside(i, j+1); append!(xs, [j+0.5, j+0.5, NaN]); append!(ys, [i-0.5, i+0.5, NaN]); end
    end
    return isempty(xs) ? ([NaN], [NaN]) : (xs, ys)
end


function map_panel(pomdp::SatellitePOMDP, M, title, colors, clims)
    n1, n2 = pomdp.map_size
    return heatmap(1:n2, 1:n1, M, c=colors, clims=clims, yflip=true, aspect_ratio=:equal,
                   xlims=(0.5, n2+0.5), ylims=(0.5, n1+0.5), axis=false, ticks=false, grid=false,
                   title=title, titlefontsize=10, colorbar=true, legend=false)
end

function add_overlays!(p, pomdp::SatellitePOMDP, prev_mapped, current_cells, plume)
    plot!(p, region_outline(pomdp.in_map)..., color=EDGE_COLOR, lw=1)
    if plume !== nothing
        plot!(p, region_outline(plume)..., color=:white, lw=3.2) #outline visible on dark cells
        plot!(p, region_outline(plume)..., color=PLUME_COLOR, lw=1.2)
    end
    plot!(p, region_outline(cell_mask(pomdp, prev_mapped))..., color=MAPPED_COLOR, lw=1.5, ls=:dash)
    plot!(p, region_outline(cell_mask(pomdp, current_cells))..., color=PASS_COLOR, lw=3)
    return p
end

function add_captured!(p, pomdp::SatellitePOMDP, mapped)
    cells = [c for c in mapped if is_valuable(pomdp, pomdp.true_map[c])]
    isempty(cells) && return p
    coords = [convert_idx_2_coord(pomdp, c) for c in cells]
    scatter!(p, [c[2] for c in coords], [c[1] for c in coords], color=:white,
             markerstrokecolor=PLUME_COLOR, markersize=3, markerstrokewidth=1)
    return p
end

function reward_panel(action_hist, reward_hist, i, n_valuable)
    T = length(action_hist)
    steps = 0:T
    cum = [0; cumsum(reward_hist)]
    tick_labels = ["$(s)\n$(s == 0 ? "start" : (s < i ? string(action_hist[s]) : ""))" for s in steps]

    p = plot(steps[1:i], cum[1:i], color=LINE_COLOR, lw=2, marker=:circle, markersize=5,
             markerstrokecolor=:white, legend=false, grid=:y, gridalpha=0.25,
             xlims=(-0.5, T+0.5), ylims=(0, 1.15*max(cum[end], 1)), xticks=(steps, tick_labels),
             ylabel="valuable cells mapped", titlefontsize=10, guidefontsize=9, tickfontsize=8,
             title="Cumulative reward: $(Int(cum[i])) of $(Int(n_valuable)) valuable cells")
    annotate!(p, steps[i], cum[i] + 0.06*max(cum[end], 1), text("$(Int(cum[i]))", 9, :bottom))
    return p
end

function legend_panel()
    p = plot(framestyle=:none, legend=:left, grid=false, ticks=false, legendfontsize=9,
             foreground_color_legend=nothing, background_color_legend=nothing)
    plot!(p, [NaN], [NaN], color=PLUME_COLOR, lw=1.2, label="plume area (value > τ)")
    plot!(p, [NaN], [NaN], color=MAPPED_COLOR, lw=1.5, ls=:dash, label="mapped earlier")
    plot!(p, [NaN], [NaN], color=PASS_COLOR, lw=3, label="current pass")
    scatter!(p, [NaN], [NaN], color=:white, markerstrokecolor=PLUME_COLOR, markersize=4, label="valuable cell captured")
    plot!(p, [NaN], [NaN], color=EDGE_COLOR, lw=1, label="map edge")
    return p
end

function step_maps(pomdp::SatellitePOMDP, state_hist, gp_hist, i, field_clims)
    s = state_hist[i]
    μ_post, ν_post, S_post = gp_posterior(gp_hist[i])
    prev_mapped = i == 1 ? Set{Int}() : state_hist[i-1].mapped
    current = i == 1 ? Int[] : swath(pomdp, s.offset, s.horizontal)

    true_plume = pomdp.true_map .> pomdp.value_threshold
    μ_map = to_map(pomdp, μ_post)
    belief_plume = μ_map .> pomdp.value_threshold

    p_true = map_panel(pomdp, pomdp.true_map, "True field", FIELD_COLORS, field_clims)
    add_overlays!(p_true, pomdp, prev_mapped, current, true_plume)
    add_captured!(p_true, pomdp, s.mapped)

    p_mean = map_panel(pomdp, μ_map, "Belief mean (outline: μ > τ)", FIELD_COLORS, field_clims)
    add_overlays!(p_mean, pomdp, prev_mapped, current, belief_plume)

    p_std = map_panel(pomdp, to_map(pomdp, sqrt.(ν_post)), "Belief uncertainty (std)", STD_COLORS, (0, 1))
    add_overlays!(p_std, pomdp, prev_mapped, current, nothing)

    return p_true, p_mean, p_std
end


function plot_frame(pomdp::SatellitePOMDP, state_hist, gp_hist, action_hist, reward_hist, i)
    field_clims = extrema(filter(!isnan, pomdp.true_map))
    n_valuable = count(c -> is_valuable(pomdp, pomdp.true_map[c]), findall(pomdp.in_map))

    p_true, p_mean, p_std = step_maps(pomdp, state_hist, gp_hist, i, field_clims)
    top = plot(p_true, p_mean, p_std, layout=(1, 3))
    bottom = plot(reward_panel(action_hist, reward_hist, i, n_valuable), legend_panel(), layout=grid(1, 2, widths=[0.72, 0.28]))

    T = length(action_hist)
    header = i == 1 ? "Step 0 of $T: prior belief" : "Step $(i-1) of $T: action $(action_hist[i-1]), +$(Int(reward_hist[i-1])) valuable cells"
    return plot(top, bottom, layout=grid(2, 1, heights=[0.62, 0.38]), size=(1300, 760),
                plot_title=header, plot_titlefontsize=13, left_margin=4Plots.mm, bottom_margin=4Plots.mm)
end

function plot_trial(pomdp::SatellitePOMDP, state_hist, gp_hist, action_hist, reward_hist, trial_num, trial_name; dir="figures", fps=1)
    path = joinpath(dir, trial_name)
    mkpath(path)

    frames = [1:length(state_hist); fill(length(state_hist), 2)] # hold  last frame
    anim = @animate for i in frames
        plot_frame(pomdp, state_hist, gp_hist, action_hist, reward_hist, i)
    end
    gif(anim, joinpath(path, "trial$(trial_num).gif"), fps=fps)

    plot_summary(pomdp, state_hist, gp_hist, joinpath(path, "trial$(trial_num)_summary"))
end


function plot_summary(pomdp::SatellitePOMDP, state_hist, gp_hist, filename)
    field_clims = extrema(filter(!isnan, pomdp.true_map))
    T = length(state_hist) - 1
    rows = unique([1, cld(T, 2) + 1, T + 1])

    panels = []
    for i in rows
        p_true, p_mean, p_std = step_maps(pomdp, state_hist, gp_hist, i, field_clims)
        plot!(p_true, title="Step $(i-1): true field")
        plot!(p_mean, title="Step $(i-1): belief mean")
        plot!(p_std, title="Step $(i-1): uncertainty")
        append!(panels, [p_true, p_mean, p_std])
    end

    p = plot(panels..., layout=(length(rows), 3), size=(1200, 380*length(rows)))
    savefig(p, filename * ".png")
    
    return p
end
