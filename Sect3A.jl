using Distributed
addprocs()

using Gadfly, Compose, DataFrames, CSV, Colors, ColorSchemes
@everywhere using Distributions, StatsBase, Distances, SharedArrays

function gen_colors(n)
    cs = distinguishable_colors(n, 
        [colorant"#66c2a5", colorant"#fc8d62", colorant"#8da0cb", colorant"#e78ac3",
            colorant"#a6d854", colorant"#ffd92f", colorant"#e5c494", colorant"#b3b3b3"],
        lchoices=Float64[58, 45, 72.5, 90],
        transform=c->deuteranopic(c, 0.1),
        cchoices=Float64[20,40],
        hchoices=[75,51,35,120,180,210,270,310]
    )
    convert(Vector{Color}, cs)
end

function bc_upd(ϵ::Float64, ev_start::Bool, n_agents::Int64, averaging)
    bc_ar = ev_start == true ? collect(range(1/n_agents, stop=1 - 1/n_agents, length=n_agents)) : rand(n_agents)
    v = zeros(Float64, n_agents)
    j = 1
    @views while true
        for i in 1:n_agents
            fa = findall(@. abs(bc_ar[:, j] - bc_ar[i, j]) <= ϵ + eps()) # we add eps() to avoid numerical issues
            v[i] = averaging(bc_ar[:, j][fa])
        end
        bc_ar = hcat(bc_ar, v)
        if bc_ar[:, j] == bc_ar[:, j + 1]
            break
        end
        j += 1
    end
    return bc_ar
end

ϵ = .1

res = bc_upd(ϵ, true, 50, mean)

df = DataFrame(res')
names!(df, [Symbol("$i") for i in 1:size(res, 1)])
df = stack(df)
df[:steps] = repeat(0:(size(res, 2) - 1), outer=size(res, 1))

plot(df, x=:steps, y=:value, color=:variable, Geom.point, Geom.line,
    Coord.cartesian(xmax=size(res, 2)),
    Guide.xlabel("Time"),
    Guide.ylabel("Opinion"),
    Guide.title("ϵ = $ϵ"),
    Theme(key_position=:none, panel_fill="white", point_size=1.25pt, line_width=1.5pt))

function bc_upd_rad(ϵ::Float64, ev_start::Bool, r_opinion::Float64, n_normals::Int64, n_radicals::Int64, averaging)
    st = ev_start == true ? collect(range(1/n_normals, stop=1 - 1/n_normals, length=n_normals)) : rand(n_normals)
    rd = fill(r_opinion, n_radicals)
    bc_ar = vcat(rd, st)
    v = vcat(rd, st)
    j = 1
    @views while true
        for i in (n_radicals + 1):(n_radicals + n_normals)
            fa = findall(@. abs(bc_ar[:, j] - bc_ar[i, j]) <= ϵ + eps())
            v[i] = averaging(bc_ar[:, j][fa])
        end
        bc_ar = hcat(bc_ar, v)
        if all(@. abs(bc_ar[:, j] - bc_ar[:, j + 1]) <= 10^-5)
            break
        end
        j += 1
    end
    return bc_ar
end

res_rad = bc_upd_rad(.3, true, 0.9, 50, 21, mean)

df = DataFrame(res_rad')
names!(df, [Symbol("$i") for i in 1:size(res_rad, 1)])
df = stack(df)
df[:steps] = repeat(0:size(res_rad, 2) - 1, outer=size(res_rad, 1));

plot(df, x=:steps, y=:value, color=:variable, Geom.point, Geom.line,
    Coord.cartesian(xmax=size(res_rad, 2)),
    Guide.xlabel("Time"),
    Guide.ylabel("Opinion"),
    Theme(key_position=:none, panel_fill="white", point_size=1.25pt, line_width=1.5pt))

@everywhere function radicalized_normals_count(ϵ::Float64, 
                                               ev_start::Bool, 
                                               r_opinion::Float64, 
                                               n_normals::Int64, 
                                               n_radicals::Int64, 
                                               averaging)
    st = ev_start == true ? collect(range(1/n_normals, stop=1 - 1/n_normals, length=n_normals)) : rand(n_normals)
    rd = fill(r_opinion, n_radicals)
    bc_ar = vcat(rd, st)
    v = vcat(rd, st)
    j = 1
    @views while true
        @inbounds for i in (n_radicals + 1):(n_radicals + n_normals)
            fa = findall(@. abs(bc_ar[:, j] - bc_ar[i, j]) <= ϵ + eps())
            v[i] = averaging(bc_ar[fa, j])
        end
        bc_ar = hcat(bc_ar, v)
        if all(@. abs(bc_ar[:, j] - bc_ar[:, j + 1]) <= 10^-5)
            break
        end
        j += 1
    end
    return sum(@. abs(bc_ar[:, end] - r_opinion) <= 10^-3) - n_radicals
end

function radical_count_parallel(ev_start::Bool, r_opinion::Float64, n_normals::Int64, averaging)
    res = Array{Int64,2}(undef, 50, 50) # creates a 50 * 50 grid, each cell corresponding to one combination of ϵ and number of radicals
    for j in 1:50
        @inbounds res[:, j] = pmap(i -> radicalized_normals_count(i/100, ev_start, r_opinion, n_normals, j, averaging), 1:50)
    end
    return res
end

out = radical_count_parallel(true, 1.0, 50, mean);

function xlabelname(x)
    n = x/100
    return "$n"
end

function ylabelname(x)
    n = abs(x - 50)
    return "$n"
end

ticks = collect(0:10:50);

Gadfly.spy(rotl90(out),
    Guide.ColorKey(title="Converted\ntruth-seekers"),
    Guide.xlabel("ϵ"), Guide.ylabel("Campaigners"),
    Guide.xticks(ticks=ticks), Scale.x_continuous(labels=xlabelname),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Scale.ContinuousColorScale(p -> get(ColorSchemes.viridis, p), minvalue=0, maxvalue=50),
    Theme(grid_color=colorant"white", panel_fill="white")
)

@everywhere function radicalized_normals_average(ϵ::Float64,
                                                 r_opinion::Float64, 
                                                 n_normals::Int64, 
                                                 n_radicals::Int64, 
                                                 averaging)
    st = rand(n_normals)
    rd = fill(r_opinion, n_radicals)
    bc_ar = vcat(rd, st)
    v = vcat(rd, st)
    j = 1
    @views while true
        for i in (n_radicals + 1):(n_radicals + n_normals)
            fa = findall(@. abs(bc_ar[:, j] - bc_ar[i, j]) <= ϵ)
            @inbounds v[i] = averaging(bc_ar[fa, j])
        end
        bc_ar = hcat(bc_ar, v)
        if all(@. abs(bc_ar[:, j] - bc_ar[:, j + 1]) <= 10^-5)
            break
        end
        j += 1
    end
    return sum(@. abs(@view(bc_ar[:, end]) - r_opinion) <= 10^-3) - n_radicals
end

function radical_average_parallel(r_opinion::Float64, n_normals::Int64, n_radicals::Int64, averaging)
    res = Array{Int64,2}(undef, 50, n_radicals)
    for j in 1:n_radicals
        @inbounds res[:, j] = pmap(i -> radicalized_normals_average(i/100, r_opinion, n_normals, j, averaging), 1:50)
    end
    return res
end

function rand_sim(r_opinion::Float64, n_normals::Int64, n_radicals::Int64, n_sim::Int64, averaging)
    rand_res = Array{Int64,3}(undef, 50, n_radicals, n_sim)
    for i in 1:n_sim
        @inbounds rand_res[:, :, i] = radical_average_parallel(r_opinion, n_normals, n_radicals, averaging)
    end
    return rand_res
end

rand_res = rand_sim(1.0, 50, 50, 100, mean)

out_av = mean(rand_res, dims=3)

Gadfly.spy(rotl90(out_av[:, :, 1]),
    Guide.ColorKey(title="Converted\ntruth-seekers"),
    Guide.xlabel("ϵ"), Guide.ylabel("Campaigners"),
    Guide.xticks(ticks=ticks), Scale.x_continuous(labels=xlabelname),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Scale.ContinuousColorScale(p -> get(ColorSchemes.viridis, p), minvalue=0, maxvalue=50),
    Theme(grid_color=colorant"white", panel_fill="white")
)

out_var = mapslices(variation, rand_res, dims=3)
mean(filter(x -> !isnan(x),out_var)), std(filter(x -> !isnan(x), out_var))

@everywhere function mixed_community_sim(ϵ::Float64, 
                                         ev_start::Bool, 
                                         r_opinion::Float64, 
                                         n_truthseekers::Int64, 
                                         n_indifs::Int64, 
                                         n_radicals::Int64, 
                                         α::Float64, 
                                         τ::Float64, 
                                         averaging_op, 
                                         averaging_eps)
    n_normals = n_truthseekers + n_indifs
    st = ev_start == true ? collect(range(1/n_normals, stop=1 - 1/n_normals, length=n_normals)) : rand(n_normals)
    #δ = ev_start == true ? 10^-12 : 0.0
    rd = fill(r_opinion, n_radicals)
    bc_ar = vcat(st, rd)
    v = vcat(st, rd)
    #eps = ϵ + δ
    j = 1
    @views while true
        for i in 1:n_normals
            fa = findall(@. abs(bc_ar[:, j] - bc_ar[i, j]) <= ϵ + eps())
            @inbounds v[i] = i <= n_truthseekers ? (1 - α)*averaging_op(bc_ar[fa, j]) + α*τ : averaging_op(bc_ar[fa, j])
        end
        bc_ar = hcat(bc_ar, v)
        if all(@. abs(bc_ar[:, j] - bc_ar[:, j + 1]) <= 10^-5)
            break
        end
        j += 1
    end
    res = bc_ar[1:n_truthseekers + n_indifs, end]
    res_t = sum(@. abs(res[1:n_truthseekers] - τ) <= 10^-3)
    res_i = sum(@. abs(res[n_truthseekers + 1:n_normals] - τ) <= 10^-3)
    return res_t/n_truthseekers, res_i/n_indifs
end

@everywhere function sim_mixed(eps::Float64, 
                               r_opinion::Float64, 
                               n_indifs::Int64, 
                               n_radicals::Int64, 
                               α::Float64, 
                               τ::Float64)
    mm = Array{Float64,2}(undef, 100, 2)
    for i in 1:100
        mm[i, :] = [mixed_community_sim(eps, false, r_opinion, 50, n_indifs, n_radicals, α, τ, mean, mean)...]
    end
    return mean(mm, dims=1)
end

τ = .65
ρ = .9
α = .5
ϵ = .1

truth_res = SharedArray{Float64,2}(50, 50)
indif_res = SharedArray{Float64,2}(50, 50)

@sync @distributed for j in 1:50
    for i in 1:50
        truth_res[i, j], indif_res[i, j] = sim_mixed(ϵ, ρ, i, j, α, τ)
    end
end

Gadfly.spy(truth_res[end:-1:1, :],
    Guide.ColorKey(title="Truth seekers\nbelieving τ (proportion)"),
    Guide.xlabel("Campaigners"), Guide.ylabel("Free riders"),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Scale.ContinuousColorScale(p -> get(ColorSchemes.viridis, p), minvalue=0, maxvalue=maximum(truth_res)),
    Guide.title("τ = $τ, ρ = $ρ, ϵ = $ϵ, α = $α"),
    Theme(grid_color=colorant"white", panel_fill="white")
)

Gadfly.spy(indif_res[end:-1:1, :],
    Guide.ColorKey(title="Free riders\nbelieving τ (proportion)"),
    Guide.xlabel("Campaigners"), Guide.ylabel("Free riders"),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Scale.ContinuousColorScale(p -> get(ColorSchemes.viridis, p), minvalue=0, maxvalue=maximum(indif_res)),
    Guide.title("τ = $τ, ρ = $ρ, ϵ = $ϵ, α = $α"),
    Theme(grid_color=colorant"white", panel_fill="white")
)

@everywhere function sim_mixed_rad(ϵ::Float64, 
                                   r_opinion::Float64, 
                                   n_indifs::Int64, 
                                   n_radicals::Int64, 
                                   α::Float64, 
                                   τ::Float64) # does basically the same as the previous functions, except that we now average over 50 simulations
    mm = Array{Float64,2}(undef, 50, 2)
    for i in 1:50
        mm[i, :] = [mixed_community_sim(ϵ, false, r_opinion, 50, n_indifs, n_radicals, α, τ, mean, mean)...]
    end
    return mean(mm, dims=1)
end

function away_from_truth(τ::Float64, ρ::Float64, α::Float64, n_indifs::Int64)
    res = Array{Float64,2}(undef, 50, 50)
    for j in 1:50
        @inbounds res[:, j] = pmap(i -> sim_mixed_rad(i/100, ρ, n_indifs, j, α, τ)[1], 1:50)
    end
    return res
end

out3 = away_from_truth(.1, .3, .5, 0)
out5 = away_from_truth(.1, .5, .5, 0)
out7 = away_from_truth(.1, .7, .5, 0)

Gadfly.spy(rotl90(out3)*50,
    Guide.ColorKey(title="Truth-seekers believing\nthe truth"),
    Guide.xlabel("ϵ"), Guide.ylabel("Campaigners"),
    Guide.xticks(ticks=ticks), Scale.x_continuous(labels=xlabelname),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Scale.ContinuousColorScale(p -> get(ColorSchemes.viridis, p), minvalue=0, maxvalue=50),
    Guide.title("τ = 0.1, ρ = 0.3, α = 0.5"),
    Theme(grid_color=colorant"white", panel_fill="white")
)

out3_50 = away_from_truth(.1, .3, .5, 50)
out5_50 = away_from_truth(.1, .5, .5, 50)
out7_50 = away_from_truth(.1, .7, .5, 50);

Gadfly.spy(rotl90(out7 - out7_50)*50,
    Guide.ColorKey(title="Truth-seekers believing\nthe truth"),
    Guide.xlabel("ϵ"), Guide.ylabel("Campaigners"),
    Guide.xticks(ticks=ticks), Scale.x_continuous(labels=xlabelname),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Scale.ContinuousColorScale(p -> get(ColorSchemes.viridis, p), minvalue=0, maxvalue=50),
    Guide.title("τ = $τ, ρ = $ρ, α = $α, #FR = 50"),
    Theme(grid_color=colorant"white", panel_fill="white")
)


