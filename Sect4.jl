using Distributed
addprocs()

using Gadfly, Compose, DataFrames, Colors, ColorSchemes, RCall
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

function mixed_community(ϵ::Float64, 
                         ev_start::Bool, 
                         r_opinion::Float64, 
                         n_truthseekers::Int64, 
                         n_indifs::Int64, 
                         n_radicals::Int64, 
                         α::Float64, 
                         τ::Float64, 
                         averaging)
    n_normals = n_truthseekers + n_indifs
    st = ev_start == true ? collect(range(1/n_normals, stop=1 - 1/n_normals, length=n_normals)) : rand(n_normals)
    rd = fill(r_opinion, n_radicals)
    bc_ar = vcat(st, rd)
    v = vcat(st, rd)
    j = 1
    @views while true
        for i in 1:n_normals
            fa = findall(isapprox.(bc_ar[:, j], bc_ar[i, j], atol = ϵ + eps()))
            @inbounds v[i] = i <= n_truthseekers ? (1 - α)*averaging(bc_ar[fa, j]) + α*τ : averaging(bc_ar[fa, j])
        end
        bc_ar = hcat(bc_ar, v)
        if all(isapprox.(bc_ar[:, j], bc_ar[:, j + 1], atol=10^-5))
            break
        end
        j += 1
    end
    df0 = DataFrame(bc_ar')
    names!(df0, [Symbol("$i") for i in 1:size(bc_ar, 1)])
    df1 = stack(df0)
    df1[:steps] = repeat(1:size(df0)[1], outer=size(bc_ar, 1))
    bb = Array{String,1}(undef, n_normals + n_radicals)
    bb[1:n_truthseekers] .= "truth-seeker"
    bb[n_truthseekers + 1:n_normals] .= "free rider"
    bb[n_normals + 1:n_normals + n_radicals] .= "campaigner"
    df1[:Agent] = repeat(bb, inner=size(bc_ar, 2))
    return df1
end

out_mixed = mixed_community(.25, false, .9, 25, 20, 10, .25, .65, mean)

plot(out_mixed, x=:steps, y=:value, group=:variable, color=:Agent, Geom.point, Geom.line,
    Coord.cartesian(xmax=maximum(out_mixed[:steps]) + 1),
    Guide.xlabel("Time"),
    Guide.ylabel("Opinion"),
    Scale.color_discrete_hue(gen_colors, levels = ["truth-seeker", "free rider", "campaigner"], preserve_order=true),
    yintercept=[0.65], Geom.hline(style=:dot, color=colorant"grey"),
    Guide.title("τ = .65, ρ = .9, ϵ = .25, α = .25"),
    Theme(panel_fill="white", point_size=1.25pt, line_width=1.5pt))

function bc_upd_rad_with_cd(ϵ_start::Float64, 
                            ev_start::Bool, 
                            r_opinion::Float64, 
                            n_normals::Int64, 
                            n_radicals::Int64, 
                            averaging_op, 
                            averaging_eps)
    st = ev_start == true ? collect(range(1/n_normals, stop=1 - 1/n_normals, length=n_normals)) : rand(n_normals)
    rd = fill(r_opinion, n_radicals)
    bc_ar = vcat(rd, st)
    v = vcat(rd, st)
    eps_rad = fill(0.0, n_radicals)
    eps_nrm = rand(Uniform(0, ϵ_start), n_normals)
    ϵ = vcat(eps_rad, eps_nrm)
    tmp = zeros(Float64, n_normals)
    j = 1
    @views while true
        for i in (n_radicals + 1):(n_radicals + n_normals)
            fa = findall(isapprox.(bc_ar[:, j], bc_ar[i, j], atol = ϵ[i] + eps()))
            v[i] = averaging_op(bc_ar[fa, j])
            tmp[i - n_radicals] = averaging_eps(ϵ[fa])
        end
        ϵ = vcat(eps_rad, tmp)
        bc_ar = hcat(bc_ar, v)
        if all(isapprox.(bc_ar[:, j], bc_ar[:, j + 1], atol=10^-5))
            break
        end
        j += 1
    end
    return bc_ar
end

@everywhere function radicalized_normals_count_with_cd(ϵ_start::Float64, 
                                                       ev_start::Bool, 
                                                       r_opinion::Float64, 
                                                       n_normals::Int64, 
                                                       n_radicals::Int64, 
                                                       averaging_op, 
                                                       averaging_eps)
    st = ev_start == true ? collect(range(1/n_normals, stop=1 - 1/n_normals, length=n_normals)) : rand(n_normals)
    rd = fill(r_opinion, n_radicals)
    bc_ar = vcat(rd, st)
    v = vcat(rd, st)
    eps_rad = fill(0.0, n_radicals)
    eps_nrm = fill(ϵ_start, n_normals)
    ϵ = vcat(eps_rad, eps_nrm)
    tmp = zeros(Float64, n_normals)
    j = 1
    @views while true
        @inbounds for i in (n_radicals + 1):(n_radicals + n_normals)
            fa = findall(isapprox.(bc_ar[:, j], bc_ar[i, j], atol = ϵ[i] + eps()))
            v[i] = averaging_op(bc_ar[fa, j])
            tmp[i - n_radicals] = averaging_eps(ϵ[fa])
        end
        ϵ = vcat(eps_rad, tmp)
        bc_ar = hcat(bc_ar, v)
        if all(isapprox.(bc_ar[:, j], bc_ar[:, j + 1], atol=10^-5))
            break
        end
        j += 1
    end
    return sum(isapprox.(@view(bc_ar[:, end]), r_opinion, atol=10^-3)) - n_radicals
end

function radical_count_parallel_with_cd(ev_start::Bool, 
                                        r_opinion::Float64, 
                                        n_normals::Int64, 
                                        averaging_op, 
                                        averaging_eps)
    res = Array{Int64,2}(undef, 50, 50)
    @inbounds for j in 1:50
        res[:, j] = pmap(i -> radicalized_normals_count_with_cd(i/100, ev_start, r_opinion, n_normals, j, averaging_op, averaging_eps), 1:50)
    end
    return res
end

out_cd = radical_count_parallel_with_cd(true, 0.8, 50, mean, mean)

function xlabelname(x)
    n = x/100
    return "$n"
end

function ylabelname(x)
    n = abs(x - 50)
    return "$n"
end

ticks = collect(0:10:50);

Gadfly.spy(rotl90(out_cd),
    Guide.ColorKey(title="Converted\ntruth-seekers"),
    Guide.xlabel("ϵ"), Guide.ylabel("Campaigners"),
    Guide.xticks(ticks=ticks), Scale.x_continuous(labels=xlabelname),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Guide.title("ρ = 0.8"),
    Scale.ContinuousColorScale(p -> get(ColorSchemes.viridis, p), minvalue=0, maxvalue=50),
    Theme(grid_color=colorant"white", panel_fill="white")
)

@everywhere function radicalized_average_cd(ϵ_start::Float64, 
                                            r_opinion::Float64, 
                                            n_normals::Int64, 
                                            n_radicals::Int64, 
                                            averaging_op, 
                                            averaging_eps)
    st = rand(n_normals)
    rd = fill(r_opinion, n_radicals)
    rd = fill(r_opinion, n_radicals)
    bc_ar = vcat(rd, st)
    v = vcat(rd, st)
    eps_rad = fill(0.0, n_radicals)
    eps_nrm = fill(ϵ_start, n_normals)
    eps = vcat(eps_rad, eps_nrm)
    tmp = zeros(Float64, n_normals)
    j = 1
    @views while true
        @inbounds for i in (n_radicals + 1):(n_radicals + n_normals)
            fa = findall(isapprox.(bc_ar[:, j], bc_ar[i, j], atol = eps[i]))
            v[i] = averaging_op(bc_ar[fa, j])
            tmp[i - n_radicals] = averaging_eps(eps[fa])
        end
        eps = vcat(eps_rad, tmp)
        bc_ar = hcat(bc_ar, v)
        if all(isapprox.(bc_ar[:, j], bc_ar[:, j + 1], atol = 10^-5))
            break
        end
        j += 1
    end
    return sum(isapprox.(@view(bc_ar[:, end]), r_opinion, atol=10^-3)) - n_radicals
end

function radical_average_parallel_cd(r_opinion::Float64, n_normals::Int64, averaging_op, averaging_eps)
    res = Array{Int64,2}(undef, 50, 50)
    for j in 1:50
        @inbounds res[:, j] = pmap(i -> radicalized_average_cd(i/100, r_opinion, n_normals, j, averaging_op, averaging_eps), 1:50)
    end
    return res
end

function rand_sim_cd(r_opinion::Float64, n_normals::Int64, n_sim::Int64, averaging_op, averaging_eps)
    rand_res = Array{Int64,3}(undef, 50, 50, n_sim)
    for i in 1:n_sim
        @inbounds rand_res[:, :, i] = radical_average_parallel_cd(r_opinion, n_normals, averaging_op, averaging_eps)
    end
    return rand_res
end

rand_res_cd = rand_sim_cd(1.0, 50, 100, mean, mean);

out_av_cd = mean(rand_res_cd, dims=3);

Gadfly.spy(rotl90(out_av_cd[:, :, 1]),
    Guide.ColorKey(title="Converted\ntruth-seekers"),
    Guide.xlabel("ϵ"), Guide.ylabel("Campaigners"),
    Guide.xticks(ticks=ticks), Scale.x_continuous(labels=xlabelname),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Scale.ContinuousColorScale(p -> get(ColorSchemes.viridis, p), minvalue=0, maxvalue=50),
    Guide.title("ρ = 1.0"),
    Theme(grid_color=colorant"white", panel_fill="white")
)

rand_res_cd = rand_sim_cd(.8, 50, 100, mean, mean);

out_av_cd = mean(rand_res_cd, dims=3);

Gadfly.spy(rotl90(out_av_cd[:, :, 1]),
    Guide.ColorKey(title="Converted\ntruth-seekers"),
    Guide.xlabel("ϵ"), Guide.ylabel("Campaigners"),
    Guide.xticks(ticks=ticks), Scale.x_continuous(labels=xlabelname),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Scale.ContinuousColorScale(p -> get(ColorSchemes.viridis, p), minvalue=0, maxvalue=50),
    Guide.title("ρ = 0.8"),
    Theme(grid_color=colorant"white", panel_fill="white")
)

function mixed_community_cd(ϵ_start::Float64, 
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
    rd = fill(r_opinion, n_radicals)
    bc_ar = vcat(st, rd)
    v = vcat(st, rd)
    eps_rad = fill(0.0, n_radicals)
    eps_nrm = rand(Uniform(0, ϵ_start), n_normals)
    ϵ = vcat(eps_nrm, eps_rad)
    tmp = zeros(Float64, n_normals)
    j = 1
    @views while true
        for i in 1:n_normals
            fa = findall(isapprox.(bc_ar[:, j], bc_ar[i, j], atol = ϵ[i] + eps()))
            @inbounds v[i] = i <= n_truthseekers ? (1 - α)*averaging_op(bc_ar[fa, j]) + α*τ : averaging_op(bc_ar[fa, j])
            tmp[i] = averaging_eps(ϵ[fa])
        end
        ϵ = vcat(tmp, eps_rad)
        bc_ar = hcat(bc_ar, v)
        if all(isapprox.(bc_ar[:, j], bc_ar[:, j + 1], atol = 10^-5))
            break
        end
        j += 1
    end
    df0 = DataFrame(bc_ar')
    names!(df0, [Symbol("$i") for i in 1:size(bc_ar, 1)])
    df1 = stack(df0)
    df1[:steps] = repeat(1:size(df0)[1], outer=size(bc_ar, 1))
    bb = Array{String,1}(undef, n_normals + n_radicals)
    bb[1:n_truthseekers] .= "truth-seeker"
    bb[n_truthseekers + 1:n_normals] .= "free rider"
    bb[n_normals + 1:n_normals + n_radicals] .= "campaigner"
    df1[:Agent] = repeat(bb, inner=size(bc_ar, 2))
    return df1
end

out_mixed_cd = mixed_community_cd(.5, false, .9, 25, 20, 10, .25, .65, mean, mean)

plot(out_mixed_cd, x=:steps, y=:value, group=:variable, color=:Agent, Geom.point, Geom.line,
    Coord.cartesian(xmax=maximum(out_mixed_cd[:steps]) + 1),
    Guide.xlabel("Time"),
    Guide.ylabel("Opinion"),
    Scale.color_discrete_hue(gen_colors, levels = ["truth-seeker", "free rider", "campaigner"], preserve_order=true),
    yintercept=[0.65], Geom.hline(style=:dot, color=colorant"grey"),
    Guide.title("τ = .65, ρ = .9, ϵ⁰ ∼ U(0, .5), α = .25"),
    Theme(panel_fill="white", point_size=1.25pt, line_width=1.5pt))

out_mixed = mixed_community(.25, false, .0, 50, 0, 0, .75, .7, mean)
out_mixed_cd = mixed_community_cd(.5, false, .0, 50, 0, 0, .75, .7, mean, mean)

plot(out_mixed, x=:steps, y=:value, color=:variable, Geom.point, Geom.line,
    Coord.cartesian(xmax=maximum(out_mixed[:steps]) + 1),
    Guide.xlabel("Time"),
    Guide.ylabel("Opinion"),
    yintercept=[.7], Geom.hline(style=:dot, color=colorant"grey"),
    Guide.title("τ = .7, ϵ = .25, α = .75"),
    Theme(key_position=:none, panel_fill="white", point_size=1.25pt, line_width=1.5pt))

plot(out_mixed_cd, x=:steps, y=:value, color=:variable, Geom.point, Geom.line,
    Coord.cartesian(xmax=maximum(out_mixed_cd[:steps]) + 1),
    Guide.xlabel("Time"),
    Guide.ylabel("Opinion"),
    yintercept=[.7], Geom.hline(style=:dot, color=colorant"grey"),
    Guide.title("τ = .7, ϵ⁰ ∼ U(0, .5), α = .75"),
    Theme(key_position=:none, panel_fill="white", point_size=1.25pt, line_width=1.5pt))

@everywhere const n_steps = 50;

@everywhere function bc_upd_ri_cd(ϵ_start::Float64, 
                                  α::Float64, τ::Float64, 
                                  r_opinion::Float64, 
                                  n_truthseekers::Int64, 
                                  n_indifs::Int64, 
                                  n_radicals::Int64, 
                                  averaging)
    n_agents = n_truthseekers + n_indifs
    rd = fill(r_opinion, n_radicals)
    bc_ar = Array{Float64,2}(undef, n_agents, n_steps + 1)
    bc_ar[:, 1] = rand(n_agents)
    eps_rad = fill(0.0, n_radicals)
    eps_nrm = fill(ϵ_start, n_agents)
    eps = vcat(eps_nrm, eps_rad)
    tmp = zeros(Float64, n_agents)
    @views @inbounds for j in 1:n_steps
        for i in 1:n_agents
            cc = vcat(bc_ar[:, j], rd)
            fa = findall(isapprox.(cc, cc[i], atol = eps[i]))
            bc_ar[i, j + 1] = i <= n_truthseekers ? (1 - α)*averaging(cc[fa]) + α*τ : averaging(cc[fa])
            tmp[i] = mean(eps[fa])
        end
        eps = vcat(tmp, eps_rad)
    end
    return bc_ar
end

function cost_sim_ri_cd(ϵ_start::Float64, 
                        α::Float64, 
                        τ::Float64, 
                        r_opinion::Float64, 
                        n_truthseekers::Int64, 
                        n_indifs::Int64, 
                        n_radicals::Int64, 
                        averaging, 
                        n_sim::Int64)
    v= SharedArray{Float64,1}(n_sim)
    @inbounds @sync @distributed for i in 1:n_sim
        res_rad = bc_upd_ri_cd(ϵ_start, α, τ, r_opinion, n_truthseekers, n_indifs, n_radicals, averaging)
        v[i] = sum(map(i->(i-τ)^2, res_rad[1:n_truthseekers, :]))
    end
    return v
end

τ = .1
ϵ = .3
α = .75
ρ1 = .3
ρ2 = .5
ρ3 = .7
nsim = 1000
without = cost_sim_ri_cd(ϵ, α, τ, ρ1, 50, 0, 0, mean, nsim)
withNarrow = cost_sim_ri_cd(ϵ, α, τ, ρ1, 50, 0, 25, mean, nsim)
withModerate = cost_sim_ri_cd(ϵ, α, τ, ρ2, 50, 0, 25, mean, nsim)
withWide = cost_sim_ri_cd(ϵ, α, τ, ρ3, 50, 0, 25, mean, nsim)

mns = DataFrame(WO = without[:], WN = withNarrow[:], WM = withModerate[:], WW = withWide[:])

plot(mns, x=Col.index, y=Col.value,
    Geom.boxplot,
    Scale.x_discrete(levels=names(mns)), 
    Guide.xlabel("Condition"),
    Guide.ylabel("SSE"),
    Guide.title("τ = $τ, ρ = $ρ1/$ρ2/$ρ3 (WN/WM/WW), ϵ = $ϵ, α = $α"),
    Theme(default_color=colorant"#66c2a5", panel_fill="white"))

@rput df_aov

R"""
m <- aov(data ~ groups, data=df_aov)
summary(m)
"""

R"TukeyHSD(m)"

R"library(lsr)"

R"etaSquared(m)"

mean(without), mean(withNarrow), mean(withModerate), mean(withWide)

std(without), std(withNarrow), std(withModerate), std(withWide)

τ = .1
ϵ = .2
α = .5
ρ = .3
nsim = 1000
without = cost_sim_ri_cd(ϵ, α, τ, ρ, 50, 0, 15, mean, nsim)
with10 = cost_sim_ri_cd(ϵ, α, τ, ρ, 50, 10, 15, mean, nsim)
with25 = cost_sim_ri_cd(ϵ, α, τ, ρ, 50, 25, 15, mean, nsim)
with50 = cost_sim_ri_cd(ϵ, α, τ, ρ, 50, 50, 15, mean, nsim);

mns = DataFrame(X= without[:], Y = with10[:], Z = with25[:], W = with50[:])

rename!(mns, :X => Symbol("0"), :Y => Symbol("10"), :Z => Symbol("25"), :W => Symbol("50"))

plot(mns, x=Col.index, y=Col.value,
    Geom.boxplot,
    Scale.x_discrete(levels=names(mns)), 
    Guide.xlabel("Free riders"),
    Guide.ylabel("SSE"),
    Guide.title("τ = $τ, ρ = $ρ, ϵ = $ϵ, α = $α"),
    Theme(default_color=colorant"#66c2a5", panel_fill="white"))

@rput df_aov

R"""
m <- aov(data ~ groups, data=df_aov)
summary(m)
"""

R"TukeyHSD(m)"

R"etaSquared(m)"

mean(without), mean(with10), mean(with25), mean(with50)

std(without), std(with10), std(with25), std(with50)


