using Distributed
addprocs()

using Gadfly, DataFrames, Colors, RCall, Compose
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

@everywhere const n_agents = 50
@everywhere const n_steps = 50

@everywhere function bc_upd_rad(ϵ::Float64, r_opinion::Float64, n_radicals::Int64, averaging)
    rd = fill(r_opinion, n_radicals)
    bc_ar = Array{Float64,2}(undef, n_agents, n_steps + 1)
    bc_ar[:, 1] = rand(n_agents)
    @views @inbounds for j in 1:n_steps, i in 1:n_agents
        cc = vcat(bc_ar[:, j], rd)
        fa = findall(@. abs(cc - cc[i]) <= ϵ)
        bc_ar[i, j + 1] = averaging(cc[fa])
    end
    return bc_ar
end

@everywhere function bc_upd_rad(ϵ::Float64, α::Float64, τ::Float64, r_opinion::Float64, n_radicals::Int64, averaging)
    rd = fill(r_opinion, n_radicals)
    bc_ar = Array{Float64,2}(undef, n_agents, n_steps + 1)
    bc_ar[:, 1] = rand(n_agents)
    @views @inbounds for j in 1:n_steps, i in 1:n_agents
        cc = vcat(bc_ar[:, j], rd)
        fa = findall(@. abs(cc - cc[i]) <= ϵ)
        bc_ar[i, j + 1] = α*averaging(cc[fa]) + (1 - α)*τ
    end
    return bc_ar
end

function cost_sim(ϵ::Float64, α::Float64, τ::Float64, r_opinion::Float64, n_radicals::Int64, averaging, n_sim::Int64)
    v= SharedArray{Float64,1}(n_sim)
    @inbounds @sync @distributed for i in 1:n_sim
        res_rad = bc_upd_rad(ϵ, α, τ, r_opinion, n_radicals, averaging)
        v[i] = sum(map(i->(i-τ)^2, res_rad))
    end
    return v
end

τ = .1
ϵ = .2
α = .5
ρ1 = .3
ρ2 = .5
ρ3 = .7
nsim = 1000
without = cost_sim(ϵ, α, τ, ρ1, 0, mean, nsim)
withNarrow = cost_sim(ϵ, α, τ, ρ1, 25, mean, nsim)
withModerate = cost_sim(ϵ, α, τ, ρ2, 25, mean, nsim)
withWide = cost_sim(ϵ, α, τ, ρ3, 25, mean, nsim);

mns = DataFrame(WO = without[:], WN = withNarrow[:], WM = withModerate[:], WW = withWide[:]);

plot(mns, x=Col.index, y=Col.value,
    Geom.boxplot,
    Scale.x_discrete(levels=names(mns)), 
    Guide.xlabel("Condition"),
    Guide.ylabel("SSE"),
    Guide.title("τ = $τ, ρ = $ρ1/$ρ2/$ρ3 (WN/WM/WW), ϵ = $ϵ, α = $α"),
    Theme(default_color=colorant"#66c2a5", panel_fill="white"))

mean(without), mean(withNarrow), mean(withModerate), mean(withWide)

std(without), std(withNarrow), std(withModerate), std(withWide) 

@rput df_aov

R"""
m <- aov(data ~ groups, data=df_aov)
summary(m)
"""

R"TukeyHSD(m)"

R"library(lsr)"

R"etaSquared(m)"

@everywhere function bc_upd_ri(ϵ::Float64, 
                               α::Float64, 
                               τ::Float64, 
                               r_opinion::Float64, 
                               n_truthseekers::Int64, 
                               n_indifs::Int64, 
                               n_radicals::Int64, 
                               averaging)
    n_agents = n_truthseekers + n_indifs 
    rd = fill(r_opinion, n_radicals)
    bc_ar = Array{Float64,2}(undef, n_agents, n_steps + 1)
    bc_ar[:, 1] = rand(n_agents)
    @views @inbounds for j in 1:n_steps, i in 1:n_agents
        cc = vcat(bc_ar[:, j], rd)
        fa = findall(@. abs(cc - cc[i]) <= ϵ)
        bc_ar[i, j + 1] = i <= n_truthseekers ? α*averaging(cc[fa]) + (1 - α)*τ : averaging(cc[fa])
    end
    return bc_ar
end

function cost_sim_ri(ϵ::Float64, 
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
        res_rad = bc_upd_ri(ϵ, α, τ, r_opinion, n_truthseekers, n_indifs, n_radicals, averaging)
        v[i] = sum(map(i->(i-τ)^2, res_rad[1:n_truthseekers, :]))
    end
    return v
end

τ = .1
ϵ = .2
α = .5
ρ1 = .3
ρ2 = .5
ρ3 = .7
nsim = 1000
without = cost_sim_ri(ϵ, α, τ, ρ1, 50, 25, 0, mean, nsim)
withNarrow = cost_sim_ri(ϵ, α, τ, ρ1, 50, 25, 25, mean, nsim)
withModerate = cost_sim_ri(ϵ, α, τ, ρ2, 50, 25, 25, mean, nsim)
withWide = cost_sim_ri(ϵ, α, τ, ρ3, 50, 25, 25, mean, nsim);

mns = DataFrame(WO = without[:], WN = withNarrow[:], WM = withModerate[:], WW = withWide[:]);

plot(mns, x=Col.index, y=Col.value,
    Geom.boxplot,
    Scale.x_discrete(levels=names(mns)), 
    Guide.xlabel("Condition"),
    Guide.ylabel("SSE"),
    Guide.title("τ = $τ, ρ = $ρ1/$ρ2/$ρ3 (WN/WM/WW), ϵ = $ϵ, α = $α"),
    Theme(default_color=colorant"#66c2a5", panel_fill="white"))

τ = .1
ϵ = .3
α = .75
ρ = .3
nsim = 1000
without = cost_sim_ri(ϵ, α, τ, ρ, 50, 0, 15, mean, nsim)
with10 = cost_sim_ri(ϵ, α, τ, ρ, 50, 10, 15, mean, nsim)
with25 = cost_sim_ri(ϵ, α, τ, ρ, 50, 25, 15, mean, nsim)
with50 = cost_sim_ri(ϵ, α, τ, ρ, 50, 50, 15, mean, nsim);

mns = DataFrame(X = without[:], Y = with10[:], Z = with25[:], W = with50[:])

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


