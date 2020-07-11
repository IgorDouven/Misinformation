using Distributions
using Gadfly
using Compose
using DataFrames
using StatsBase
using Distances

function gen_colors(n) # to create your own colors, here based on one of the brewer series
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

const n_agents = 50
const n_steps = 50
const start = rand(n_agents)

bc_ar = Array{Float64,2}(undef, n_agents, n_steps + 1)

function bc_upd(ϵ::Float64, α::Float64, τ::Float64, averaging)
    bc_ar[:, 1] = start
    for j in 1:n_steps, i in 1:n_agents
        bc_ar[i, j + 1] = (1 - α)*averaging(bc_ar[:, j][findall(abs.(bc_ar[:, j] .- bc_ar[i, j]) .< ϵ)]) + α*τ
    end
    return bc_ar
end

ϵ = 0.1
α = 0.5
τ = 0.7

res = bc_upd(ϵ, α, τ, mean);

df = DataFrame(res')
names!(df, [Symbol("$i") for i in 1:n_agents])
df = stack(df)
df[:steps] = repeat(1:n_steps + 1, outer=n_agents);

plot(df, x=:steps, y=:value, color=:variable, Geom.point, Geom.line,
    Coord.cartesian(xmax=51),
    Guide.xlabel("Time"),
    Guide.ylabel("Opinion"),
    Guide.title("ϵ = $ϵ / α = $α / τ = $τ / arithmetic mean"),
    yintercept=[τ], Geom.hline(style=:dot, color=colorant"grey"),
    Guide.annotation(compose(context(), Compose.text(22, τ + 0.015, "τ"), fontsize(11pt))),
    Theme(key_position=:none, point_size=1.25pt,
        major_label_font_size=12pt,
        minor_label_font_size=8pt))

