using Gadfly
using DataFrames
using StatsBase

function bc_upd(ϵ::Float64, α::Float64, τ::Float64, n_agents::Int, n_steps::Int)
    bc_ar = Array{Float64,2}(undef, n_agents, n_steps + 1)
    bc_ar[:, 1] = rand(n_agents)
    @views for j in 1:n_steps, i in 1:n_agents
        @inbounds bc_ar[i, j + 1] = (1 - α)*mean(bc_ar[:, j][findall(abs.(bc_ar[:, j] .- bc_ar[:, j][i]) .< ϵ)]) + α*τ
    end
    return bc_ar
end

const ϵ = 0.1
const α = 0.2
const τ = 0.7

res = bc_upd(ϵ, α, τ, 50, 30)

df₀ = res |> rotr90 |> DataFrame
rename!(df₀, [Symbol("$i") for i in 1:size(df₀, 2)])
df = df₀ |> stack
df[!, :steps] = repeat(0:size(df₀, 1) - 1, outer=size(df₀, 2))

plot(df, x=:steps, y=:value, color=:variable, Geom.point, Geom.line,
    Coord.cartesian(xmax=size(df₀, 1)),
    Guide.xlabel("Time"),
    Guide.ylabel("Opinion"),
    Guide.title("ϵ = $ϵ / α = $α / τ = $τ"),
    Theme(key_position=:none, point_size=1.5pt))
