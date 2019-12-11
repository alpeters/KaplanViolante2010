# A development file for testing implementation of non-stationary rouwenhorst method (à la FGP2019)
# built on existing function for stationary processes from package QuantEcon

# Citations
# FGP2019: Fella, Gallipoli, Pan, 2019
# KV2010: Kaplan, Violante, 2010

using QuantEcon, Plots, Random, Statistics

function simulate_PTincome(N, T, σ_u, σ_ε::Real, ρ::Real, σ_0)
# Simulates persistent-transitory, non-stationary income process as per FGP2019
# Only gaussian persistent shock is implemented
# Inputs are standard deviations
# σ_0 is the standard deviation of η_0 (denoted simply as σ in FGP2019)
    Y = zeros(Float64, N, T)
    η_t1 = σ_0 .* randn(N)

    for t in 1:T
        η_t = ρ * η_t1 + σ_ε .* randn(N)
        Y[:,t] = exp.( η_t + σ_u .* randn(N) )
        η_t1 = copy(η_t)
    end
    return Y'
end


### Check simulation
σ_ε = sqrt(0.0161)    #variance is σ_η^2=0.0161 in FGP2019
σ_u = 0. #sqrt(0.063)   #variance is σ_u^2=0.063 in FGP2019
σ_η0 = 0.             # In FGP2019
ρ = 0.9            # ρ ∈ {0.95,0.98,1} in FGP2019 section 4.1.1
N = 10000
T = 20
Random.seed!(1234)
Y = simulate_PTincome(N, T, σ_u, σ_ε, ρ, σ_η0)
plot(1:T,Y[:,1])
plot(var(Y,dims=2))
histogram(Y[1,:], bins = 100)
histogram!(Y[end,:], bins = 100, alpha = 0.5)
histogram!(Y[end-1,:], bins = 100, alpha = 0.5)

### Test rouwenhorst_ns
σ_ε = sqrt(0.0161)    #variance is σ_η^2=0.0161 in FGP2019
σ_u = sqrt(0.063)   #variance is σ_u^2=0.063 in FGP2019
σ_η0 = 0.             # In FGP2019
ρ = 0.98            # ρ ∈ {0.95,0.98,1} in FGP2019 section 4.1.1
T = 15
test = rouwenhorst_ns(19, T, 0.9, σ_ε, σ_η0)
test[1].state_values
test[1].p
heatmap(test[15].p)

# Compare to stationary
test_stat = rouwenhorst(19, 0.9, σ_ε)
heatmap(test_stat.p)
maximum(test[15].p - test_stat.p)

# Difference due to initial variance? This is constrained for stationary case
test = rouwenhorst_ns(19, T, 0.9, σ_ε, σ_ε/sqrt(1-ρ^2))
maximum(test[15].p - test_stat.p)
# virtually identical at steady state
heatmap(test[5].p)


### Markov Chain simulation for non-stationary
@doc doc"""
Simulate one sample path of the non-stationary vector of Markov chains `mcs`.
The resulting vector has the state values of `mcs` as elements.

### Arguments

- `mcs::Vector{MarkovChain}` : Vector of MarkovChains.
- `;init::Int` : Index of initial state

### Returns

- `X::Vector` : Vector containing the sample path, with length
  length(mcs)+1 (includes initial state)
"""
function simulate_ns(mcs::Vector, init::Int)
    ind_0 = init
    ind = zeros(Int, length(mcs))
    X = zeros(Real, length(mcs))

    for t in eachindex(mcs)
        X[t] = simulate(mcs[t], 2, init = ind_0)[2]
        ind[t] = findmin( abs.(mcs[t].state_values .- X[t]) )[2]
        if t < length(mcs)
            ind_0 = ind[t] #findmin( abs.(mcs[t].state_values .- X[t]) )[2]
        end
    end
    ind = [init; ind]
    X = [mcs[1].state_values[init]; X]
end

# Check that my simulation function gives same results as QuantEcon's for
# stationary process
T = 25
N = 100000
ρ = 0.9
mc_qe = rouwenhorst(19, ρ, σ_ε)

# QuantEcon's MC simulation
# simulate(mc, T, init = 10) |> plot
Y_mc_qe = zeros(T+1,N)
[Y_mc_qe[:,i] = simulate(mc_qe, T+1, init = 10) for i in 1:N]
plot(0:T, var(Y_mc_qe, dims = 2), label="QE MC", xlabel="t", ylabel="Variance", legend=:bottomright)

# DGP simulation
Y = log.(simulate_PTincome(N, T, 0, σ_ε, ρ, 0.))
plot!(1:T, var(Y, dims = 2), label="DGP")

# My MC simulation with QE MC
mcs = fill(mc_qe,T)
Y_mc_qe1 = zeros(length(mcs)+1,N)
[Y_mc_qe1[:,i] = simulate_ns(mcs, 10) for i in 1:N]
plot!(0:T, var(Y_mc_qe1, dims = 2), label="MC")

# Compare to theoretical variance
expected_vars = function(σ, ρ, σ_y0, T)
    var_y = zeros(T)
    for t in 1:T
        var_y[t] = ρ^2 * σ_y0^2 + σ^2
        σ_y0 = sqrt(var_y[t])
    end
    return var_y
end
scatter!(1:T, expected_vars(σ_ε,ρ,0.,T), label = "Theory")

# My MC simulation with my MC
mcs_ns = rouwenhorst_ns(19, T, ρ, σ_ε, 0.)
Y_mc_ns = zeros(length(mcs_ns)+1,N)
[Y_mc_ns[:,i] = simulate_ns(mcs_ns, 10) for i in 1:N]
plot!(0:T, var(Y_mc_ns, dims = 2), label="MC all mine")

# Means
plot(mean(Y, dims=2))
plot!(mean(Y_mc_qe, dims = 2))
plot!(mean(Y_mc_ns, dims = 2))

# Check for ρ=1
ρ=1
T=30
scatter(1:T, expected_vars(σ_ε,ρ,0.,T), label = "Theory", legend=:bottomright)

# DGP simulation
Y = log.(simulate_PTincome(N, T, 0, σ_ε, ρ, 0.))
plot!(1:T, var(Y, dims = 2), label="DGP")

# My MC simulation with my MC
mcs_ns = rouwenhorst_ns(19, T, ρ, σ_ε, 0.)
Y_mc_ns = zeros(length(mcs_ns)+1,N)
[Y_mc_ns[:,i] = simulate_ns(mcs_ns, 10) for i in 1:N]
plot!(0:T, var(Y_mc_ns, dims = 2), label="MC all mine")


### Now let's try to discretize the transitory-persistent process!!!
σ_ε = sqrt(0.0161)    #variance is σ_η^2=0.0161 in FGP2019
σ_u = sqrt(0.063)   #variance is σ_u^2=0.063 in FGP2019
σ_η0 = 0.             # In FGP2019
ρ = 1.            # ρ ∈ {0.95,0.98,1} in FGP2019 section 4.1.1
N = 200000
T = 40
n = 11

# Simulate DGP
Random.seed!(1234)
Y = simulate_PTincome(N, T, σ_u, σ_ε, ρ, σ_η0)
plot(1:T, var(Y, dims = 2), label="DGP", legend=:bottomright)

# Markov chain for persistent component
mcs_FGP = rouwenhorst_ns(n, T, ρ, σ_ε, σ_η0)
η = zeros(length(mcs_FGP)+1,N)
[η[:,i] = simulate_ns(mcs_FGP, round(Int,n/2)) for i in 1:N]
# plot!(0:T, var(η, dims = 2), label="eta_FGP")

# Markov simulation with actual transitory component
u = σ_u*randn(size(η))
Y_m = exp.(η + u)
plot!(0:T, var(Y_m, dims = 2), label="Y_m_FGP")

# Markov simulation with discretized transitory component
# Transitory component
mc_trans = rouwenhorst(n, 0., σ_u)
u_mc = zeros(T+1,N)
[u_mc[:,i] = simulate(mc_trans, T+1, init = round(Int,n/2)) for i in 1:N]
# plot!(0:T, var(u_mc, dims = 2), label="u_mc", xlabel="t", ylabel="Variance", legend=:bottomright)

# Put together
Y_mm = exp.(η + u_mc)
plot!(0:T, var(Y_mm, dims = 2), label="Y_mm_FGP")
xlims!(1.,35.)
ylims!(0.,1.8)

Y_mm = Y_mm[2:end,:]
sim_error = (var(Y_mm, dims=2)-var(Y,dims=2))./var(Y,dims=2)*100
sim_error[end]
sim_error = (mean(Y_mm, dims=2)-mean(Y,dims=2))./mean(Y,dims=2)*100
sim_error[end]
# These compare quite well with FPG2019 Table 1


plot()
### Then we calculate expectations!!!

# PROBLEMS
# NONE


stop



# Boneyard
# function simulate_AR1(N, T, σ_ε, ρ, σ_0)
# # η_t = ρ_t *η_{t-1} + ε_t, ε_t ~ N(0,σ_εt)
# # σ_0 is the standard deviation of η_0 (denoted simply as σ in FGP2019)
#     Y = zeros(Float64, N, T-1)
#     η_t1 = sqrt(σ_0) .* randn(N)
#
#     for t in 1:(T-1)
#         η_t = η_t1 + sqrt(σ_ε) .* randn(N)
#         η_t1 = copy(η_t)
#     end
#     return Y
# end


# function rouwenhorst(N::Integer, ρ::Real, σ::Real, μ::Real=0.0; stationary::Bool=true)
# # Note: μ is not implemented for non-stationary case
#     if stationary
#         return rouwenhorst(N, ρ, σ, μ)
#     else
#         return rouwenhorst_ns(N, T, repeat(ρ,T), repeat(σ,T))
# end

# Y[:,2]
# test = function()
#     pY = fill(plot(1:T-1,Y[:,1]),10)
#     for i in 1:10
#         pY[i] = plot(1:T-1,Y[:,i])
#     end
#     return pY
# end
# test()
# display(pY[2])
#
# plot()
# for i in 1:10
#     display(plot!(1:T+1,Y_mc_qe[:,i]))
# end
# plot()
# for i in 1:10
#     display(plot!(1:T+1,Y_mc_qe1[:,i]))
# end
#
#
# #######
# plot()
# for i in 100:110
#     plot!(Y[:,i], legend=:none) |>display
#     plot!(Y_m[:,i], line=:dash) |>display
#     # plot!(η[:,i]) |>display
# end
