# Verify discretization technique by simulating income generating process
# and markov chain simulation for T_ret periods and then comparing distributions

using Plots, LinearAlgebra, Statistics, Random, Roots, QuantEcon, Interpolations
using BenchmarkTools

# Simulate income generating process
# Parameters
## Demographics
N = 1000000
T_ret = 10
κ = zeros(T_ret)

# Directly from KV2010
σ_η = 0.01
σ_z0 = 0.15
σ_ε = 0.05

function net_labor_income(N, T_ret, σ_ε, σ_η, σ_z0, κ)
    Y = zeros(Float64, N, T_ret-1)
    z_t1 = sqrt(σ_z0) .* randn(N)

    for t in 1:(T_ret-1)
        z_t = z_t1 + sqrt(σ_η) .* randn(N)
        Y[:,t] = exp.( κ[t] .+ z_t + sqrt(σ_ε) .* randn(N) )
        z_t1 = copy(z_t)
    end
    return Y
end

Y_igp = net_labor_income(N, T_ret, σ_ε, σ_η, σ_z0, κ)
histogram(Y_igp[:,1], alpha = 0.3, bins = 100)
histogram!(Y_igp[:,end], alpha = 0.3, bins = 100)
means = mean(Y_igp,dims=1)'
vars = var(Y_igp,dims=1)'
plot(1:(T_ret-1), means, label="means")
plot!(1:(T_ret-1), vars, label="vars", xlabel="t")



# Tauchen discretization
tauc_ε = tauchen(19,0.,sqrt(σ_ε))
tauc_η = tauchen(39,0.99999,sqrt(σ_η))

tauc_ε.p
