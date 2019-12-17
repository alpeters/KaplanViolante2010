# Reproduce Kaplan Violante 2010
# Allen Peters
# November 7, 2019

# t ∈ [1,T]
# vectors n x T

using Plots, LinearAlgebra, Statistics, Random
using BenchmarkTools

# Income process
function sim_income1(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)
    Y = zeros(eltype(y_0), n, T_ret)
    ε = sqrt(σ_ε) .* randn(n, T_ret)
    η = sqrt(σ_η) .* randn(n, T_ret)
    z_t1 = sqrt(σ_z0) .* randn(n)

    for t in 1:(T_ret-1)
        z_t = z_t1 + η[:,t]
        Y[:,t] = exp.( κ[t] .+ z_t + ε[:,t] )
        z_t1 = copy(z_t)
    end
    return Y
end

function sim_income2(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)
    Y = zeros(eltype(y_0), n, T_ret)
    z_t1 = sqrt(σ_z0) .* randn(n)

    for t in 1:(T_ret-1)
        z_t = z_t1 + sqrt(σ_η) .* randn(n)
        Y[:,t] = exp.( κ[t] .+ z_t + sqrt(σ_ε) .* randn(n) )
        z_t1 = copy(z_t)
    end
    return Y
end

function sim_income3(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)
    Y = zeros(eltype(y_0), n, T_ret)
    z_t1 = sqrt(σ_z0) .* randn(n)
    z_t = similar(z_t1)
    for t in 1:(T_ret-1), i in 1:N
        z_t[i] = z_t1[i] + sqrt(σ_η) * randn()
        Y[i,t] = exp.( κ[t] + z_t[i] + sqrt(σ_ε) * randn() )
        z_t1[i] = copy(z_t[i])
    end
    return Y
end

N = 50000
T_ret = 40
T = 70
y_0 = 0.0
κ = zeros(eltype(y_0),N,T)
σ_ε = 1
σ_η = 1
σ_z0 = 1
t = 1:T

@btime Y = sim_income1(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)
@btime Y = sim_income2(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)
@btime Y = sim_income2flip(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)
@time Y = sim_income3(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)
@time Y = sim_income3flip(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)


Random.seed!(1234)
Y = sim_income2(100, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)

plot!(Y[6,:])

plot()


function sim_income2flip(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)
    Y = zeros(eltype(y_0), T_ret,n)
    z_t1 = sqrt(σ_z0) .* randn(n)

    for t in 1:(T_ret-1)
        z_t = z_t1 + sqrt(σ_η) .* randn(n)
        Y[t,:] = exp.( κ[t] .+ z_t + sqrt(σ_ε) .* randn(n) )
        z_t1 = copy(z_t)
    end
    return Y
end

function sim_income3flip(N, T_ret, σ_ε, σ_η, σ_z0, κ; y_0 = 0.0)
    Y = zeros(eltype(y_0), T_ret,n)
    z_t1 = sqrt(σ_z0) .* randn(n)
    z_t = similar(z_t1)
    for i in 1:N, t in 1:(T_ret-1)
        z_t[i] = z_t1[i] + sqrt(σ_η) * randn()
        Y[t,i] = exp.( κ[t] + z_t[i] + sqrt(σ_ε) * randn() )
        z_t1[i] = copy(z_t[i])
    end
    return Y
end
