using Plots, Random, Statistics

N = 1000000
σ_η = 0.01  # variance
σ_ε = 0.05  # variance


# IID process
y = sqrt(σ_ε).*randn(N)
# Check data generation
abs(var(y) - σ_ε) < 0.001

# Plot discretizaiton vs simulated data
tauc = tauchen(19,0.,sqrt(σ_ε))
histogram(y, bins = range(tauc.state_values[1].-step(tauc.state_values)/2, step=step(tauc.state_values),
                    length=length(tauc.state_values)+1), normalize = :probability)
scatter!(tauc.state_values,tauc.p[1,:])

rouw = rouwenhorst(19,0.,sqrt(σ_ε))
histogram!(y, bins = range(rouw.state_values[1].-step(rouw.state_values)/2, step=step(rouw.state_values),
                    length=length(rouw.state_values)+1), normalize = :probability, alpha = 0.5)
compare = scatter!(rouw.state_values,rouw.p[1,:],alpha=0.5)
plot(compare)



# Sum of IID processes
y = sqrt(σ_ε).*randn(N) + sqrt(σ_η).*randn(N)
# Check data generation
abs(var(y) - (σ_ε + σ_η)) < 0.001

# Plot discretizaiton vs simulated data
tauc_ε = tauchen(19,0.,sqrt(σ_ε))
tauc_η = tauchen(19,0.,sqrt(σ_η))
plot(tauc_ε.state_values, tauc_η.state_values)
histogram(y, bins = range(tauc.state_values[1].-step(tauc.state_values)/2, step=step(tauc.state_values),
                    length=length(tauc.state_values)+1), normalize = :probability)
scatter!(tauc.state_values,tauc.p[1,:])

rouw = rouwenhorst(19,0.,sqrt(σ_ε))
histogram!(y, bins = range(rouw.state_values[1].-step(rouw.state_values)/2, step=step(rouw.state_values),
                    length=length(rouw.state_values)+1), normalize = :probability, alpha = 0.5)
compare = scatter!(rouw.state_values,rouw.p[1,:],alpha=0.5)
plot(compare)



# With persistent process
κ = zeros(Float64,N,T_ret-1) # *** Get from data
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

N = 100000
T_ret = 2
Y_l = net_labor_income(N, T_ret, σ_ε, σ_η, 0., κ)
histogram(Y_l, alpha = 0.5)
tauc_ε = tauchen(19,0.,σ_ε)
tauc_η1 = tauchen(39,0.,var(Y_l[:,1]))


# unconditional variance of z_t
T_ret = 40
ρ = 1.
σ

plot(σ)

using QuantEcon
rouwenhorst(19,0.,sqrt(0.2)).p == rouwenhorst(19,0.,sqrt(0.1)).p

import QuantEcon.rouwenhorst
function rouwenhorst(N::Integer, ρ::Real, σ::Real, μ::Real=0.0; stationary::Bool = true)
    if stationary
        σ_y = σ / sqrt(1-ρ^2)
        m = μ / (1 - ρ)
    else
        σ_y = σ
        m = 0.0  #μ not used if non-stationary
    end

    p  = (1+ρ)/2
    Θ = [p 1-p; 1-p p]
    ψ = sqrt(N-1) * σ_y
    state_values, p = _rouwenhorst(p, p, m, ψ, N)
    MarkovChain(p, state_values)
end

function _rouwenhorst(p::Real, q::Real, m::Real, Δ::Real, n::Integer)
    if n == 2
        return [m-Δ, m+Δ],  [p 1-p; 1-q q]
    else
        _, θ_nm1 = _rouwenhorst(p, q, m, Δ, n-1)
        θN = p    *[θ_nm1 zeros(n-1, 1); zeros(1, n)] +
             (1-p)*[zeros(n-1, 1) θ_nm1; zeros(1, n)] +
             q    *[zeros(1, n); zeros(n-1, 1) θ_nm1] +
             (1-q)*[zeros(1, n); θ_nm1 zeros(n-1, 1)]

        θN[2:end-1, :] ./= 2

        return range(m-Δ, stop=m+Δ, length=n), θN
    end
end

function rouwenhorst(N::Integer, ρ::Real, σ::Real, σ_0::Real, T::Int)
    σs = zeros(eltype(σ_0), T, 1)
    σs[1] = σ_0
    [ σs[t] = ρ^2*σs[t-1] + σ  for t in 2:T ] # from Fella et al. 2019
    θNs = rouwenhorst.(N, ρ, σs, stationary = false)
    return θNs
    # unnecessary to run this multiple times because transition matrix is same
end

T_ret = 40
σ_η, σ_z0
test = rouwenhorst(19, 1., σ_η, σ_z0, T_ret)
test[1].state_values
test[2].state_values
test[1].p == test[30].p





stop





plot(rouwenhorst(19,0.,sqrt(σ_ε),2.).state_values)
y = μ + ρ









#old

#
# σ_η = 0.01
# σ_ε = 0.05
#
# # Set up check
# N = 1000000
# y = sqrt(σ_ε).*randn(N)
# abs(var(y) - σ_ε) < 0.001
# tauc = tauchen(19,0.,sqrt(σ_ε))
# rouw = rouwenhorst(19,0.,sqrt(σ_ε))
# histogram(y, bins = range(tauc.state_values[1].-step(tauc.state_values)/2, step=step(tauc.state_values),
#                     length=length(tauc.state_values)+1), normalize = :probability)
# scatter!(tauc.state_values,tauc.p[1,:])
# histogram(y, bins = range(rouw.state_values[1].-step(rouw.state_values)/2, step=step(rouw.state_values),
#                     length=length(rouw.state_values)+1), normalize = :probability, alpha = 0.5)
# compare = scatter!(rouw.state_values,rouw.p[1,:],alpha=0.5)
# plot(compare)
#
#
#
#
#
# π_εr = rouwenhorst(19,0.,sqrt(σ_ε)).p
# distt = π_εt[10,:]
# distr = π_εr[10,:]
# plot()
# # histogram!(y, bins = range(-3*sqrt(σ_ε)-3*sqrt(σ_ε)/19, stop = 3*sqrt(σ_ε)+3*sqrt(σ_ε)/19, length = 20), normalize = :probability)
# histogram!(y, bins = range(-3*sqrt(var(y))-3*sqrt(var(y))/19, stop = 3*sqrt(var(y))+3*sqrt(var(y))/19, length = 20), normalize = :probability)
# scatter!(range(-3*sqrt(var(y)), stop = 3*sqrt(var(y)), length = 19),distt)
# sqrt(var(y))*sqrt(18)
# scatter!(range(-3*sqrt(var(y)), stop = 3*sqrt(var(y)), length = 19),distr)
# ylims!(0.,0.2)
# count( x -> (x < sqrt(var(y))/6 && x > -sqrt(var(y))/6), y)/N
# π_ε[10,10]
#
#
#
# # test sum of random vars
# σ_η = 0.2
# # σ_ε
# N = 1000000
# y = sqrt(σ_ε).*randn(N) .+ sqrt(σ_η).*randn(N)
# abs(var(y) - (σ_ε + σ_η)) < 0.001
# π_ε = tauchen(19,0.,sqrt(σ_ε)).p
# π_η = tauchen(19,0.,sqrt(σ_η),5).p
# dist = (π_ε * π_η)[10,:]
# # count( x -> (x < sqrt(σ_ε)/6 && x > -sqrt(σ_ε)/6), y)/N
# # count( x -> (x < sqrt(var(y))/6 && x > -sqrt(var(y))/6), y)/N
# (π_ε * π_η)[10,10]
# plot()
# histogram!(y, bins = range(-3*sqrt(var(y))-3*sqrt(var(y))/19, stop = 3*sqrt(var(y))+3*sqrt(var(y))/19, length = 20), normalize = :probability)
# scatter!(range(-3*sqrt(var(y)), stop = 3*sqrt(var(y)), length = 19),dist)
# ylims!(0.,0.2)
#
#
#
#
#
#
#
# y = sqrt(σ_ε).*randn(N) .+ sqrt(σ_η).*randn(N)
# histogram(y, bins = 100, normalize = true)
# count( x -> (x < 0 && x > -0.05), y)/N
# π_ε = rouwenhorst(19,0.,σ_ε)
# π_η = rouwenhorst(19,0.,σ_η)
#
#
# plot(π_η[1])
# π_η.p[1,:] .== π_η.p[2]
# π_η.p[10,10]
