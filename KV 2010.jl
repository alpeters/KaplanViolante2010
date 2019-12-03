
# Reproduce Kaplan Violante 2010
# Allen Peters
# November 7, 2019

# t ∈ [1,T]
# vectors N x T

using Plots, LinearAlgebra, Statistics, Random, Roots, QuantEcon, Interpolations
using BenchmarkTools

# Parameters
## Demographics
N = 50000
T_ret = 35
T = 70
t = 1:T
age = t .+ 25

## Uncondtional probability of surviving to age t, from NCHS 1992 (Chung 1994)
age_data = [60:5:85; 95.]
S_data = [85.89; 80.08; 72.29; 61.52; 48.13; 32.33; 0]./100
S_interp = LinearInterpolation(age_data, S_data)
ξ_ret = map(S_interp,collect(60:95))
ξ = [ones(Float64,T_ret-1); ξ_ret]
# Visual check
# scatter(t,ξ)

## Preferences
u(c, γ = 2.) = 1/(1-γ)*c^(1-γ)

## Discount factor and interest rate
r = 0.03
β = 0.971
WI_ratio = 2.5

## Income process
# κ = zeros(eltype(y_0),N,T) # *** Get from data
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

## Test
# function linreg(x,y)
#     x = [x ones(length(y))]
#     x'*x \ x'*y
# end
#
# Random.seed!(1234)
# plot(labor_income(1,T_ret, 0., 0., 0., κ; y_0 = 0.0)')
# any(labor_income(1,T_ret, 0., 0., 0., κ; y_0 = 0.0) .== 1.) != false
#
# Random.seed!(1234)
# testy = labor_income(10, T_ret, 0., σ_η, 0., κ; y_0 = 0.0)'
# plot(testy)
# i = 7
# linreg(testy[i,1:end-1],testy[i,2:end])[2]
#
# Random.seed!(1234)
# testy = labor_income(10,T_ret, σ_ε, 0., 0., κ; y_0 = 0.0)'
# plot!(testy)
# i = 6
# linreg(testy[i,1:end-1],testy[i,2:end])[2]

## Initial wealth
A_0 = zeros(Float64,N)
# empirical one too?

## Borrowing limit
# how to calc? A_min =   # NBC
A_min = 0 # ZBC

# Gross earnings Y_tilde
τ_b = 0.258
τ_ρ = 0.768
τ(Y_tilde,τ_s) = τ_b*(Y_tilde-(Y_tilde^(-τ_ρ) + τ_s)^(-1/τ_ρ))
Y_tilde_fn(Y, τ_s) = find_zero(Y_tilde -> Y_tilde - τ(Y_tilde,τ_s) - Y, Y) # Select an algorithm, with a derivative
# Linear interpolation may be much faster
## Test
# Random.seed!(1234)
# N = 1000
# Y_l = net_labor_income(N, T_ret, σ_ε, σ_η, σ_z0, κ)
# i = rand(1:N)
# plot(Y_l[i,:])
# τ_s = 1.45
# Y_tilde = Y_tilde_fn.(Y_l,τ_s)
# plot!(Y_tilde[i,:])
# scatter(Y_tilde[i,:], 1. .- Y_l[i,:] ./ Y_tilde[i,:], ylabel="tax rate", xlabel="gross income", legend=false)
# scatter(Y_tilde, 1. .- Y_l ./ Y_tilde, ylabel="tax rate", xlabel="gross income", legend=false)


# Iterate to calibrate τ_s
function gross_labor_income(Y_l; τ_s_0 = 0.25, Kp = 10., Kd = 0.1, tol = 1E-3, max_iter = 1000, nonconvergence_message = true, verbose = false)
    #iterate to
    Y_tilde = similar(Y_l)
    τ_s = τ_s_0
    err = tol + 0.1
    iter = 1
    while abs(err) > tol && iter <= max_iter
        Y_tilde = Y_tilde_fn.(Y_l,τ_s)
        err_old = err
        err = sum(τ.(Y_tilde,τ_s))/sum(Y_l) - 0.25
        verbose ? println(err) : nothing
        τ_s_old = τ_s
        τ_s += -Kp*err + Kd*(err-err_old)
        iter += 1
    end
    iter <= max_iter ? (Y_tilde = Y_tilde, τ_s = τ_s, iters = iter - 1, err = err) : ( !nonconvergence_message ? (Y_tilde = Y_tilde, iters = iter - 1) : println("Did not converge after $(iter-1) iterations") )
end
## Test
# Random.seed!(1234)
# N = 100
# Y_l = net_labor_income(N, T_ret, σ_ε, σ_η, σ_z0, κ)
# sol = gross_labor_income(Y_l, τ_s_0 = 0.1, Kp = 10, Kd = 1., tol = 1E-3, max_iter = 100, nonconvergence_message = true)
# sol.err
# sol.iters
# sol.τ_s


# Social security benefits
function gross_SS_income(Y_tilde)
    Y_tilde_SS = mean(Y_tilde, dims = 2)
    y_ave = mean(Y_tilde_SS)
    SS_schedule(y) = y < 0.18*y_ave ? 0.9*y : (y > 1.1*y_ave ? 0.9*0.18*y_ave + 0.32*(1.1-0.18)*y_ave + 0.15*(y-1.1*y_ave) : 0.9*0.18*y_ave + 0.32*(y-0.18*y_ave) )
    Y_SS = 0.45*y_ave/SS_schedule(y_ave).*replace(SS_schedule, Y_tilde_SS)
end

# ## Test, using Y_l instead of Y_tilde for simplicity
# Random.seed!(1234)
# N = 10000
# Y_l = net_labor_income(N, T_ret, σ_ε, σ_η, σ_z0, κ)
#
# # Bendpoints
# Y_ave = mean(Y_l,dims=2) #lifetime average income
# y_ave = mean(Y_ave)
# bp1 = 0.18*y_ave
# bp2 = 1.1*y_ave
# Y_l_SS = gross_SS_income(Y_l)
# scatter(Y_ave,Y_l_SS)
# vline!([bp1 bp2])
# xlims!(0.,1.)
# ylims!(0.,0.4)
#
# # Test average income gets 45% of earnings
# y_ave = mean(sum(Y_l,dims=2)) #average lifetime net labour earnings
# ind_mean = findmin(abs.( sum(Y_l,dims=2) .- y_ave ))
# mean_ind = ind_mean[2][1]
# sum(Y_l[mean_ind,:])
# mean(Y_l[mean_ind,:])
# Y_l_SS[mean_ind] - 0.45*mean(Y_l[mean_ind,:]) < 1E-4
# plot(Y_l[mean_ind,:])
# Y = hcat(Y_l, repeat(Y_l_SS, 1, T-(T_ret-1)))
# plot(Y[mean_ind,:], ylims=(0,:))

# Test income generation
# Random.seed!(1234)
# Y_l = net_labor_income(N, T_ret, σ_ε, σ_η, σ_z0, κ)
# (Y_tilde, τ_s) = gross_labor_income(Y_l, τ_s_0 = 1.45)
# Y_tilde_SS = gross_SS_income(Y_tilde)
# Y_SS = Y_tilde_SS - τ.(0.85.*Y_tilde_SS, τ_s)
# Y = hcat(Y_l, repeat(Y_SS, 1, T-(T_ret-1)) )
# for num = 1:10
#     num == 1 ? plot() : nothing
#     i = rand(1:N)
#     display(plot!(Y[i,:], ylims=(0,8), label=i, title="Net income"))
# end


# Policy function via endogenous grid method, as per footnote 23
a_min = 0.
a_gridpoints = 100
a_max = 10.
A_vals = exp10.( range(log10(a_min+1), stop = log10(a_max+1), length = a_gridpoints) ) .- 1

y_ave_gridpoints = 19

# Discretize permanent component of income
# 39 equally spaced points on an age-varying grid chosen to match the age-specific unconditional variances



# Discretize transitory component of income
 #transitory component is approximated with 19 equally spaced points



# # Test
# scatter(A_vals, ones(length(A_vals)))
# histogram(A_vals,bins = 30)

for t in reverse(t)
    if t < T_ret

    elseif t < T && t >= T_ret

    elseif t == T

    end
end


# linear interpolation of decision rule










# Main program
Random.seed!(1234)
Y_l = net_labor_income(N, T_ret, σ_ε, σ_η, σ_z0, κ)
(Y_tilde, τ_s) = gross_labor_income(Y_l, τ_s_0 = 1.45, verbose = true)
Y_tilde_SS = gross_SS_income(Y_tilde)
Y_SS = Y_tilde_SS - τ.(0.85.*Y_tilde_SS, τ_s)
Y = hcat(Y_l, repeat(Y_SS, 1, T-(T_ret-1)) )


# calculate individual debt limits ??

# solve..
