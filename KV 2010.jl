
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

## Uncondtional probability of surviving to age t
# Data from NCHS 1992 (Chung 1994)
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
WI_ratio = 2.5  # aggregate wealth-income ratio

## Income process
# KV 2010: "The estimated profile peaks after 21 years of labor market experience at roughly twice the initial value, and
# then it slowly declines to about 80 percent of the peak value."

# From master's thesis (estimated from Hansen?)
# Ey = [0.5676772162 0.6084932081 0.6493091999 0.6901819595 0.7290110811 0.7678969704 0.8067828597 0.845668749 0.8845546383 0.9234405276 0.9623264169 0.9773698632 0.9899155296 1.0075135233 1.0226137373 1.0376571835 1.0527573975 1.0678008437 1.0828442899 1.0979445039 1.1129879501 1.1139530014 1.1149180526 1.1159398716 1.1169049229 1.1178699742 1.1188350254 1.1198000767 1.120765128 1.1217301792 1.1226952305 1.1183241159 1.1139530014 1.1095818868 1.1052107722 1.1008396577 1.0964685431 1.0920974285 1.087726314 1.0833551994 1.0789840849 1.0419147626 1.0048454404 0.9677761182 0.9307635637]
# plot([20:64],Ey'./maximum(Ey))

# From Huggett et al. 2006, PSID data up to 1992
age_y = [20; 25; 30; 35; 40; 45; 50; 55; 57.5]
y_data = [55.12; 76.59; 91.38; 101.63; 106.67; 107.64; 106.67; 103.58; 100]
y_interp = LinearInterpolation(age_y, y_data, extrapolation_bc = Line())
# plot([0:39],κ./maximum(κ), ylims=(0,1))
# Assume KV's description is about levels. Then my data peaks slightly later and doesn't decline as much as KV describes.
κ = log.(map(y_interp,collect(20:59)))

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
# Optional: Do empirical one too
# "Precisely, we target the empirical distribution of financial wealth-earnings ratios in the population of house
# holds aged 20-30 in the SCF. We assume that the initial draw of earnings is independent of the initial draw of this
# ratio, since in the data the empirical correlation is 0.02."

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
function gross_labor_income(Y_l; τ_s_0 = 0.25, Kp = .8, Kd = .1, tol = 1E-3, max_iter = 1000, nonconvergence_message = true, verbose = false)
    #iterate to
    Y_tilde = similar(Y_l)
    τ_s = τ_s_0
    err = tol + 0.1
    iter = 1
    while abs(err) > tol && iter <= max_iter
        Y_tilde = Y_tilde_fn.(Y_l,τ_s)
        err_old = err
        err = sum(τ.(Y_tilde,τ_s))/sum(Y_l) - 0.25
        verbose ? println(err, " ", τ_s) : nothing
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
(Y_tilde, τ_s) = gross_labor_income(Y_l, τ_s_0 = 0.031, verbose = true) #0.031 - Initial guess from Gouveia Strauss 1994
Y_tilde_SS = gross_SS_income(Y_tilde)
Y_SS = Y_tilde_SS - τ.(0.85.*Y_tilde_SS, τ_s)
Y = hcat(Y_l, repeat(Y_SS, 1, T-(T_ret-1)) )


# calculate individual debt limits ??

# solve..
