# Replicate Kaplan Violante 2010
# Allen Peters
# November 7, 2019
using Plots, LinearAlgebra, Statistics, Random, Roots, Interpolations
using QuantEcon

###############################################################################
# Omit if using my development version of QuantEcon
# command 'free QuantEcon' to revert to proper version of QuantEcon

import QuantEcon._rouwenhorst

@doc doc"""
Extended Rouwenhorst's method to approximate non-stationary AR(1) processes,
following Fella, Gallipoli, Pan (2019).

The process follows

```math
    y_t = \rho_t y_{t-1} + \epsilon_t
```

where ``\epsilon_t \sim N (0, \sigma_t^2)``

##### Arguments
- `N::Integer` : Number of points in markov process
- `T::Integer` : Length of simulation
- `ρ::Array or Real` : Persistence parameter(s) in AR(1) process, can be >=1
- `σ::Array or Real` : Standard deviation(s) of random component of AR(1) process
- `σ_y0::Real` : Standard deviation of initial y value y_0

##### Returns

- `mc::Vector{MarkovChain}` : Vector of Markov chains (length T) holding
                                the state values and transition matrix

"""
function rouwenhorst_ns(N::Integer, T::Integer, ρ::Array, σ::Array, σ_y0::Real)
    #for input to _rouwenhorst(), because μ is not implemented
    m = 0.0

    # a cheap way to initialize a vector of MarkovChains
    MarkovChains = fill(rouwenhorst(N, 0.9, σ[1]), T)

    for t in 1:T
        σ_yt = sqrt(ρ[t]^2 * σ_y0^2 + σ[t]^2)
        p  = (1+ρ[t]*σ_y0/σ_yt)/2
        Θ = [p 1-p; 1-p p]
        ψ = sqrt(N-1) * σ_yt
        state_values, p = _rouwenhorst(p, p, m, ψ, N)
        MarkovChains[t] = MarkovChain(p, state_values)
        σ_y0 = σ_yt
    end
    return MarkovChains
end

function rouwenhorst_ns(N::Integer, T::Integer, ρ::Real, σ::Real, σ_y0::Real)
    rouwenhorst_ns(N, T, ρ.*ones(T), σ.*ones(T), σ_y0)
end

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
            ind_0 = ind[t]
        end
    end
    ind = [init; ind]
    X = [mcs[1].state_values[init]; X]
end
###############################################################################

# Parameters
## Demographics
N = 50000
T_ret = 35
T = 70
t = 1:T   # period T is last period of life, T_ret is last period of work
age = t .+ 24

## Uncondtional probability of surviving to age t
# Data from NCHS 1992 (Chung 1994)
age_data = [60:5:85; 95.]
S_data = [85.89; 80.08; 72.29; 61.52; 48.13; 32.33; 0]./100
S_interp = LinearInterpolation(age_data, S_data)
ξ_ret = map(S_interp,collect(60:95))
ξ_raw = [ones(Float64,T_ret); ξ_ret]
plot(age, ξ_raw[1:end-1], label = "Data",
 title="Unconditional Survival Probability", xlabel="Age", ylabel="Data",
 linestyle=:dash)
ξ = [ones(Float64,T_ret); ξ_ret .+ (1-ξ_ret[1])]
plot!(age, ξ[1:end-1], title="Unconditional Survival Probability",
 label="Adjusted")
savefig("survival_prob.png")

## Preferences
γ = 2.

## Discount factor and interest rate
r = 0.03
β = 0.971
WI_ratio = 2.5  # aggregate wealth-income ratio

## Income process
# KV 2010: "The estimated profile peaks after 21 years of labor market experience
# at roughly twice the initial value, and then it slowly declines to about 80
# percent of the peak value."

# From master's thesis (estimated from Hansen?)
# Ey = 161/0.768.*[0.5676772162;0.6084932081;0.6493091999;0.6901819595;0.7290110811;
#  0.7678969704;0.8067828597;0.845668749;0.8845546383;0.9234405276;0.9623264169;
#  0.9773698632;0.9899155296;1.0075135233;1.0226137373;1.0376571835;1.0527573975;
#  1.0678008437;1.0828442899;1.0979445039;1.1129879501;1.1139530014;1.1149180526;
#  1.1159398716;1.1169049229;1.1178699742;1.1188350254;1.1198000767;1.120765128;
#  1.1217301792;1.1226952305;1.1183241159;1.1139530014;1.1095818868;1.1052107722;
#  1.1008396577;1.0964685431;1.0920974285;1.087726314;1.0833551994;1.0789840849;
#  1.0419147626;1.0048454404;0.9677761182;0.9307635637]
# # plot!([20:64],Ey./maximum(Ey).*161/0.768)
# y_interp = LinearInterpolation(20:64, Ey, extrapolation_bc = Line())


# From Huggett et al. 2006, PSID data up to 1992
age_y = [20; 25; 30; 35; 40; 45; 50; 55; 57.5]
y_data = [55.12; 76.59; 91.38; 101.63; 106.67; 107.64; 106.67; 103.58; 100]
 # trim down last couple points and compress ages to match KV's description
y_data[8] *= 0.95
y_data[9] *= 0.85
age_y = range(25, stop=57.5, length=length(age_y))
y_data .*= 15/55.12 # scale to match plot in KV2010, work in 1000's
# Assume KV's description is about levels. Then my data peaks slightly later and
# doesn't decline as much as KV describes.
y_interp = LinearInterpolation(age_y, y_data, extrapolation_bc = Line())
κ = log.(map(y_interp,collect(25:59)))

# Net labour income
function Y(N, T_ret, σ_ε, σ_η, σ_z0, κ)
# Takes standard deviations as arguments (not varicances)!
    Y = zeros(Float64, N, T_ret)
    z = similar(Y)
    ε = similar(Y)
    z[:,1] = σ_z0 .* randn(N) + σ_η .* randn(N)

    for t in 1:(T_ret)
        ε[:,t] = σ_ε .* randn(N)
        Y[:,t] = exp.( κ[t] .+ z[:,t] + ε[:,t] )
        t < (T_ret) ? z[:,t+1] = z[:,t] + σ_η .* randn(N) : nothing
    end
    return (Y, z, ε)
end

# Gross earnings Y_tilde
τ_b = 0.258
τ_ρ = 0.768
τ(Y_tilde,τ_s) = τ_b*(Y_tilde-(Y_tilde^(-τ_ρ) + τ_s)^(-1/τ_ρ))
Y_tilde_fn(Y, τ_s) = find_zero(Y_tilde -> Y_tilde - τ(Y_tilde,τ_s) - Y, Y)

# Iterate to calibrate τ_s (Returns gross labour income)
function G(Y_l; τ_s_0 = 0.25, Kp = .4, Kd = .07, tol = 1E-2, max_iter = 1000,
 nonconvergence_message = true, verbose = false)
    verbose ? println("Solving for gross labour income") : nothing
    #iterate to
    Y_tilde = similar(Y_l)
    τ_s = τ_s_0
    err = tol + 0.1
    iter = 1
    while abs(err) > tol && iter <= max_iter
        Y_tilde = Y_tilde_fn.(Y_l,τ_s)
        err_old = err
        err = sum(τ.(Y_tilde,τ_s))/sum(Y_l) - 0.25
        verbose ? println("error=", err, " τs=", τ_s) : nothing
        τ_s_old = τ_s
        τ_s += -Kp*err + Kd*(err-err_old)
        iter += 1
    end
    iter <= max_iter ?
        (Y_tilde = Y_tilde, τ_s = τ_s, iters = iter - 1, err = err) :
        ( !nonconvergence_message ? (Y_tilde = Y_tilde, iters = iter - 1) :
        println("Did not converge after $(iter-1) iterations") )
end


# Social security benefits (After tax retirement income)
function P(Y_tilde_l)
    Y_tilde_l_iave = mean(Y_tilde_l, dims = 2)
    # Denoted \tilde{Y}_i^SS in the paper just to be confusing.
    # This is each individual's average lifetime earnings
    y_tilde_l_ave = mean(Y_tilde_l_iave) # Average lifetime earnings across individuals
    SS_schedule(y) = y < 0.18*y_tilde_l_ave ? 0.9*y : (y > 1.1*y_tilde_l_ave ?
     0.9*0.18*y_tilde_l_ave + 0.32*(1.1-0.18)*y_tilde_l_ave +
     0.15*(y-1.1*y_tilde_l_ave) : 0.9*0.18*y_tilde_l_ave + 0.32*
     (y-0.18*y_tilde_l_ave) )
    Y_tilde_SS = 0.45*y_tilde_l_ave/SS_schedule(y_tilde_l_ave).*
     replace(SS_schedule, Y_tilde_l_iave)
    Y_SS = Y_tilde_SS - τ.(0.85.*Y_tilde_SS, τ_s)
end

# Directly from KV2010
σ_η = sqrt(0.01)
σ_z0 = sqrt(0.15)
σ_ε = sqrt(0.05)

# Discretize permanent component of income
# 39 equally spaced points on an age-varying grid chosen to match the
# age-specific unconditional variances
z_gridpoints = 39
mcs_z = rouwenhorst_ns(z_gridpoints, T_ret, 1., σ_η, σ_z0)
# Discretize transitory component of income
# transitory component is approximated with 19 equally spaced points
ε_gridpoints = 19
mc_ε = rouwenhorst(ε_gridpoints, 0., σ_ε)

function invert_rule(A, A_vals; extrapolation_bc = Throw())
    #linear interpolation of a column of Y on that column of X, evaluated at point x
    A_vals_interp = similar(A)
    for col in eachindex(A[1,:])
        li = LinearInterpolation(A[:,col], A_vals, extrapolation_bc=extrapolation_bc)
        A_vals_interp[:,col] = li(A_vals)
    end
    return A_vals_interp
end

function invert_rule_3d(A3d, A_vals; extrapolation_bc = Throw())
    wrapper(A) = invert_rule(A,A_vals,extrapolation_bc = extrapolation_bc)
    A3d_inv = mapslices(wrapper, A3d, dims=1)
end


# Policy function via endogenous grid method, as per footnote 23
# 100 exponentially spaced grid points for assets.
## Borrowing limit defined by a_min. Default is ZBC
# how to calc NBC? A_min = ...
a_gridpoints = 100
a_min = 0.
a_max = 220.

# The grid for lifetime average earnings has 19 points.
# I will build a grid on P(y_tilde) rather than y_tilde because
# don't have to worry about matching mean for accuracy
PY_tilde_gridpoints = 19
# Based on simulating the income process:
# histogram(Y_SS, bins = 100)
# minimum(Y_SS)
# maximum(Y_SS)
PY_tilde_vals = range(0., stop = 45., length = PY_tilde_gridpoints)
# scatter!(PY_tilde_vals,zeros(length(PY_tilde_vals)))
# histogram!(Y_SS, normalize=:true, bins = PY_tilde_vals, alpha = 0.5)

function policyfn(mcs_z, mc_ε, r, β, γ, PY_tilde_vals, z_gridpoints, ε_gridpoints, a_gridpoints, a_max, a_min = 0.)
    println("Policy function calculation...")
    A_vals = exp10.( range(log10(a_min+1), stop = log10(a_max+1), length =
     a_gridpoints) ) .- 1

    # We have 3 state variables: age (t), wealth (a),
    # and either permanent component of income (z)
    # or average lifetime earnings y_tilde
    # Matrices of form a x z/y_tilde
    A = vcat(fill(zeros(Float64, a_gridpoints, z_gridpoints, ε_gridpoints), T_ret+1),
     fill(zeros(Float64, a_gridpoints, PY_tilde_gridpoints), (T+2)-(T_ret+1)) )

    # Last period and period after last (for A[t+2] when t = T-1)
    A[T+2] = zeros(Float64, a_gridpoints, PY_tilde_gridpoints)
    A[T+1] = zeros(Float64, a_gridpoints, PY_tilde_gridpoints)
    # no need to shift grid because all zeros

    println("Retirement")
    A_grid_ret = repeat(A_vals, 1, PY_tilde_gridpoints)
    PY_tilde_grid = repeat(PY_tilde_vals', a_gridpoints, 1)

    for t in reverse(T_ret+1:T-1)
        println(t)
        C_t1 = (1+r).*A_grid_ret + PY_tilde_grid - A[t+2]
        C_t = ( β*ξ[t+1]/ξ[t]*(1+r) )^(-1/γ) .* C_t1
        A_t = 1/(1+r) .* (C_t - PY_tilde_grid + A_grid_ret)
        A[t+1] = invert_rule( A_t, A_vals, extrapolation_bc = a_min )
        # extrapolation_bc affects borrowing constraint
    end

    println("Year before retirement")
    for t = T_ret
        C_t1 = (1+r).*A_grid_ret + PY_tilde_grid - A[t+2]
        C_t = ( β*ξ[t+1]/ξ[t]*(1+r) )^(-1/γ) .* C_t1
        # interpolate for shifting onto 3d grid
        C_li = LinearInterpolation((A_vals, PY_tilde_vals), C_t,
         extrapolation_bc = (Flat(), Line()) )
        # Be careful with this extrapolation
        A_t = zeros(Float64, a_gridpoints, z_gridpoints, ε_gridpoints)
        for (ε_ind, ε) in enumerate(mc_ε.state_values), (z_ind,
         z) in enumerate(mcs_z[t].state_values), (a_t1_ind, a_t1) in enumerate(A_vals)

            A_t[a_t1_ind, z_ind, ε_ind] = 1/(1+r) * ( C_li(a_t1,
             exp(κ[t] + z + ε)) - exp(κ[t] + z + ε) + a_t1 )
        end
        A[t+1] = invert_rule_3d(A_t, A_vals, extrapolation_bc = a_min)
        # extrapolation_bc affects borrowing constraint
    end

    println("Working years")
    for t in reverse(1:T_ret-1)
        println(t)
        A_t2 = LinearInterpolation((A_vals, mcs_z[t+1].state_values,
         mc_ε.state_values), A[t+2])
        # interpolate for shifting A[t+2] to z[t] grid vals
        A_t = zeros(a_gridpoints,z_gridpoints,ε_gridpoints)
        # i = 0
        for (ε_ind, ε) in enumerate(mc_ε.state_values), (z_ind,
         z) in enumerate(mcs_z[t].state_values), (a_t1_ind, a_t1) in enumerate(A_vals)
            # @show i += 1
            Ec = 0.
            for (ε_t1_ind, ε_t1) in enumerate(mc_ε.state_values), (z_t1_ind,
             z_t1) in enumerate(mcs_z[t].state_values)

                Ec += mcs_z[t].p[z_ind, z_t1_ind] * mc_ε.p[1,ε_t1_ind] *
                 ( (1+r).*a_t1 + exp(κ[t+1] + z_t1 + ε_t1) - A_t2(a_t1, z_t1,
                 ε_t1) )^(-γ)
            end
            A_t[a_t1_ind, z_ind, ε_ind] = 1/(1+r) * ( ( β*(1+r)*Ec )^(-1/γ) -
             exp(κ[t] + z + ε) + a_t1 )
        end
        A[t+1] = invert_rule_3d(A_t, A_vals, extrapolation_bc = a_min)
        # extrapolation_bc affects borrowing constraint
    end

    return A
end

function simulate_economy(A, z_l, ε_l, κ, mcs_z, mc_ε, PY_tilde_vals, a_max, a_min, a_gridpoints, r)
    A_vals = exp10.( range(log10(a_min+1), stop = log10(a_max+1),
     length = a_gridpoints) ) .- 1
    Ai = zeros(Float64, N, T+2)
    Ci = similar(Ai)
    # Optional: Do empirical one too
    # "Precisely, we target the empirical distribution of financial wealth-earnings
    # ratios in the population of households aged 20-30 in the SCF. We assume
    # that the initial draw of earnings is independent of the initial draw of this
    # ratio, since in the data the empirical correlation is 0.02."

    ## Initial wealth is zero => Ai[:,1]=0.
    for t in 1:T_ret
        li = LinearInterpolation((A_vals, mcs_z[t].state_values,
         mc_ε.state_values), A[t+1], extrapolation_bc = Flat())
         #extrapolate with boundary rules
        Ai[:,t+1] = li.(Ai[:,t], z_l[:,t], ε_l[:,t])
        Ci[:,t] = (1+r) .* Ai[:,t] + exp.(κ[t] .+ z_l[:,t] + ε_l[:,t]) - Ai[:,t+1]
    end
    for t in T_ret+1:T
        li = LinearInterpolation((A_vals, PY_tilde_vals),
         A[t+1], extrapolation_bc = Flat()) #extrapolate with boundary rules
        Ai[:,t+1] = li.(Ai[:,t], Y_SS)
        Ci[:,t] = (1+r) .* Ai[:,t] + Y_SS - Ai[:,t+1]
    end
    return (Ai, Ci)
end

# Main program
A = policyfn(mcs_z, mc_ε, r, β, γ, PY_tilde_vals, z_gridpoints, ε_gridpoints, a_gridpoints, a_max, a_min)
# A[69]
# plot()
# heatmap(A[67])
# A[1][:,:,1]
# heatmap(A[35][:,:,10])
using JLD2, FileIO
vec_A = vec(A)
size_A = size(A)
@save "A.jld2" vec_A size_A
# @load "A.jld2" vec_A size_A
# A1 = reshape(vec_A, size_A[1])
# @test dropdims(sol.V, dims = 3) == V


Random.seed!(1234)
(Y_l, z_l, ε_l) = Y(N, T_ret, σ_ε, σ_η, σ_z0, κ)
# histogram(vec(Y_l), bins = 100)
(Y_tilde_l, τ_s) = G(Y_l, τ_s_0 = 0.105, verbose = true)
# Had 0.04057 before
#Should use 0.031, which is initial guess from Gouveia Strauss 1994
Y_SS = P(Y_tilde_l)
Y_tot = hcat(Y_l, repeat(Y_SS, 1, T-T_ret) )
# i = 100
# plot(age,Y_tot[i,:], ylims=(0,2000))
(Ai, Ci) = simulate_economy(A, z_l, ε_l, κ, mcs_z, mc_ε, PY_tilde_vals, a_max, a_min, a_gridpoints, r)


### Plotting
# Life-cycle means
Ci_ave = mean(Ci, dims = 1)'
plot(age, Ci_ave[1:T], label = "Consumption", linestyle=:dash,
 xlabel="Age", ylabel="\$ (,000)", title="Life-cycle Means (ZBC)",
 grid=false, ylims=(-20,200), xlims=(25,95), xticks=25:5:95)

Yi_ave = mean(Y_tot, dims = 1)'
plot!(age, Yi_ave, linestyle=:dash, label = "Income")

Ai_ave = mean(Ai, dims = 1)'
plot!(age, Ai_ave[1:T], label = "Wealth", linestyle=:dash)
savefig("zbc_lifecycle_means.png")

plot(age[1:T_ret], exp.(κ), legend=:false, title="Average Income Profile",
 xlabel="Age", ylabel="\$ (,000)", ylims=(0,30), grid=false)
savefig("kappa.png")

#
# # "Initial should be"
# 54/168*0.5
# Yi_ave[1]
# # "Max should be"
# 127/168*0.5
# maximum(Yi_ave)
# # Pension should be
# 64/235*0.5
# Yi_ave[36]
# # All are quite close!!


# Life-cycle Inequality
Ci_varlog = var(log.(Ci), dims = 1)'
plot(age, Ci_varlog[1:T], label = "Consumption", linestyle=:dash,
 title="Life-cycle Inequality (ZBC)",
 ylims=(0.05,0.6), xlims=(25,95), yticks=0.05:0.05:0.55, grid=false)

Yi_varlog = var(log.(Y_tot), dims = 1)'
plot!(age, Yi_varlog[1:T], label = "Income", linestyle=:dash)

savefig("zbc_lifecycle_inequality.png")
