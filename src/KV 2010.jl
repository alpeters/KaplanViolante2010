# Replicate Kaplan Violante 2010
# Allen Peters
# November 7, 2019
# command free QuantEcon to revert to proper version of QuantEcon
using Plots, LinearAlgebra, Statistics, Random, Roots, Interpolations
using QuantEcon

########
# From development version of QuantEcon
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
            ind_0 = ind[t] #findmin( abs.(mcs[t].state_values .- X[t]) )[2]
        end
    end
    ind = [init; ind]
    X = [mcs[1].state_values[init]; X]
end
############

# Parameters
## Demographics
N = 50000
T_ret = 35
T = 70
t = 1:T-1   # Agent is dead in period T, retired in T_ret
age = t .+ 25

## Uncondtional probability of surviving to age t
# Data from NCHS 1992 (Chung 1994)
age_data = [60:5:85; 95.]
S_data = [85.89; 80.08; 72.29; 61.52; 48.13; 32.33; 0]./100
S_interp = LinearInterpolation(age_data, S_data)
ξ_ret = map(S_interp,collect(60:95))
ξ = [ones(Float64,T_ret-1); ξ_ret]

## Preferences
γ = 2.

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
# Assume KV's description is about levels. Then my data peaks slightly later and doesn't decline as much as KV describes.
κ = log.(map(y_interp,collect(25:59)))

# Net labour income
function Y(N, T_ret, σ_ε, σ_η, σ_z0, κ)
# Takes standard deviations as arguments (not varicances)!
    Y = zeros(Float64, N, T_ret-1)
    z = similar(Y)
    ε = similar(Y)
    z[:,1] = σ_z0 .* randn(N) + σ_η .* randn(N)

    for t in 1:(T_ret-1)
        ε[:,t] = σ_ε .* randn(N)
        Y[:,t] = exp.( κ[t] .+ z[:,t] + ε[:,t] )
        t < (T_ret-1) ? z[:,t+1] = z[:,t] + σ_η .* randn(N) : nothing
    end
    return (Y,z, ε)
end

# Gross earnings Y_tilde
τ_b = 0.258
τ_ρ = 0.768
τ(Y_tilde,τ_s) = τ_b*(Y_tilde-(Y_tilde^(-τ_ρ) + τ_s)^(-1/τ_ρ))
Y_tilde_fn(Y, τ_s) = find_zero(Y_tilde -> Y_tilde - τ(Y_tilde,τ_s) - Y, Y)
# Select an algorithm, with a derivative?
# Linear interpolation may be much faster

# Iterate to calibrate τ_s (Returns gross labour income)
function G(Y_l; τ_s_0 = 0.25, Kp = .8, Kd = .1, tol = 1E-3, max_iter = 1000, nonconvergence_message = true, verbose = false)
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


# Social security benefits (After tax retirement income)
function P(Y_tilde_l)
    Y_tilde_l_iave = mean(Y_tilde_l, dims = 2) # Denoted \tilde{Y}_i^SS in the paper just to be confusing. This each individuals average lifetime earnings
    y_tilde_l_ave = mean(Y_tilde_l_iave) # Average lifetime earnings across individuals
    SS_schedule(y) = y < 0.18*y_tilde_l_ave ? 0.9*y : (y > 1.1*y_tilde_l_ave ? 0.9*0.18*y_tilde_l_ave + 0.32*(1.1-0.18)*y_tilde_l_ave + 0.15*(y-1.1*y_tilde_l_ave) : 0.9*0.18*y_tilde_l_ave + 0.32*(y-0.18*y_tilde_l_ave) )
    Y_tilde_SS = 0.45*y_tilde_l_ave/SS_schedule(y_tilde_l_ave).*replace(SS_schedule, Y_tilde_l_iave)
    Y_SS = Y_tilde_SS - τ.(0.85.*Y_tilde_SS, τ_s)
end

# Directly from KV2010
σ_η = sqrt(0.01)
σ_z0 = sqrt(0.15)
σ_ε = sqrt(0.05)

# Discretize permanent component of income
# 39 equally spaced points on an age-varying grid chosen to match the age-specific unconditional variances
z_gridpoints = 39
mcs_z = rouwenhorst_ns(z_gridpoints, T_ret, 1., σ_η, σ_z0)
# Discretize transitory component of income
# transitory component is approximated with 19 equally spaced points
ε_gridpoints = 19
mc_ε = rouwenhorst(ε_gridpoints, 0., σ_ε)

function invert_rule(A,A_vals)
    A_interp = similar(A)
    for col in eachindex(A[1,:])
        # li = LinearInterpolation(A[:,col], A_vals, extrapolation_bc = (0.,Line()))
        # use extrapolate and do different for each side fixed value/line
        A_interp[:,col] = extrapolate(interpolate(A[:,col], BSpline(Linear())), 0.)(A_vals)
    end
    return A_interp
end

function grid_shift(A, A_vals_old, A_vals_new, Y_vals_old, Y_vals_new)
    A_grid = zeros( eltype(A[1]), length(A_vals_new), length(Y_vals_new) )

    # Find relative coordinates of new basis
    li = LinearInterpolation(Y_vals_old, 1:length(Y_vals_old), extrapolation_bc=Line())
    Y_ind = li(Y_vals_new)
    li = LinearInterpolation(A_vals_old, 1:length(A_vals_old), extrapolation_bc=Line())
    A_ind = li(A_vals_new)

    # Bi-linear interpolation on new coordinates
    li = extrapolate(interpolate(A, BSpline(Linear()) ), Line())  #extrapolate with boundary rules
    [ A_grid[i,j] = li(A_ind, Y_ind) for  (i, A_ind) in enumerate(A_ind), (j, Y_ind) in enumerate(Y_ind)]
    return A_grid
end

function policyfn(mcs_z, mc_ε)
    # Policy function via endogenous grid method, as per footnote 23
    # 100 exponentially spaced grid points for assets.
    ## Borrowing limit
    # how to calc? A_min =   # NBC
    a_min = 0. # ZBC
    a_gridpoints = 100
    a_max = 10.
    A_vals = exp10.( range(log10(a_min+1), stop = log10(a_max+1), length = a_gridpoints) ) .- 1

    # The grid for lifetime average earnings has 19 points.
    # I will build a grid on P(y_tilde) rather than y_tilde because
    # don't have to worry about matching mean for accuracy
    PY_tilde_gridpoints = 19
    # Based on simulating the income process:
    # histogram(Y_SS, normalize=:true, bins = 100)
    # minimum(Y_SS)
    # maximum(Y_SS)
    PY_tilde_vals = range(8., stop = 170., length = PY_tilde_gridpoints)
    # scatter!(PY_tilde_vals,zeros(length(PY_tilde_vals)))
    # histogram!(Y_SS, normalize=:true, bins = PY_tilde_vals, alpha = 0.5)

    # We have 3 state variables: age (t), wealth (a),
    # and either permanent component of income (z)
    # or average lifetime earnings y_tilde
    # Matrices of form a x z/y_tilde
    A = vcat(fill(zeros(Float64, a_gridpoints, z_gridpoints, ε_gridpoints), T_ret-1),
     fill(zeros(Float64, a_gridpoints, PY_tilde_gridpoints), (T)-(T_ret-1)) )

    # Last period
    A[T] = zeros(Float64, a_gridpoints, PY_tilde_gridpoints)
    # no need to shift grid because all zeros

    # Retirement
    A_grid_ret = repeat(A_vals, 1, PY_tilde_gridpoints)
    PY_tilde_grid = repeat(PY_tilde_vals', a_gridpoints, 1)

    for t in reverse(T-2:T_ret)
        C_t1 = (1+r).*A_grid_ret + PY_tilde_grid - A[t+2]
        C_t = ( β*ξ[t+1]/ξ[t]*(1+r) )^(-1/γ) .* C_t1
        At = 1/(1+r) .* (C_t - PY_tilde_grid + A_grid_ret)
        A[t+1] = invert_rule(At, A_vals)
    end

    # Year before retirement
    A_grid_l = repeat(A_vals, 1, z_gridpoints)

    for t = T_ret-1
        C_t1 = (1+r).*A_grid_ret + PY_tilde_grid - A[t+2]
        C_t = ( β*ξ[t+1]/ξ[t]*(1+r) )^(-1/γ) .* C_t1
        # replace with working BC: At = 1/(1+r) .* (C_t - PY_tilde_grid + A_grid_ret)
        A[t+1] = invert_rule(At, A_vals)
    end

    # Working years
    for t in reverse(1:T_ret-2)
        At = zeros(a_gridpoints,z_gridpoints,ε_gridpoints)
        li = extrapolate(interpolate(A[t+2], BSpline(Linear())), 0. )
        for (ε_ind, ε) in enumerate(mc_ε.state_values), (z_ind, z) in enumerate(mcs_z[t].state_values), (a_t1_ind, a_t1) in enumerate(A_vals)
            Ec = 0.
            for (ε_t1_ind, ε_t1) in enumerate(mc_ε.state_values), (z_t1_ind, z_t1) in enumerate(mcs_z[t].state_values)
                A_t2 = li(a_t1_ind, z_t1_ind, ε_t1_ind) # CHECK THESE INDICES ARE RIGHT
                Ec += mcs_z[t].p[z_ind, z_t1_ind] * mc_ε.p[1,ε_t1_ind] * ( (1+r).*a_t1 + exp(κ[t+1] + z_t1 + ε_t1) - A_t2 )^(-γ)
            end
            At[a_t1_ind, z_ind, ε_ind] = 1/(1+r) .* ( ( β*(1+r)*Ec )^(-1/γ) - exp(κ[t] + z + ε) + a_t1 )
        end
        return At
        # A[t+1] = grid_shift(invert_rule(At, A_vals), A_vals, A_vals, Z_vals_old, Z_vals_new) # ********
        # update invert_rule and grid_shift to deal with 3d matrices
    end

    return A
end

# A = policyfn(mcs_z, mc_ε)
# A[69]
# heatmap(A[T_ret+2])
# plot()
# plot!(A_vals, A[T_ret+5][:,9])
# A_vals |> length
# A[1] |> size

function simulate_economy(A,Y_state)
    ## Initial wealth is zero => Ai[:,1]=0.
    Ai = zeros(Float64, N, T)
    # Optional: Do empirical one too
    # "Precisely, we target the empirical distribution of financial wealth-earnings ratios in the population of house
    # holds aged 20-30 in the SCF. We assume that the initial draw of earnings is independent of the initial draw of this
    # ratio, since in the data the empirical correlation is 0.02."
    for t in T_ret+1:T-1
        li = extrapolate(interpolate(A[t+1], BSpline(Linear()) ), Line())
        Ai[:,t+1] = li.(Ai[:,t],Y_state[:,t])  #extrapolate with boundary rules
        # ************These need to be indices, not values!
    end
    return Ai
end

# Main program
Random.seed!(1234)
(Y_l, z_l, ε_l) = Y(N, T_ret, σ_ε, σ_η, σ_z0, κ)
(Y_tilde_l, τ_s) = G(Y_l, τ_s_0 = 0.04057, verbose = true) #0.031 - Initial guess from Gouveia Strauss 1994
Y_SS = P(Y_tilde_l)
Y_tot = hcat(Y_l, repeat(Y_SS, 1, T-(T_ret-1)) )
Y_state = hcat(z_l, repeat(Y_SS, 1, T-(T_ret-1)) )
A = policyfn(mcs_z, mc_ε)

Ai = simulate_economy(A,Y_state)
Ai_ave = mean(Ai, dims = 1)

plot(age, Ai_ave[1:end-1], xlabel="Age", ylabel="\$ (00,000)",
 title="Life-cycle Means", label = "Zero BC", linestyle=:dash)

Ai_var = var(Ai, dims = 1)
plot(age, Ai_var[1:end-1], xlabel="Age", ylabel="Variance of logs",
 title="Life-cycle Inequality", label = "Zero BC", linestyle=:dash)

heatmap(A[69])
li = extrapolate(interpolate(A[69], BSpline(Linear()) ), Line())
li.(29.,17.5)  #extrapolate with boundary rules


# Todo
# 1. policy rule for working period with 3rd dimension
# 2. Check expectation for working years
# 3. update and check invert_rule and grid_shift
# 4. check if borrowing constraint is working
# 5. use indices in simulate_economy
# 6. Non-zero borrowing limit (not idiosyncratic)
# We allow for borrowing subject only to the restriction that with probability one,
# households who live up to age T do not die in debt (i.e., the "natural debt limit").
# The level of the natural debt limit depends on the discretization of the income
# process, through the level of the lowest possible income realization. In the
# benchmark economy, the natural borrowing limit decreases from approximately
# 5.8 times average annual earnings at age 25 to 2 5 times average earnings at age 50.
# 7. Survival probability in KV? test both ways
