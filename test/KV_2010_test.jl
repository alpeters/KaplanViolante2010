# Test routines for KV_2010.jl

## Mortality probability:
# Visual check
scatter(t,ξ)

## Check κ
plot([0:39],κ./maximum(κ), ylims=(0,1))

## Net labour income function Y()
function linreg(x,y)
    x = [x ones(length(y))]
    x'*x \ x'*y
end

Random.seed!(1234)
plot(labor_income(1,T_ret, 0., 0., 0., κ; y_0 = 0.0)')
any(labor_income(1,T_ret, 0., 0., 0., κ; y_0 = 0.0) .== 1.) != false

Random.seed!(1234)
testy = labor_income(10, T_ret, 0., σ_η, 0., κ; y_0 = 0.0)'
plot(testy)
i = 7
linreg(testy[i,1:end-1],testy[i,2:end])[2]

Random.seed!(1234)
testy = labor_income(10,T_ret, σ_ε, 0., 0., κ; y_0 = 0.0)'
plot!(testy)
i = 6
linreg(testy[i,1:end-1],testy[i,2:end])[2]

## Gross earnings Y_tilde_fn()
Random.seed!(1234)
N = 1000
Y_l = Y(N, T_ret, σ_ε, σ_η, σ_z0, κ)
i = rand(1:N)
plot(Y_l[i,:])
τ_s = 1.45
Y_tilde = Y_tilde_fn.(Y_l,τ_s)
plot!(Y_tilde[i,:])
scatter(Y_tilde[i,:], 1. .- Y_l[i,:] ./ Y_tilde[i,:], ylabel="tax rate", xlabel="gross income", legend=false)
scatter(Y_tilde, 1. .- Y_l ./ Y_tilde, ylabel="tax rate", xlabel="gross income", legend=false)

## Calibration of τ, function G()
Random.seed!(1234)
N = 100
Y_l = Y(N, T_ret, σ_ε, σ_η, σ_z0, κ)
sol = G(Y_l, τ_s_0 = 0.1, Kp = 10, Kd = 1., tol = 1E-3, max_iter = 100, nonconvergence_message = true)
sol.err
sol.iters
sol.τ_s

## Test P(Y_tilde)
## using Y_l instead of Y_tilde for simplicity
Random.seed!(1234)
N = 10000
Y_l = Y(N, T_ret, σ_ε, σ_η, σ_z0, κ)

## Bendpoints
Y_ave = mean(Y_l,dims=2) #lifetime average income
y_tilde_l_ave = mean(Y_ave)
bp1 = 0.18*y_tilde_l_ave
bp2 = 1.1*y_tilde_l_ave
Y_l_SS = P(Y_l)
scatter(Y_ave,Y_l_SS)
vline!([bp1 bp2])
xlims!(0.,1.)
ylims!(0.,0.4)

## Test average income gets 45% of earnings
y_tilde_l_ave = mean(sum(Y_l,dims=2)) #average lifetime net labour earnings
ind_mean = findmin(abs.( sum(Y_l,dims=2) .- y_tilde_l_ave ))
mean_ind = ind_mean[2][1]
sum(Y_l[mean_ind,:])
mean(Y_l[mean_ind,:])
Y_l_SS[mean_ind] - 0.45*mean(Y_l[mean_ind,:]) < 1E-4
plot(Y_l[mean_ind,:])
Y = hcat(Y_l, repeat(Y_l_SS, 1, T-(T_ret-1)))
plot(Y[mean_ind,:], ylims=(0,:))

## Test income generation
Random.seed!(1234)
Y_l = Y(N, T_ret, σ_ε, σ_η, σ_z0, κ)
(Y_tilde, τ_s) = G(Y_l, τ_s_0 = 1.45)
Y_tilde_SS = P(Y_tilde)
Y_SS = Y_tilde_SS - τ.(0.85.*Y_tilde_SS, τ_s)
Y = hcat(Y_l, repeat(Y_SS, 1, T-(T_ret-1)) )
for num = 1:10
    num == 1 ? plot() : nothing
    i = rand(1:N)
    display(plot!(Y[i,:], ylims=(0,8), label=i, title="Net income"))
end

# Discretization
# Test: Simulate the markov chain process and compare to simulated income process
Y_l = Y(N, T_ret, σ_ε, σ_η, σ_z0, κ)
scatter(1:T_ret-1, var(Y_l', dims = 2), label="Y_l", legend=:bottomright)
# Does this match graphs from literature??

# Generate z0's
z_0 = σ_z0 .* randn(N)
var(exp.(z_0))
# Find index of Markov chain corresponding to each z_0
ind_0 = [findmin( abs.(mcs_per[1].state_values .- z_0) )[2] for z_0 in z_0]
z = zeros(length(mcs_per)+1,N)
[z[:,i] = simulate_ns(mcs_per, ind_0[i]) for i in 1:N]
var(exp.(z[1,:]))
z = z[2:end,:]
ε = zeros(T_ret+1,N)
[ε[:,i] = simulate(mc_tran, T_ret+1, init = round(Int,19/2)) for i in 1:N]
ε = ε[2:end,:]
Κ = repeat(κ, 1, N)
Y_mc = exp.(Κ + z + ε)
scatter!(1:T_ret, var(Y_mc, dims = 2), label="Y_mc")
(var(Y_l', dims=2) - var(Y_mc, dims=2)[1:end-1]) ./ var(Y_l', dims=2)*100 |> plot
# Errors are small and don't change systematically over time

scatter(1:T_ret-1, mean(Y_l', dims = 2), label="Y_l", legend=:bottomright)
scatter!(1:T_ret, mean(Y_mc, dims = 2), label="Y_mc")
# Average are very close

# Check initial values are working
scatter(z_0,z_0 - z[1,:])
vline!(mcs_per[1].state_values)












# Boneyard

# Begin from last period
for t in reverse(T_ret:T-1) #reverse(t)
    if t < T_ret
        Ec = zeros(Float64,a_gridpoints,z_gridpoints)
        for (a_t1_ind, a_t1) in enumerate(A_vals), (z_ind, z) in enumerate(mcs_z[t].state_values)
            A_t2 = extrapolate(interpolate(A[t+2], BSpline(Linear())), 0. )(a,z)
            for (ε_ind, ε) in enumerate(mc_ε.state_values), (z_t1_ind, z_t1) in enumerate(mcs_z[t].state_values)
                Ec[a_ind, z_ind] += mcs_z[t].p[z_ind, z_t1_ind] * mc_ε.p[1,ε_ind] * ( (1+r).*a_t1 + exp(κ[t+1] + z_t1 + ε) - A_t2 )^(-γ)
            end
        end
        C[t] = ( β*(1+r) .* Ec).^(-1/γ)
        At = 1/(1+r) .* (C[t] - repeat(PY_tilde_vals', a_gridpoints, 1) + A_grid_l) # ******how to account for all possible combos of z,ε??
        A[t+1] = grid_shift(invert_rule(At, A_vals), A_vals, A_vals, Z_vals_old, Z_vals_new) # ********

    elseif t < (T-1) && t >= T_ret # Be careful with crossover year!
        C_t1 = (1+r).*A_grid_ret + PY_tilde_grid - A[t+2]
        C_t = ( β*ξ[t+1]/ξ[t]*(1+r) )^(-1/γ) .* C_t1
        At = 1/(1+r) .* (C_t - PY_tilde_grid + A_grid_ret)
        A[t+1] = invert_rule(At, A_vals)

    elseif t == T-1
        A[t+1] = zeros(Float64, a_gridpoints, PY_tilde_gridpoints)
        # no need to shift because all zeros
    end
end
