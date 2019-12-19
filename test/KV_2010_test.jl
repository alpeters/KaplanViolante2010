# Test routines for KV_2010.jl

## Mortality probability:
# Visual check
scatter(t,ξ)

## Check κf
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

# Check 3d interpolation
A = zeros(3,4,2)
A[:,:,1] = [1 2 3 4; 5 6 7 8; 9 10 11 12]
A[:,:,2] = 2*[1 2 3 4; 5 6 7 8; 9 10 11 12]
li = interpolate(A, BSpline(Linear()))
li(1,2.5,1)
mean(A,dims=3)

A = [1 2 3 4; 5 6 7 8]
fill(A, 1,1,2)

A
function testfn()
    A = Float64[]
    for a in 1:3, b in 2:5, c in 1:2
        push!(A, a + b + c)
    end
    return A
end

testfn()

A = [a + b + c for a in 1:3, b in 2:5, c in 1:2 ]
A = [println(a) for a in 1:3, b in 2:5, c in 1:2 ]
A = [(a,b,c) for a in 1:3, b in 4:5, c in 6:7]

# Ec = zeros(Float64,a_gridpoints,z_gridpoints)   # Ec =  E_t[c_{ŧ+1}^{-γ}]
# for (a_t1_ind, a_t1) in enumerate(A_vals), (z_ind, z) in enumerate(mcs_z[t].state_values), ε in enumerate(mc_ε.state_values)
# C_t = ( β*(1+r) .* fill(Ec,1,1,ε_gridpoints).^(-1/γ)
# repeat(mcs_z[1].state_values, 1, ε_gridpoints)
# At = 1/(1+r) .* (C[t] - exp.(κ[t] + z + ε) + A_grid_l) # ******how to account for all possible combos of z,ε??

a_min = 0. # ZBC
a_gridpoints = 10
a_max = 10.
A_vals = exp10.( range(log10(a_min+1), stop = log10(a_max+1), length = a_gridpoints) ) .- 1



for t in reverse(1:T_ret-2) #reverse(t)
    At = zeros(a_gridpoints,z_gridpoints,ε_gridpoints)
    for a_t1 in A_vals, (z_ind, z) in enumerate(mcs_z[t].state_values), ε in mc_ε.state_values
        Ec = 0.
        for (ε_t1_ind, ε_t1) in enumerate(mc_ε.state_values), (z_t1_ind, z_t1) in enumerate(mcs_z[t].state_values)
            A_t2 = extrapolate(interpolate(A[t+2], BSpline(Linear())), 0. )(a_t1, z_t1, ε_t1) # NEED INDICES, NOT VALUES
            Ec += mcs_z[t].p[z_ind, z_t1_ind] * mc_ε.p[1,ε_t1_ind] * ( (1+r).*a_t1 + exp(κ[t+1] + z_t1 + ε_t1) - A_t2 )^(-γ)
        end
        At = 1/(1+r) .* ( ( β*(1+r)*Ec )^(-1/γ) - exp(κ[t] + z + ε) + a_t1 )
    end
    A[t+1] = grid_shift(invert_rule(At, A_vals), A_vals, A_vals, Z_vals_old, Z_vals_new) # ********
end

using ProgressMeter
function testfn()
    for t in 1 #reverse(t)
        i=0
        At = zeros(a_gridpoints,z_gridpoints,ε_gridpoints)
        li = extrapolate(interpolate(A[t+2], BSpline(Linear())), 0. )
        for (ε_ind, ε) in enumerate(mc_ε.state_values), (z_ind, z) in enumerate(mcs_z[t].state_values), (a_t1_ind, a_t1) in enumerate(A_vals)
            Ec = 0.
            for (ε_t1_ind, ε_t1) in enumerate(mc_ε.state_values), (z_t1_ind, z_t1) in enumerate(mcs_z[t].state_values)
                A_t2 = li(a_t1_ind, z_t1_ind, ε_t1_ind) # CHECK THESE INDICES ARE RIGHT
                Ec += mcs_z[t].p[z_ind, z_t1_ind] * mc_ε.p[1,ε_t1_ind] * ( (1+r).*a_t1 + exp(κ[t+1] + z_t1 + ε_t1) - A_t2 )^(-γ)
            end
            At[a_t1_ind,z_ind,ε_ind] = 1/(1+r) .* ( ( β*(1+r)*Ec )^(-1/γ) - exp(κ[t] + z + ε) + a_t1 )
            @show i+= 1
        end
        return At

        # At = Float32[]  #zeros(a_gridpoints,z_gridpoints,ε_gridpoints)
        # i = 0
        # li = extrapolate(interpolate(A[t+2], BSpline(Linear())), 0. )
        # for ε in mc_ε.state_values, (z_ind, z) in enumerate(mcs_z[t].state_values), (a_t1_ind, a_t1) in enumerate(A_vals)
        #     @show i += 1
        #     Ec = 0.
        #     for (ε_t1_ind, ε_t1) in enumerate(mc_ε.state_values), (z_t1_ind, z_t1) in enumerate(mcs_z[t].state_values)
        #         A_t2 = li(a_t1_ind, z_t1_ind, ε_t1_ind) # CHECK THESE INDICES ARE RIGHT
        #         Ec += mcs_z[t].p[z_ind, z_t1_ind] * mc_ε.p[1,ε_t1_ind] * ( (1+r).*a_t1 + exp(κ[t+1] + z_t1 + ε_t1) - A_t2 )^(-γ)
        #     end
        #     push!(At, 1/(1+r) .* ( ( β*(1+r)*Ec )^(-1/γ) - exp(κ[t] + z + ε) + a_t1 ) )
        # end
        # return At
        # # A[t+1] = grid_shift(invert_rule(At, A_vals), A_vals, A_vals, Z_vals_old, Z_vals_new) # ********
    end
end

At = testfn()
A = reshape(At,a_gridpoints,z_gridpoints,ε_gridpoints)
heatmap(A)

function myfunc()
    @showprogress for i in 1:1E9, b in 1:15
        i += i
    end
end


myfunc()



function invert_rule(A,A_vals)
    A_interp = similar(A)
    for col in eachindex(A[1,:])
        # li = LinearInterpolation(A[:,col], A_vals, extrapolation_bc = (0.,Line()))
        # use extrapolate and do different for each side fixed value/line
        A_interp[:,col] = extrapolate(interpolate(A[:,col], BSpline(Linear())), 0.)(A_vals)
    end
    return A_interp
end


A = zeros(11,2)
A_vals = 0:0.1:1
A[:,1] = [(x-5)^3 + 100 for x in A_vals]
A[:,1] = [x^2 for x in A_vals]

plot!(1:25,1:25)
plot()
scatter!(A_vals,A[:,1])
A_inv = invert_rule(A, A[:,1])
scatter!(A[:,1],A_inv[:,1])

plot(x -> (x-5)^3 + 100, 0, 10, ylims=(0,200))

function interpolate_cols(X, Y, X_grid::Array; extrapolation_bc = Throw())
    #linear interpolation of a column of Y on that column of X, evaluated at point x
    Y_interp = similar(X_grid)
    for col in eachindex(X_grid[1,:])
        li = LinearInterpolation(X[:,col], Y[:,col],extrapolation_bc=extrapolation_bc)
        Y_interp[:,col] = li.(X_grid[:,col])
    end
    return Y_interp
end



A_prime_grid = repeat(A_vals, 1, 2)
A_prime = interpolate_cols(A,A_prime_grid,A_prime_grid, extrapolation_bc = Flat())


A_vals = 0:0.1:1
A = [log(x) for x in A_vals]
scatter(A_vals,A)
li = LinearInterpolation(A,A_vals, extrapolation_bc=Flat())
A_inv = li.(A_vals)
scatter!(A_vals,A_inv)

# Check invert_rule
using Interpolations, Plots
A_vals = 0.1:0.01:1.5
Y_vals = 0.1:0.1:0.3
A = [y + x^3 for x in A_vals, y in Y_vals]
minimum(A[:,1])
plot([0; 1],[0; 1], linestyle=:dash, legend=:bottomright)
plot!(A_vals,A[:,2])
# plot!(A_vals,A[:,2])
# plot!(A_vals,A[:,3])
A_inv = invert_rule(A, A_vals, extrapolation_bc = 0.)
plot!(A_vals,A_inv[:,2])
# plot!(A_vals,A_inv[:,2])
# plot!(A_vals,A_inv[:,3])
# ylims!(0,1)

function invert_rule(A, A_vals; extrapolation_bc = Throw())
    #linear interpolation of a column of Y on that column of X, evaluated at point x
    A_vals_interp = similar(A)
    for col in eachindex(A[1,:])
        li = LinearInterpolation(A[:,col], A_vals,extrapolation_bc=extrapolation_bc)
        A_vals_interp[:,col] = li(A_vals)
    end
    return A_vals_interp
end

A
A3d = zeros(size(A)[1],size(A)[2],3)
function myfunc(A)
    for i in 1:size(A3d)[3]
        A3d[:,:,i] = A
    end
    return A3d
end
A3d = myfunc(A)
A3d[:,:,1]==A3d[:,:,3]

function invert_rule_3d(A3d, A_vals; extrapolation_bc = Throw())
    wrapper(A) = invert_rule(A,A_vals,extrapolation_bc = extrapolation_bc)
    A3d_inv = mapslices(wrapper, A3d, dims=1)
end

A3d_inv = invert_rule_3d(A3d, A_vals, extrapolation_bc = Flat())
plot([0; 1],[0; 1], linestyle=:dash, legend=:bottomright)
plot!(A_vals,A3d[:,2,3])
plot!(A_vals,A3d_inv[:,2,3])





# Try to reconstruct construction
A_inv2 = extrapolate(scale(interpolate(A, BSpline(Linear())), A), Flat())(A_vals)
scatter!(A_vals,A_inv2)
A_inv2 = extrapolate()
scale(interpolate(A_vals, BSpline(Linear())), A)
#fail

# Multidimensional LinearInterpolation test (NOT INVERTED)
A_vals = 0:0.1:1
Y_vals = 1:3
A = [y + x^2 for x in A_vals, y in Y_vals]
li = LinearInterpolation((A_vals,Y_vals),A , extrapolation_bc=Flat())
li(0,2)
plotly()
plot(A,st=:surface)
plot!([2 2.5],[1 1],[li(0,2) li(0,2.5)],st=:scatter3d)

function mytestfunc(A_vals, Y_vals)
    for i = 1
        li1 = LinearInterpolation(A_vals,Y_vals)
    end
    for i = 2:3
        @show s = li1(2.)
        li2 = LinearInterpolation(A_vals.+i,Y_vals)
    end
end

mytestfunc([1;2; 3],[1; 4; 9])
# interpolation object doesn't get passed between for loops,
# have to define at the beginning of each



        #             (A_vals, mcs_z[t+1].state_values, mc_ε.state_values)
        # grid_shift(A[t+1], Z_vals_old, Z_vals_new, dims = 2)
        # li = LinearInterpolation((A_vals,mcs_z[t].state_values,),A[t+2] , extrapolation_bc=Flat())


        # function grid_shift(A, A_vals_old, A_vals_new, Y_vals_old, Y_vals_new)
        #     A_grid = zeros( eltype(A[1]), length(A_vals_new), length(Y_vals_new) )
        #
        #     # Find relative coordinates of new basis
        #     li = LinearInterpolation(Y_vals_old, 1:length(Y_vals_old), extrapolation_bc=Line())
        #     Y_ind = li(Y_vals_new)
        #     li = LinearInterpolation(A_vals_old, 1:length(A_vals_old), extrapolation_bc=Line())
        #     A_ind = li(A_vals_new)
        #
        #     # Bi-linear interpolation on new coordinates
        #     li = extrapolate(interpolate(A, BSpline(Linear()) ), Line())  #extrapolate with boundary rules
        #     [ A_grid[i,j] = li(A_ind, Y_ind) for  (i, A_ind) in enumerate(A_ind),
        #                                                 (j, Y_ind) in enumerate(Y_ind)]
        #     return A_grid
        # end



## Income process troubleshoot/calibration

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
    return (Y, z, ε)
end

(Y_l, z_l, ε_l) = Y(10000000, 2, σ_ε, σ_η, σ_z0, κ)
(Y_l, z_l, ε_l) = Y(N, T_ret, σ_ε, 0., 0., κ)
Y_tot = hcat(Y_l, repeat(Y_SS, 1, (T-1)-(T_ret-1)) )
mean(Y_l, dims = 1)
exp(κ[1])
mean(z_l, dims = 1)
exp(κ[1] + mean(z_l[:,1]) + mean(ε_l[:,1]))


li = LinearInterpolation(zeros(10), 1:10)
