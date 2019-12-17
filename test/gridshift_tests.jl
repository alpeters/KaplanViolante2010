# Interpolation
xs = 1:5
ys = 1:8
g = Float64[x^2 * sin(y) for x in xs, y in ys]

gitp_quad2d = interpolate(g, BSpline(Quadratic(Line(OnCell()))))
using Gadfly
display(plot(x=xs,y=ys,z=g,Geom.contour))
display(plot(x=1:.1:5, y=1:.1:8, z=[gitp_quad2d(x,y) for x in 1:.1:5, y in 1:.1:8], Geom.contour))

xs = [1; 2]
ys = [3; 9]
g = ys
myinterp = LinearInterpolation([1 3; 2 4], BSpline(Linear()))
myinterp(1.5,1.5)

function grid_shift(A, row_vals, )

myinterp = LinearInterpolation([1 3; 2 4], BSpline(Linear()))
myinterp(1.5,1.5)


function interpolate_cols(X, Y, X_grid::Array; extrapolation_bc = Throw())
    #linear interpolation of a column of Y on that column of X, evaluated at point x
    Y_interp = similar(X_grid)
    for col in eachindex(X_grid[1,:])
        li = LinearInterpolation(X[:,col], Y[:,col],extrapolation_bc=extrapolation_bc)
        Y_interp[:,col] = li.(X_grid[:,col])
    end
    return Y_interp
end

A = [1. 3.; 2. 4.; 3. 5.]
A_vals = [6.; 7.; 8.]
Y_vals = [3; 5]

function invert_rule(A,A_vals)
    A_interp = similar(A)
    for col in eachindex(A[1,:])
        li = LinearInterpolation(A[:,col], A_vals, extrapolation_bc = Line())
        # use extrapolate and do different for each side fixed value/line
        A_interp[:,col] = li.(A_vals)
    end
    return A_interp
end


A_inv = invert_rule(A, A_vals)
li = interpolate(invert_rule(A, A_vals), BSpline(Linear()))
li(1.5,1.5)

function grid_shift(A, A_vals_old, A_vals_new, Y_vals_old, Y_vals_new)
    A_grid = zeros( eltype(A_inv[1]), length(A[:,1]), length(Y_vals) )

    # Find relative coordinates of new basis
    li = LinearInterpolation(Y_vals_old, 1:length(Y_vals_old), extrapolation_bc=Line())
    Y_ind = li(Y_vals_new)
    li = LinearInterpolation(A_vals_old, 1:length(A_vals_old), extrapolation_bc=Line())
    A_ind = li(A_vals_new)

    # Bi-linear interpolation on new coordinates
    li = interpolate(A_inv, BSpline(Linear()) )  #extrapolate with boundary rules
    [ A_grid[i,j] = li(A_ind, Y_ind) for  (i, A_ind) in enumerate(A_ind), (j, Y_ind) in enumerate(Y_ind)]
    return A_grid
end


A_inv = [1. 3.; 2. 4.; 2.5 5.5]
A_vals = [6.; 7.; 8.]
Y_vals = [3; 5]
Y_vals_new = [3.1; 4.9]
A_shift = grid_shift(A, A_vals, A_vals, Y_vals, Y_vals_new)
Plots.heatmap(A_inv)
Plots.heatmap(A_shift)
plot()
Plotly.plot(A_inv, st=:surface, camera=(-30,30))
Plotly.plot(A_shift, st=:surface, camera=(-30,30))


#############
plot(A_vals, Y_vals, A_inv, st=:scatter, camera=(45,45))
li = interpolate(A_inv, BSpline(Linear()))
plot(li, st=:surface, camera=(-30,30))
li(1.5,1.5)

A_vals_new = [6.5; 7.; 8.5]



Y_vals_new = [3; 5]
grid_shift(A_inv, A_vals, Y_vals, A_vals_new, Y_vals_new)

A_vals_old = [3.5;4.0;5.0;7.0]
scatter(A_vals_old)
scatter!(A_vals_new)
A_vals_new = [3.6;4.0;6;7.0]
li = LinearInterpolation(A_vals_old, 1:length(A_vals_old), extrapolation_bc=Line())
li(A_vals_new)



# Find indices for new grid bases
li_A = LinearInterpolation(A_vals_old, A_vals_new, extrapolation_bc=Line())
@show A_inds = li_A(A_vals_old) .- minimum(A_vals_old) .+ 1
li_Y = LinearInterpolation(Y_vals_old, Y_vals_new, extrapolation_bc=Line())
@show Y_inds = li_Y(Y_vals_old) .- minimum(Y_vals_old) .+ 1



A = [1. 3.; 2. 4.; 3. 5.]
A_vals = [6.; 7.; 8.]
Y_vals = [3; 5]
Y_vals_new = [4]
A_shift = grid_shift(A, A_vals, A_vals, Y_vals, Y_vals_new)
plot(repeat(A_vals,1,length(Y_vals)),repeat(Y_vals', length(A_vals),1), A, st=:surface, camera=(-30,30))
Plotly.plot(A_shift, st=:surface, camera=(-30,30))

A |> size
repeat(A_vals,1,length(Y_vals)) |> size
repeat(Y_vals', length(A_vals),1) |> size
using Plots
pyplot()

plot(A_vals, Y_vals, vec(A), st=:surface, camera=(-30,30))

vec(A)
A


A = [3. 4; 5 6; 7 7]
A_vals = [6.; 7.; 8.]
A_vals_new = [3.1;5;6.9]
Y_vals = [3; 5]
Y_vals_new = [2.5; 4.5]
li = interpolate([1;2;3], A_vals, BSpline(Linear()))
li = extrapolate(interpolate(A[:,1], A_vals, Gridded(Linear())), extrapolation_bc = (0.,Line()))
minimum(A[:,1])
maximum(A[:,1])
li(A_vals_new)
A_inv1 = invert_rule(A,A_vals)
A_inv2 = invert_rule(A,A_vals_new)
A_shift = grid_shift(A, A_vals, A_vals_new, Y_vals, Y_vals)
plotly()
plot(repeat(A_vals,length(Y_vals)),vec(repeat(Y_vals', length(A_vals),1 )),vec(A), st=:scatter, camera=(-30,30))
plot!(repeat(A_vals_new,length(Y_vals)),vec(repeat(Y_vals', length(A_vals_new),1 )),vec(A_shift), st=:scatter, camera=(-30,30))



plotly()
plot(repeat(A_vals,length(Y_vals)),vec(repeat(Y_vals', length(A_vals),1 )),vec(A_inv1), st=:scatter, camera=(-30,30))
plot!(repeat(A_vals_new,length(Y_vals)),vec(repeat(Y_vals', length(A_vals_new),1 )),vec(A_inv2), st=:scatter, camera=(-30,30))


plotly()
plot(repeat(A_vals,length(Y_vals)),vec(repeat(Y_vals', length(A_vals),1 )),vec(A), st=:scatter, camera=(-30,30))
plot!(repeat(A_vals,length(Y_vals)),vec(repeat(Y_vals', length(A_vals),1 )),vec(A_inv), st=:scatter, camera=(-30,30))

plot!(repeat(A_vals_new,length(Y_vals_new)),vec(repeat(Y_vals_new', length(A_vals_new),1 )),vec(A_shift), st=:scatter, camera=(-30,30))


using Plots, Interpolations
B = [x^2 for x in 1.:10]
scatter(1.:10,B, legend=:bottomright)
li = extrapolate(interpolate(B, BSpline(Linear())), 0.)
x2=0.:11.
B_interp = li(x2)
scatter(x2, B_interp)
B_interp = extrapolate(interpolate(B, BSpline(Linear())), 0.)(x2)
