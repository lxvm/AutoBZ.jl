#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
as well as functions in Demos.jl
=#

using StaticArrays
using FastChebInterp

include("../src/AutoBZ.jl")

include("Demos.jl")
include("LagrangeInterpolation.jl")

# import Fourier coefficients of Wannier Hamiltonian
coeffs = Demos.loadW90Hamiltonian("epsilon_mn.h5")
# define the periods of the axes of the Brillouin zone for example material
periods = fill(round(2π/3.858560, digits=6), SVector{3,Float64})
# construct the Hamiltonian datatype
H = AutoBZ.Applications.FourierSeries(coeffs, periods)

# import self energies from an equispaced grid
sigma_data = Demos.import_self_energy("srvo_sigma_ftps_T0.h5")

# construct a Chebyshev interpolant
order = 1000 # about one-third of data points
sigma_cheb_interp = chebregression(sigma_data.ω, sigma_data.Σ, (order,))
# reduce the domain to mitigate Runge's phenomenon
len = only(sigma_cheb_interp.ub) - only(sigma_cheb_interp.lb)
lb = only(sigma_cheb_interp.lb) + 0.05len
ub = only(sigma_cheb_interp.ub) - 0.05len

# construct a Barycentric Lagrange interpolant
degree = 8
sigma_bary_interp = LagrangeInterpolation.LocalEquiBaryInterpolant(sigma_data.ω, sigma_data.Σ, degree)

# construct the self energy datatype
Σ = AutoBZ.Applications.ScalarEnergy(sigma_cheb_interp, lb, ub)

# define problem parameters
μ = 12.3958 # eV
Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
# T = 0.0 # K # this will break my window functions
T = 50.0 # K # guess of the effective temperature

# define constants
kB = 8.617333262e-5 # eV/K

# derived parameters
β = inv(kB*T)

# set error tolerances
atol = 1e-3
rtol = 0.0

# run script
results = Demos.OCscript_parallel("OC_results_ftps.h5", H, Σ, β, Ωs, μ, atol, rtol)
#= timings with order 1000 Chebyshev interpolant
Omega time
0.0 finished in 1891.8583600521088
0.0031622776601683794 finished in 1774.606281042099
0.0037275937203149418 finished in 1731.514804840088
0.004393970560760791 finished in 1706.7738990783691
0.005179474679231213 finished in 1725.965322971344
0.006105402296585327 finished in 1686.2967910766602
0.0071968567300115215 finished in 1676.939858198166
0.008483428982440717 finished in 1675.8258249759674
0.01 finished in 1627.036689043045
0.011787686347935873 finished in 1564.3018748760223
0.013894954943731374 finished in 1515.2122359275818
0.016378937069540637 finished in 1378.5542709827423
0.019306977288832506 finished in 1497.1238269805908
0.022758459260747887 finished in 1168.6416590213776
0.02682695795279726 finished in 1036.325215101242
0.03162277660168379 finished in 874.3924720287323
0.037275937203149395 finished in 816.725564956665
0.043939705607607904 finished in 754.6311550140381
0.0517947467923121 finished in 677.3088920116425
0.0610540229658533 finished in 750.7328979969025
0.07196856730011521 finished in 662.8946158885956
0.08483428982440722 finished in 692.1851789951324
0.1 finished in 599.5399580001831
0.11787686347935872 finished in 492.77669405937195
0.13894954943731377 finished in 303.9363009929657
=#
# @profview Demos.test_OCscript(H, Σ, β, Ωs, μ, atol, rtol, 0.23, 0.21, 0.11)