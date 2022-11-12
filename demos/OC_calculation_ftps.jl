#=
In this script we compute OC at various Ω using frequency-dependent self energy
data that we interpolate with a high-order Chebyshev regression
=#

using FastChebInterp

using AutoBZ


# define the periods of the axes of the Brillouin zone for example material
period = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
HV = AutoBZ.Applications.load_hamiltonian_velocities("svo_hr.dat"; period=period)

# import self energies from an equispaced grid
sigma_data = AutoBZ.Jobs.import_self_energy("srvo_sigma_ftps_T0.h5")

# construct a Chebyshev interpolant
order = 1000 # about one-third of data points
sigma_cheb_interp = chebregression(sigma_data.ω, sigma_data.Σ, (order,))
# reduce the domain to mitigate Runge's phenomenon
len = only(sigma_cheb_interp.ub) - only(sigma_cheb_interp.lb)
lb = only(sigma_cheb_interp.lb) + 0.05len
ub = only(sigma_cheb_interp.ub) - 0.05len

# construct a Barycentric Lagrange interpolant
degree = 8
sigma_bary_interp = AutoBZ.EquiBaryInterp.LocalEquiBaryInterp(sigma_data.ω, sigma_data.Σ, degree)

# construct the self energy datatype
# Σ = AutoBZ.Applications.ScalarEnergy(sigma_cheb_interp, lb, ub)
Σ = AutoBZ.Applications.ScalarEnergy(sigma_bary_interp, minimum(sigma_data.ω), maximum(sigma_data.ω))

# define problem parameters
μ = 12.3958 # eV
Ωs = [0.0]
# Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
# T = 0.0 # K # this will break my window functions
T = 50.0 # K # guess of the effective temperature

# define constants
kB = 8.617333262e-5 # eV/K

# derived parameters
β = inv(kB*T)

# set error tolerances
atol = 1e-1
rtol = 0.0

# run script
results = AutoBZ.Jobs.OCscript_parallel("OC_results_ftps.h5", HV, Σ, β, Ωs, μ, atol, rtol)
