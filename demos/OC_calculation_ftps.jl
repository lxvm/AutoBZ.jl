#=
In this script we compute OC at various Ω using frequency-dependent self energy
data that we interpolate with a high-order Chebyshev regression
=#

using AutoBZ
using AutoBZ.Applications


# define the periods of the axes of the Brillouin zone for example material
b = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
HV = load_hamiltonian_velocities("svo_hr.dat"; period=b)

# import self energies from an equispaced grid
sigma_data = AutoBZ.Jobs.import_self_energy("srvo_sigma_ftps_T0.h5")

# construct a Barycentric Lagrange interpolant
degree = 8
sigma_bary_interp = AutoBZ.EquiBaryInterp.LocalEquiBaryInterp(sigma_data.ω, sigma_data.Σ, degree)

# construct the self energy datatype
# Σ = AutoBZ.Applications.ScalarEnergy(sigma_cheb_interp, lb, ub)
Σ = ScalarEnergy(sigma_bary_interp, minimum(sigma_data.ω), maximum(sigma_data.ω))

# define problem parameters
μ = 12.3958 # eV
Ωs = [0.0]
# Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
# T = 0.0 # K # this will break my window functions
T = 50.0 # K # guess of the effective temperature

# define constants
kB = 8.617333262e-5 # eV/K
n = 0 # zeroth kinetic coefficient == OC

# derived parameters
β = inv(kB*T)

# set error tolerances
atol = 1e-1
rtol = 0.0

# run script
results = AutoBZ.Jobs.run_kinetic(HV, Σ, β, μ, n, Ωs, atol, rtol)
AutoBZ.Jobs.write_nt_to_h5(results, "OC_results_ftps.h5")
