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
# Load the (IBZ) limits of integration for the Brillouin zone
BZ = TetrahedralLimits(CubicLimits(period(HV)))
# load the self energy data
Σ = load_self_energy("svo_self_energy_scalar.txt")

# define problem parameters
μ = 12.3958 # eV
Ωs = 0.0
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
results = AutoBZ.Jobs.run_kinetic(HV, Σ, β, μ, n, Ωs, BZ, rtol, atol)
AutoBZ.Jobs.write_nt_to_h5(results, "OC_results_ftps.h5")
