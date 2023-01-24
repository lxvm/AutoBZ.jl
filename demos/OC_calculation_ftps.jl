#=
In this script we compute OC at various Ω using frequency-dependent self energy
data that we interpolate with a high-order Chebyshev regression
=#

using AutoBZ
using AutoBZ.Jobs

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
HV, FBZ = load_wannier90_data("svo"; velocity_kind=:orbital)

ibz_limits = AutoBZ.TetrahedralLimits(period(HV)) # Cubic symmetries
IBZ = IrreducibleBZ(FBZ.a, FBZ.b, ibz_limits)

# load the self energy data
Σ = load_self_energy("svo_self_energy_scalar.txt")

# define problem parameters
μ = 12.3958 # eV
Ωs = 0.0
# Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
# T = 0.0 # K # this will break my window functions
T = 50.0 # K # guess of the effective temperature

shift!(HV, μ) # shift the Fermi energy to zero

# define constants
kB = 8.617333262e-5 # eV/K
β = inv(kB*T)
n = 0 # zeroth kinetic coefficient == OC

# set error tolerances
atol = 1e-1
rtol = 0.0

# run script
results = AutoBZ.Jobs.run_kinetic(HV, Σ, β, n, Ωs, IBZ, rtol, atol)
AutoBZ.Jobs.write_nt_to_h5(results, "OC_results_ftps.h5")
