#=
In this script we compute OC at various Ω at a single η, where the temperature
is inferred from a Fermi liquid scaling, i.e. η = c*T^2
=#

using AutoBZ
using AutoBZ.Applications

# define the periods of the axes of the Brillouin zone for example material
b = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
HV = load_hamiltonian_velocities("svo_hr.dat"; period=b)
# Load the (IBZ) limits of integration for the Brillouin zone
BZ = TetrahedralLimits(CubicLimits(period(HV)))

# define problem parameters
μ = 12.3958 # eV
Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
η = 0.005 # eV
n = 0 # zeroth kinetic coefficient == OC

# define constants
kB = 8.617333262e-5 # eV/K

T₀ = 300
Z  = 0.5
c = kB*pi/(Z*T₀)

# derived parameters
Σ = EtaSelfEnergy(η)
T = sqrt(η/c)
β = inv(kB*T)

# set error tolerances
atol = 1e-3
rtol = 0.0

# set k-grid density for equispace integration (check solution is converged)
npt = 79

# run script
results = AutoBZ.Jobs.run_kinetic_equispace(HV, Σ, β, μ, n, Ωs, BZ, npt, rtol, atol)
AutoBZ.Jobs.write_nt_to_h5(results, "OC_results_fermi_liquid_equispace_$(npt)kpts.h5")
