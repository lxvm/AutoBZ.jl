#=
In this script we compute OC at various Ω at a single η, where the temperature
is inferred from a Fermi liquid scaling, i.e. η = c*T^2
=#

using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
HV, FBZ = load_wannier90_data("svo"; velocity_kind=:orbital)

ibz_limits = AutoBZ.TetrahedralLimits(period(HV)) # Cubic symmetries
IBZ = IrreducibleBZ(FBZ.a, FBZ.b, ibz_limits)

# define problem parameters
μ = 12.3958 # eV
Ωs = 0.0
# Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
η = 0.5 # eV

shift!(H, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)

# define constants
kB = 8.617333262e-5 # eV/K
n = 0 # zeroth kinetic coefficient == OC

T₀ = 300
Z  = 0.5
c = kB*pi/(Z*T₀)

# derived parameters
T = sqrt(η/c)
β = inv(kB*T)

# set error tolerances
atol = 1e-3
rtol = 0.0

# run script
results = AutoBZ.Jobs.run_kinetic_adaptive(HV, Σ, β, n, Ωs, IBZ, rtol, atol)
AutoBZ.Jobs.write_nt_to_h5(results, "OC_results_fermi_liquid.h5")
