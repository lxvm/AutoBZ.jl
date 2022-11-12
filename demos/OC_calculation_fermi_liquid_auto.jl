#=
In this script we compute OC at various Ω at a single η, where the temperature
is inferred from a Fermi liquid scaling, i.e. η = c*T^2
=#

using AutoBZ

# define the periods of the axes of the Brillouin zone for example material
period = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
HV = AutoBZ.Applications.load_hamiltonian_velocities("svo_hr.dat"; period=period)

# define problem parameters
μ = 12.3958 # eV
Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
η = 0.5 # eV

# define constants
kB = 8.617333262e-5 # eV/K

T₀ = 300
Z  = 0.5
c = kB*pi/(Z*T₀)

# derived parameters
Σ = AutoBZ.Applications.EtaEnergy(η)
T = sqrt(η/c)
β = inv(kB*T)

# set error tolerances
rtol = 1e-3
atol = 1e-2

# run script
results = AutoBZ.Jobs.OCscript_auto_parallel("OC_results_fermi_auto_rtol$(-floor(Int, log10(rtol))).h5", HV, Σ, β, Ωs, μ, rtol, atol)
