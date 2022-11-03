#=
In this script we compute OC at various Ω at a single η, where the temperature
is inferred from a Fermi liquid scaling, i.e. η = c*T^2
=#

using AutoBZ

include("Demos.jl")

# define the periods of the axes of the Brillouin zone for example material
period = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
H = AutoBZ.Applications.load_hamiltonian("svo_hr.dat"; period=period)

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
atol = 1e-3
rtol = 0.0

# run script
results = Demos.OCscript_parallel("OC_results_fermi_liquid.h5", H, Σ, β, Ωs, μ, atol, rtol)
