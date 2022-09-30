#=
In this script we compute OC at various Ω at a single η, where the temperature
is inferred from a Fermi liquid scaling, i.e. η = c*T^2
=#

using StaticArrays

include("../src/AutoBZ.jl")

include("Demos.jl")

# import Fourier coefficients of Wannier Hamiltonian
coeffs = Demos.loadW90Hamiltonian("epsilon_mn.h5")
# define the periods of the axes of the Brillouin zone for example material
periods = fill(round(2π/3.858560, digits=6), SVector{3,Float64})
# construct the Hamiltonian datatype
H = AutoBZ.Applications.FourierSeries(coeffs, periods)

# define problem parameters
μ = 12.3958 # eV
Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
η = 0.005 # eV

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

# set k-grid density for equispace integration (check solution is converged)
npt = 79

# run script
results = Demos.OCscript_equispace_parallel("OC_results_fermi_liquid_equispace_$(npt)kpts.h5", H, Σ, β, Ωs, μ, npt, atol, rtol)