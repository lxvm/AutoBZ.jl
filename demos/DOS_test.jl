#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
=#

using AutoBZ
using AutoBZ.Applications

# define the periods of the axes of the Brillouin zone for example material
b = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
H = load_hamiltonian("svo_hr.dat"; period=b)

# Define problem parameters
ω = 0.0 # eV
η = 0.1 # eV
μ = 12.3958 # eV

# initialize integrand and limits
Σ = EtaEnergy(η)
D = DOSIntegrand(H, ω, Σ, μ)
c = CubicLimits(period(H))
t = TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

int, err = iterated_integration(D, t; atol=atol, rtol=rtol)
inte, erre, other = automatic_equispace_integration(D, t; atol=atol, rtol=rtol)
