#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
=#

using AutoBZ

# define the periods of the axes of the Brillouin zone for example material
period = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
H = AutoBZ.Applications.load_hamiltonian("svo_hr.dat"; period=period)

# Define problem parameters
ω = 0.0 # eV
η = 0.1 # eV
μ = 12.3958 # eV

# initialize integrand and limits
Σ = AutoBZ.Applications.EtaEnergy(η)
D = AutoBZ.Applications.DOSIntegrand(H, ω, Σ, μ)
c = AutoBZ.CubicLimits(H.period)
t = AutoBZ.Applications.TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

int, err = AutoBZ.iterated_integration(D, t; atol=atol, rtol=rtol)
inte, erre, other = AutoBZ.automatic_equispace_integration(D, t; atol=atol, rtol=rtol)
# inte, erre, other = AutoBZ.automatic_equispace_integration(D, c; atol=atol, rtol=rtol)
