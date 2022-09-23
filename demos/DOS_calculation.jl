#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
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
atol = 1e-5
rtol = 0.0

int, err = AutoBZ.iterated_integration(D, t; callback=AutoBZ.Applications.contract, atol=atol, rtol=rtol)
inte, erre, other... = AutoBZ.equispace_integration(D, t; atol=atol, rtol=rtol)