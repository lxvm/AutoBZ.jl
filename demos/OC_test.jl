#=
In this script we compute OC at single point using the interface in AutoBZ.jl
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
Ω = 0.0 # eV
η = 1.0 # eV
μ = 12.3958 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling

# initialize integrand and limits
Σ = AutoBZ.Applications.EtaEnergy(η)
σ = AutoBZ.Applications.OCIntegrand(H, Σ, Ω, β, μ)
f = AutoBZ.Applications.fermi_window_limits(Ω, β)
c = AutoBZ.CubicLimits(H.period)
t = AutoBZ.Applications.TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

# fully adaptive integration
int, err = AutoBZ.iterated_integration(σ, AutoBZ.CompositeLimits(t, f); callback=AutoBZ.Applications.contract, atol=atol, rtol=rtol)

# adaptive in frequency, automatic equispace in BZ
# Eσ = AutoBZ.Applications.EquispaceOCIntegrand(σ, t, atol, rtol; pre_eval=AutoBZ.Applications.pre_eval_fft, npt_update=AutoBZ.generic_npt_update)
Eσ = AutoBZ.Applications.EquispaceOCIntegrand(σ, t, atol, rtol; pre_eval=AutoBZ.Applications.pre_eval_contract, npt_update=AutoBZ.generic_npt_update)
# Eσ = AutoBZ.Applications.EquispaceOCIntegrand(σ, c, atol, rtol; pre_eval=AutoBZ.Applications.pre_eval_fft, npt_update=AutoBZ.generic_npt_update)
# Eσ = AutoBZ.Applications.EquispaceOCIntegrand(σ, c, atol, rtol; pre_eval=AutoBZ.Applications.pre_eval_contract, npt_update=AutoBZ.generic_npt_update)

inte, erre = AutoBZ.iterated_integration(Eσ, f; atol=atol, rtol=rtol)