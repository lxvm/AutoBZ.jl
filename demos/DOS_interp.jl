#=
In this script we interpolate DOS over a frequency interval using the interface in AutoBZ.jl
=#

using Plots

using AutoBZ
using AutoBZ.Applications
using AutoBZ.AdaptChebInterp

# define the periods of the axes of the Brillouin zone for example material
b = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
H = load_hamiltonian("svo_hr.dat"; period=b)

# Define problem parameters
ω_lo = -2.0 # eV
ω_hi =  2.0 # eV
η = 0.1 # eV
μ = 12.3958 # eV

# initialize integrand and limits
Σ = EtaEnergy(η)
c = CubicLimits(period(H))
t = TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

D = DOSEvaluator(H, Σ, μ, t; atol=atol, rtol=rtol)
p = adaptchebinterp(D, ω_lo, ω_hi; atol=1e-1)

plot(range(ω_lo, ω_hi, length=1000), p; xguide="ω (eV)", yguide="DOS(ω)")
savefig("DOS_interp.png")