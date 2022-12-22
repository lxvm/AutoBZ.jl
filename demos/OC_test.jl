#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using AutoBZ
using AutoBZ.Applications

# define the periods of the axes of the Brillouin zone for example material
b = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
HV = load_hamiltonian_velocities("svo_hr.dat"; period=b)

# Define problem parameters
Ω = 0.0 # eV
η = 1.0 # eV
μ = 12.3958 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling

# initialize integrand and limits
Σ = EtaSelfEnergy(η)
σ = KineticIntegrand(HV, Σ, β, μ, 0, Ω)
f = fermi_window_limits(Ω, β)
c = CubicLimits(period(HV))
t = TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

# fully adaptive integration
int, err = iterated_integration(σ, CompositeLimits(t, f); atol=atol, rtol=rtol)

# adaptive in frequency, automatic equispace in BZ
Eσ = AutoEquispaceKineticIntegrand(σ, t, atol, rtol)

inte, erre = iterated_integration(Eσ, f; atol=atol, rtol=rtol)
