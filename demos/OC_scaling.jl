#=
In this script, we extract timings for adaptive and equispace OC for diminishing
eta
=#

using AutoBZ

# define the periods of the axes of the Brillouin zone for example material
period = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
HV = AutoBZ.Applications.load_hamiltonian_velocities("svo_hr.dat"; period=period)

# Define problem parameters
Ω = 0.0 # eV
η = 1.0 # eV
μ = 12.3958 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling

# initialize integrand and limits
Σ = AutoBZ.Applications.EtaEnergy(η)
σ = AutoBZ.Applications.OCIntegrand(HV, Σ, Ω, β, μ)
f = AutoBZ.Applications.fermi_window_limits(Ω, β)
c = AutoBZ.CubicLimits(H.period)
t = AutoBZ.Applications.TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

int, err = AutoBZ.iterated_integration(σ, AutoBZ.CompositeLimits(t, f); atol=atol, rtol=rtol)
Eσ = AutoBZ.Applications.AutoEquispaceOCIntegrand(σ, t, atol, rtol)
inte, erre = AutoBZ.iterated_integration(Eσ, f; atol=atol, rtol=rtol)
