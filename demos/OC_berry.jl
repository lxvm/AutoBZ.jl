#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using AutoBZ

# define the periods of the axes of the Brillouin zone for example material
period = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
H = AutoBZ.Applications.load_hamiltonian("svo_hr.dat"; period=period)
# load the gradient of the Berry connection
A = AutoBZ.Applications.load_position_operator("svo_r.dat"; period=period)
# construct the Fourier series for the Hamiltonian and Berry-modified velocities
HV = AutoBZ.Applications.BandEnergyBerryVelocity(H, A)

# Define problem parameters
Ω = 0.0 # eV
η = 1.0 # eV
μ = 12.3958 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling

# initialize integrand and limits
Σ = AutoBZ.Applications.EtaEnergy(η)
σ = AutoBZ.Applications.OCIntegrand(HV, Σ, Ω, β, μ) # using the (Berry) band velocities
f = AutoBZ.Applications.fermi_window_limits(Ω, β)
c = AutoBZ.CubicLimits(H.period)
t = AutoBZ.Applications.TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

# fully adaptive integration
int, err = AutoBZ.iterated_integration(σ, AutoBZ.CompositeLimits(t, f); callback=AutoBZ.Applications.contract, atol=atol, rtol=rtol)

# adaptive in frequency, automatic equispace in BZ
# Eσ = AutoBZ.Applications.AutoEquispaceOCIntegrand(σ, t, atol, rtol; pre_eval=AutoBZ.Applications.pre_eval_fft, npt_update=AutoBZ.generic_npt_update)
Eσ = AutoBZ.Applications.AutoEquispaceOCIntegrand(σ, t, atol, rtol; pre_eval=AutoBZ.Applications.pre_eval_contract, npt_update=AutoBZ.generic_npt_update)
# Eσ = AutoBZ.Applications.AutoEquispaceOCIntegrand(σ, c, atol, rtol; pre_eval=AutoBZ.Applications.pre_eval_fft, npt_update=AutoBZ.generic_npt_update)
# Eσ = AutoBZ.Applications.AutoEquispaceOCIntegrand(σ, c, atol, rtol; pre_eval=AutoBZ.Applications.pre_eval_contract, npt_update=AutoBZ.generic_npt_update)

inte, erre = AutoBZ.iterated_integration(Eσ, f; atol=atol, rtol=rtol)
