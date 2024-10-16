#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using AutoBZ
using AutoBZ.Jobs

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
HV, FBZ = load_wannier90_data("svo"; velocity_kind=:orbital)

ibz_limits = AutoBZ.TetrahedralLimits(period(HV)) # Cubic symmetries
IBZ = IrreducibleBZ(FBZ.a, FBZ.b, ibz_limits)

# Define problem parameters
Ω = 0.0 # eV
η = 1.0 # eV
μ = 12.3958 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling

Σ = EtaSelfEnergy(η)
shift!(HV, μ)

# initialize integrand and limits
σ = KineticIntegrand(HV, Σ, β, 0, Ω)
f = fermi_window_limits(Ω, β)

# set error tolerances
atol = 1e-3
rtol = 0.0

# fully adaptive integration
int, err = AutoBZ.iterated_integration(σ, AutoBZ.CompositeLimits(IBZ, f); atol=atol, rtol=rtol)

# adaptive in frequency, automatic equispace in BZ
Eσ = AutoEquispaceKineticIntegrand(σ, IBZ, atol, rtol)

inte, erre = AutoBZ.iterated_integration(Eσ, f; atol=atol, rtol=rtol)
