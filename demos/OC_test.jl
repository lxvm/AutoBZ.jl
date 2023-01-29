#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using AutoBZ.Jobs

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
HV, FBZ = load_wannier90_data("svo"; velocity_kind=:orbital, read_pos_op=false)

IBZ = Jobs.cubic_sym_ibz(FBZ; atol=1e-5) # for lattices with cubic symmetry only


# Define problem parameters
Ω = 0.0 # eV
η = 1.0 # eV
μ = 12.3958 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling
shift!(HV, μ)

# initialize integrand and limits
Σ = EtaSelfEnergy(η)
σ = KineticIntegrand(HV, 0, Σ, β, Ω)

# set error tolerances
atol = 1e-3
rtol = 0.0

# fully adaptive integration
int, err = Jobs.iterated_integration(σ, IBZ; atol=atol, rtol=rtol)

# adaptive in frequency, automatic equispace in BZ
σ = KineticIntegrator(HV, 0, Σ, β; atol=atol, rtol=rtol) # adaptive default
# σ = KineticIntegrator(HV, 0, Σ, β; routine=Jobs.equispace_integration)
# σ = KineticIntegrator(HV, 0, Σ, β; routine=Jobs.automatic_equispace_integration, atol=atol, rtol=rtol)

σ(Ω)
