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
Σ = EtaSelfEnergy(η)

# set error tolerances
atol = 1e-3
rtol = 0.0
tols = (atol=atol, rtol=rtol)

# fully adaptive integration
σ = KineticIntegrand(HV, 0, Σ, β, Ω; quad_kw=tols) # here quad_kw passes info to the inner integrator
@time int, err = Jobs.iterated_integration(σ, IBZ; tols...)

# adaptive in frequency, automatic equispace in BZ
σ = KineticIntegrator(IBZ, HV, 0, Σ, β; tols..., quad_kw=tols) # adaptive default
# σ = KineticIntegrator(IBZ, HV, 0, Σ, β; routine=Jobs.equispace_integration, quad_kw=tols)
# σ = KineticIntegrator(IBZ, HV, 0, Σ, β; routine=Jobs.automatic_equispace_integration, tols..., quad_kw=tols)

@time σ(Ω)
