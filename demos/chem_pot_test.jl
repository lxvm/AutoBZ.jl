#=
In this script we compute the electron count at single chemical potential using the interface in AutoBZ.jl
=#

using AutoBZ.Jobs

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
H, FBZ = load_wannier90_data("svo")

IBZ = Jobs.cubic_sym_ibz(FBZ; atol=1e-5) # for lattices with cubic symmetry only


# Define problem parameters
μ = 12.3958 # eV
η = 0.1 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling
Σ = EtaSelfEnergy(η)

# set error tolerances
atol = 1e-3
rtol = 0.0
tols = (atol=atol, rtol=rtol)
flims = Jobs.CubicLimits(μ-20.0, μ+20.0) # TODO: figure out how to safely truncate frequency integral limits

# fully adaptive integration
n = ElectronCountIntegrand(H, Σ, β, μ; lims=flims, quad_kw=tols) # here quad_kw passes info to the inner integrator
@time int, err = Jobs.iterated_integration(n, IBZ; tols...)

# adaptive in frequency, automatic equispace in BZ
# n = ElectronCountIntegrator(IBZ, H, Σ, β; tols..., lims=lims, quad_kw=tols) # adaptive default
n = ElectronCountIntegrator(IBZ, H, Σ, β; npt=100, routine=Jobs.equispace_integration, lims=flims, quad_kw=tols)
# n = ElectronCountIntegrator(IBZ, H, Σ, β; routine=Jobs.automatic_equispace_integration, lims=flims, tols..., quad_kw=tols)

@time n(μ)
