#=
In this script we compute the electron count at single chemical potential using the interface in AutoBZ.jl
=#

using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
H, FBZ = load_wannier90_data("svo")

IBZ = AutoBZ.cubic_sym_ibz(FBZ; atol=1e-5) # for lattices with cubic symmetry only


# Define problem parameters
μ = 12.3958 # eV
η = 0.1 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling
Σ = EtaSelfEnergy(η)

Σ = load_self_energy("svo_self_energy_scalar.txt")
T = 50 # K
β = 1/(8.617333262e-5*T) # eV

# set error tolerances
atol = 1e-3
rtol = 0.0
tols = (atol=atol, rtol=rtol)
quad_kw = (atol=atol/Jobs.nsyms(IBZ), rtol=rtol) # error tolerance for innermost frequency integral
# k-points/dim for equispace integration
npt = 100

# fully adaptive integration with innermost frequency integral
n = ElectronDensityIntegrand(H, Σ, β, μ; quad_kw=tols) # here quad_kw passes info to the inner integrator
@time int, err = Jobs.iterated_integration(n, IBZ; tols...)

# vary routines for the BZ integral
for (routine, kwargs) in (
    (Jobs.iterated_integration, tols),
    (Jobs.equispace_integration, (npt=npt,)),
    (Jobs.automatic_equispace_integration, tols)
)
    n_ = ElectronDensityIntegrator(IBZ, H, Σ, β; routine=routine, kwargs..., quad_kw=quad_kw)
    @show @time Jobs.norm(int - first(n_(μ)))
end

@time n(μ)
