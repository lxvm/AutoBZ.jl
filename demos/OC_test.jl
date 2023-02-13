#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using AutoBZ.Jobs

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
HV, FBZ = load_wannier90_data("svo"; gauge=:Wannier, vcomp=:whole, read_pos_op=false)

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
quad_kw = (atol=atol/Jobs.nsyms(IBZ), rtol=rtol) # error tolerance for innermost frequency integral
# k-points/dim for equispace integration
npt = 100

# fully adaptive integration with innermost frequency integral
σ = KineticCoefficientIntegrand(HV, 0, Σ, β, Ω; quad_kw=quad_kw)
@time int, err = Jobs.iterated_integration(σ, IBZ; tols...)

# vary routines for the BZ integral
for (routine, kwargs) in (
    (Jobs.iterated_integration, tols),
    (Jobs.equispace_integration, (npt=npt,)),
    (Jobs.automatic_equispace_integration, tols)
)
    σ_ = KineticCoefficientIntegrator(IBZ, HV, 0, Σ, β; routine=routine, kwargs..., quad_kw=quad_kw)
    @show @time Jobs.norm(int - first(σ_(Ω)))
end

int
