#=
In this script we compute OC using the IAI algorithm with the BZ integral on the
outside and the frequency integral on the inside at various Ω at a single η,
where the temperature is inferred from a Fermi liquid scaling, i.e. η = c*T^2
=#

using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
hv, bz = load_wannier90_data("svo/svo"; gauge=:Wannier, vkind=:covariant, vcomp=:whole)

bz = AutoBZ.cubic_sym_ibz(bz; atol=1e-5) # for lattices with cubic symmetry only

# define problem parameters
μ = 12.3958 # eV
Ωs = [0.0]
Ωs = range(0, 1, length=10)
# Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
η = 0.5 # eV

shift!(hv, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)

# define constants
kB = 8.617333262e-5 # eV/K

T₀ = 300
Z  = 0.5
c = kB*pi/(Z*T₀)

# derived parameters
T = sqrt(η/c)
β = inv(kB*T)

# set error tolerances
rtol = 1e-3
atol = 1e-2

# setup oc_solver
# IMPORTANT: pre-allocating rules/memory for algorithms helps with performance
# compute types to pre-allocate resources for algorithms
DT = eltype(bz) # domain type of Brillouin zone
RT = fourier_type(hamiltonian(hv), DT) # range type of oc integrand
NT = Base.promote_op(AutoBZ.norm, RT) # norm type for RT

# setup limits and algorithm for frequency integral
## cannot allocate a segbuf yet. See https://github.com/SciML/Integrals.jl/pull/151
falg = QuadGKJL()#; segbuf=alloc_segbuf(DT, RT, NT))
# create bz integrand, which evaluates a frequency integral at each kpt
oc_integrand = OpticalConductivityIntegrand(hv, Σ, β; alg=falg, abstol=atol/nsyms(bz), reltol=rtol)

# setup algorithm for Brillouin zone integral
kalg = IAI(; order=7, segbufs=ntuple(n -> IteratedIntegration.alloc_segbuf(DT,RT,NT), ndims(hv)))

#= alternative algorithms that save work for IAI when requesting a reltol
npt = 50
ptr = PTR(; npt=npt, rule=AutoBZCore.alloc_rule(hv, DT, bz.syms, npt))
iai = IAI(; order=7, segbufs=ntuple(n -> IteratedIntegration.alloc_segbuf(DT,RT,NT), ndims(hv)))
kalg = AutoPTR_IAI(; ptr=ptr, iai=iai)
=#
#=
ptr = AutoPTR(; buffer=AutoBZCore.alloc_autobuffer(hv, DT, bz.syms))
iai = IAI(; order=7, segbufs=ntuple(n -> IteratedIntegration.alloc_segbuf(DT,RT,NT), ndims(hv)))
kalg = AutoPTR_IAI(; ptr=ptr, iai=iai)
=#

## construct and test solver
oc_solver = IntegralSolver(oc_integrand, bz, kalg; abstol=atol, reltol=rtol)

# run calculation
batchsolve("oc_fl_iai.h5", oc_solver, Ωs, RT)
