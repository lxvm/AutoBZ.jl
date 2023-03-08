#=
In this script we compute OC with the BZ integral on the inside and the
frequency integral on the outside at various Ω at a single η, where the
temperature is inferred from a Fermi liquid scaling, i.e. η = c*T^2
=#

# using SymmetryReduceBZ # add package to use bzkind=:ibz
using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
hv, bz = load_wannier90_data("svo"; gauge=:Wannier, vkind=:covariant, vcomp=:whole, bzkind=:cubicsymibz)

# define problem parameters
μ = 12.3958 # eV
Ωs = range(0, 1, length=10) # eV
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

# setup algorithm for Brillouin zone integral
npt = 50; kalg = PTR(; npt=npt, rule=AutoBZCore.alloc_rule(hv, DT, bz.syms, npt))
# kalg = AutoPTR(; buffer=AutoBZCore.alloc_autobuffer(hv, DT, bz.syms))
## alert: IAI does not compile quickly (trying to fix this bug)
# kalg = IAI(; order=7, segbufs=AutoBZCore.alloc_segbufs(DT,RT,NT,ndims(hv)))

# create frequency integrand, which evaluates a bz integral at each frequency point
oc_integrand = OpticalConductivityIntegrand(bz, kalg, hv, Σ, β; abstol=atol, reltol=rtol)

# setup limits and algorithm for frequency integral
## alert: cannot allocate a segbuf yet. See https://github.com/SciML/Integrals.jl/pull/151
falg = QuadGKJL()#; segbuf=alloc_segbuf(DT, RT, NT))

# construct solver
oc_solver = IntegralSolver(oc_integrand, lb(Σ), ub(Σ), falg; abstol=atol, reltol=rtol)

# run calculation
nthreads = kalg isa AutoPTR ? 1 : Threads.nthreads() # kpt parallelization (default) is preferred for large k-grids
h5batchsolve("oc_fl_fk.h5", oc_solver, Ωs, RT; nthreads=nthreads)

# show kpts/dim of converged ptr grid
kalg isa AutoPTR && @show kalg.buffer.npt1[] kalg.buffer.npt2[]

nothing