#=
In this script we compute the electron count at single chemical potential using the interface in AutoBZ.jl
=#
using LinearAlgebra

# using SymmetryReduceBZ # add package to use bzkind=:ibz
using AutoBZ
using EquiBaryInterp

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
h, bz = load_wannier90_data("svo/svo"; bzkind=:cubicsymibz)

# Define problem parameters
μs = range(11, 20, length=4) # eV

# Fermi liquid scaling
# η = 0.1 # eV
# β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling
# Σ = EtaSelfEnergy(η)

omegas, values = load_self_energy("svo_self_energy_scalar.txt")
Σ = ScalarSelfEnergy(LocalEquiBaryInterp(omegas, values), extrema(omegas)...)

T = 50 # K
β = 1/(8.617333262e-5*T) # eV

# set error tolerances
atol = 1e-3
rtol = 0.0
npt = 100

# setup oc_solver
# IMPORTANT: pre-allocating rules/memory for algorithms helps with performance
# compute types to pre-allocate resources for algorithms
DT = eltype(bz) # domain type of Brillouin zone
RT = real(eltype(fourier_type(hamiltonian(h), DT))) # range type of oc integrand
NT = Base.promote_op(AutoBZ.norm, RT) # norm type for RT

# setup algorithm for Brillouin zone integral
# npt = 50; kalg = PTR(; npt=npt, rule=AutoBZCore.alloc_rule(hv, DT, bz.syms, npt))
# kalg = AutoPTR(; buffer=AutoBZCore.alloc_autobuffer(hv, DT, bz.syms))
## alert: IAI does not compile quickly (trying to fix this bug)
kalg = IAI(; order=7, segbufs=AutoBZCore.alloc_segbufs(DT,RT,NT,ndims(h)))
#= alternative algorithms that save work for IAI when requesting a reltol
npt = 50
ptr = PTR(; npt=npt, rule=AutoBZCore.alloc_rule(hv, DT, bz.syms, npt))
iai = IAI(; order=7, segbufs=AutoBZCore.alloc_segbufs(DT,RT,NT,ndims(hv)))
kalg = AutoPTR_IAI(; ptr=ptr, iai=iai)
=#
#=
ptr = AutoPTR(; buffer=AutoBZCore.alloc_autobuffer(hv, DT, bz.syms))
iai = IAI(; order=7, segbufs=AutoBZCore.alloc_segbufs(DT,RT,NT,ndims(hv)))
kalg = AutoPTR_IAI(; ptr=ptr, iai=iai)
=#

# setup algorithm for frequency integral
## alert: cannot allocate a segbuf yet. See https://github.com/SciML/Integrals.jl/pull/151
falg = QuadGKJL()#; segbuf=alloc_segbuf(DT, RT, NT))

# construct solver with frequency integral inside
n_integrand = ElectronDensityIntegrand(falg, h, Σ, β; abstol=atol/nsyms(bz), reltol=rtol)
n_solver = IntegralSolver(n_integrand, bz, kalg; abstol=atol, reltol=rtol)

# construct solver with BZ integral inside
# n_integrand = ElectronDensityIntegrand(bz, kalg, h, Σ, β; abstol=atol, reltol=rtol)
# n_solver = IntegralSolver(n_integrand, lb(Σ), ub(Σ), falg; abstol=atol, reltol=rtol)

# run calculation
nthreads = kalg isa AutoPTR ? 1 : Threads.nthreads() # kpt parallelization (default) is preferred for large k-grids
@show h5batchsolve("n_ftps.h5", n_solver, μs, RT; nthreads=nthreads)

# show kpts/dim of converged ptr grid
kalg isa AutoPTR && @show kalg.buffer.npt1[] kalg.buffer.npt2[]
