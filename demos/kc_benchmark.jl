#=
In this script we compute kinetic coefficients using the PTR algorithm with the
BZ integral on the inside and the frequency integral on the outside at various Ω
at a single η, where the temperature is inferred from a Fermi liquid scaling,
i.e. η = c*T^2
=#

using SymmetryReduceBZ  # load package to use bz=IBZ()
using HDF5              # load package for batchsolve
using AutoBZ

seed = "svo"; μ = 12.3958 # eV

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone
hv, bz = load_wannier90_data(seed; gauge=Hamiltonian(), interp=CovariantVelocityInterp, coord=Cartesian(), vcomp=Inter(), bz=IBZ())

# define problem parameters
Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
η = 0.005 # eV

shift!(hv, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)

# define constants
kB = 8.617333262e-5 # eV/K

T₀ = 25
Z  = 0.3
c = kB*pi/(Z*T₀)

# derived parameters
T = sqrt(η/c)
β = inv(kB*T)

# set error tolerances
rtol = 1e-2
atol = 1e-8

# setup kc_solver
# IMPORTANT: pre-allocating rules/memory for algorithms helps with performance
# compute types to pre-allocate resources for algorithms
DT = eltype(bz) # domain type of Brillouin zone
RT = typeof(complex(bz.A)) # range type of oc integrand
NT = Base.promote_op(AutoBZ.norm, RT) # norm type for RT

# setup limits and algorithm for frequency integral
## alert: cannot allocate a segbuf yet. See https://github.com/SciML/Integrals.jl/pull/151
falg = QuadGKJL()#; segbuf=alloc_segbuf(DT, RT, NT))

# setup algorithm for Brillouin zone integral
npt = 15
kalg = PTR(; npt=npt, rule=AutoBZCore.alloc_rule(hv, DT, bz.syms, npt))
# kalg = AutoPTR(; buffer=AutoBZCore.alloc_autobuffer(hv, DT, bz.syms))
## alert: IAI does not compile quickly (trying to fix this bug)
# kalg = IAI(; order=7, segbufs=AutoBZCore.alloc_segbufs(DT,RT,NT,ndims(hv)))

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

# create frequency integrand, which evaluates a bz integral at each frequency
kc_integrand = KineticCoefficientIntegrand(bz, kalg, hv, Σ; abstol=atol/nsyms(bz), reltol=rtol)
# construct solver
kc_solver = IntegralSolver(kc_integrand, lb(Σ), ub(Σ), falg; abstol=atol, reltol=rtol)

#=
# create bz integrand, which evaluates a frequency integral at each kpt
kc_integrand = KineticCoefficientIntegrand(falg, hv, Σ; abstol=atol/nsyms(bz), reltol=rtol)
# construct solver
kc_solver = IntegralSolver(kc_integrand, bz, kalg; abstol=atol, reltol=rtol)
=#

# run calculation
nthreads = kalg isa AutoPTR ? 1 : Threads.nthreads() # kpt parallelization (default) is preferred for large k-grids
h5open("sro_tetra_oc_fl_ptr_eta$(η)_atol$(atol)_rtol$(rtol)_k$(npt).h5", "w") do h5
    kc_0 = create_group(h5, "kc_0")
    batchsolve(kc_0, kc_solver, paramproduct(Ω=Ωs, n=0, β=β), RT; nthreads=nthreads)
    kc_1 = create_group(h5, "kc_1")
    batchsolve(kc_1, kc_solver, paramproduct(Ω=0.0, n=1, β=β), RT; nthreads=nthreads)
end
kalg isa AutoPTR && @show kalg.buffer.npt1[] kalg.buffer.npt2[]

nothing
