#=
In this script we compute OC using the PTR algorithm with the BZ integral on the
inside and the frequency integral on the outside at various Ω at a single η,
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

# setup algorithm for Brillouin zone integral
## choose fixed number of kpts
npt = 50
kalg = PTR(; npt=npt, rule=AutoBZCore.alloc_rule(hv, DT, bz.syms, npt))
## choose automatic k-grid refinement
# kalg = AutoPTR(; buffer=AutoBZCore.alloc_autobuffer(hv, DT, bz.syms))

# create bz integrand
transport_solver = IntegralSolver(TransportDistributionIntegrand(hv, Σ), bz, kalg; abstol=atol, reltol=rtol)

# setup limits and algorithm for frequency integral
## cannot allocate a segbuf yet. See https://github.com/SciML/Integrals.jl/pull/151
falg = QuadGKJL()#; segbuf=alloc_segbuf(DT, RT, NT))
# create frequency integrand of the oc
oc_integrand(ω, (Γ, β, Ω)) = β*fermi_window(β*ω, β*Ω)*Γ((ω, ω+Ω))

## construct and test solver
oc_solver = IntegralSolver(oc_integrand, lb(Σ), ub(Σ), falg, (transport_solver, β); abstol=atol, reltol=rtol)

# run calculation
nthreads = kalg isa AutoPTR ? 1 : Threads.nthreads() # kpt parallelization (by default is preferred for large k-grids)
batchsolve("oc_fl_ptr.h5", oc_solver, Ωs, RT; nthreads=nthreads)

# show kpts/dim of converged ptr grid
kalg isa AutoPTR ? (@show kalg.buffer.npt1[] kalg.buffer.npt2[]) : (@show kalg.npt)
nothing