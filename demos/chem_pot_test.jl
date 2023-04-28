#=
In this script we compute the electron count at single chemical potential using the interface in AutoBZ.jl
=#
using LinearAlgebra

# using SymmetryReduceBZ # add package to use bzkind=:ibz
using AutoBZ
using EquiBaryInterp

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
h, bz = load_wannier90_data("svo"; interp=HamiltonianInterp, bz=CubicSymIBZ())

# Define problem parameters
μ = 11.0

# Fermi liquid scaling
# η = 0.1 # eV
# β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling
# Σ = EtaSelfEnergy(η)

Σ = load_self_energy("svo_self_energy_scalar.txt")

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
RT = real(eltype(fourier_type(h, DT))) # range type of density integrand
NT = Base.promote_op(AutoBZ.norm, RT) # norm type for RT

kalgs = (
    IAI(; order=7, segbufs=AutoBZCore.alloc_segbufs(DT,RT,NT,ndims(h))),
    TAI(),
    AutoPTR(; buffer=AutoBZCore.alloc_autobuffer(h, DT, bz.syms)),
    PTR(; npt=npt, rule=AutoBZCore.alloc_rule(h, DT, bz.syms, npt)),
)

# setup algorithm for frequency integral
## alert: cannot allocate a segbuf yet. See https://github.com/SciML/Integrals.jl/pull/151
falg = QuadGKJL()#; segbuf=alloc_segbuf(DT, RT, NT))


# loop to test various routines with the frequency integral on the inside
integrand = ElectronDensityIntegrand(falg, h, Σ; β=β, abstol=atol/nsyms(bz), reltol=rtol)
for kalg in (IAI(), TAI(), AutoPTR(), PTR(; npt=npt))
    @show nameof(typeof(kalg))
    solver = IntegralSolver(integrand, bz, kalg; abstol=atol, reltol=rtol)
    @time @show solver(μ=μ)
end

# loop to test various routines with the frequency integral on the outside
for kalg in (IAI(), TAI(), AutoPTR(),  PTR(; npt=npt))
    local integrand = ElectronDensityIntegrand(bz, kalg, h, Σ; β=β, abstol=atol/nsyms(bz), reltol=rtol)
    @show nameof(typeof(kalg))
    solver = IntegralSolver(integrand, lb(Σ), ub(Σ), falg; abstol=atol, reltol=rtol)
    @time @show solver(μ=μ)
end