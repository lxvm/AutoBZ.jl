#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using LinearAlgebra

# using SymmetryReduceBZ # add package to use bzkind=:ibz
using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone
# keywords
seed = "svo"
hv, bz = load_wannier90_data(seed; interp=CovariantVelocityInterp, gauge=Wannier(), vcomp=Whole(), coord=Cartesian(), bz=CubicSymIBZ())

# Define problem parameters
Ω = 0.0 # eV
η = 1.0 # eV
μ = 12.3958 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling
shift!(hv, μ)
Σ = EtaSelfEnergy(η)

#= this is how you use FTPS self energies
using EquiBaryInterp
omegas, values = load_self_energy("svo_self_energy_scalar.txt")
Σ = ScalarSelfEnergy(LocalEquiBaryInterp(omegas, values), extrema(omegas)...)
=#

# set error tolerances
atol = 1e-2
rtol = 1e-3
npt = 25

# setup kc_solver
# IMPORTANT: pre-allocating rules/memory for algorithms helps with performance
# compute types to pre-allocate resources for algorithms
DT = eltype(bz) # domain type of Brillouin zone
RT = typeof(complex(bz.A)) # range type of oc integrand
NT = Base.promote_op(AutoBZ.norm, RT) # norm type for RT

kalgs = (
    IAI(; order=7, segbufs=AutoBZCore.alloc_segbufs(DT,RT,NT,ndims(hv))),
    TAI(),
    AutoPTR(; buffer=AutoBZCore.alloc_autobuffer(hv, DT, bz.syms)),
    PTR(; npt=npt, rule=AutoBZCore.alloc_rule(hv, DT, bz.syms, npt)),
)

falg = QuadGKJL()

# loop to test various routines with the frequency integral on the inside
integrand = OpticalConductivityIntegrand(falg, hv, Σ; β=β, abstol=atol/nsyms(bz), reltol=rtol)
for kalg in kalgs
    @show nameof(typeof(kalg))
    solver = IntegralSolver(integrand, bz, kalg; abstol=atol, reltol=rtol)
    @time @show solver(Ω=Ω)
end

# loop to test various routines with the frequency integral on the outside
for kalg in kalgs
    local integrand = OpticalConductivityIntegrand(bz, kalg, hv, Σ; β=β, abstol=atol, reltol=rtol)
    @show nameof(typeof(kalg))
    solver = IntegralSolver(integrand, lb(Σ), ub(Σ), falg; abstol=atol, reltol=rtol)
    @time @show solver(Ω=Ω)
end
