#=
In this script we compute the electron count at single chemical potential using the interface in AutoBZ.jl
=#
using LinearAlgebra

using AutoBZ
using EquiBaryInterp
import IteratedIntegration: alloc_segbufs

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
h, bz = load_wannier90_data("svo/svo")

bz = AutoBZ.cubic_sym_ibz(bz; atol=1e-5) # for lattices with cubic symmetry only


# Define problem parameters
μ = 12.3958 # eV
η = 0.1 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling
Σ = EtaSelfEnergy(η)

omegas, values = load_self_energy("svo_self_energy_scalar.txt")
Σ = ScalarSelfEnergy(LocalEquiBaryInterp(omegas, values), extrema(omegas)...)

T = 50 # K
β = 1/(8.617333262e-5*T) # eV

# set error tolerances
atol = 1e-3
rtol = 0.0
tols = (atol=atol, rtol=rtol)
quad_kw = (atol=atol/nsyms(bz), rtol=rtol) # error tolerance for innermost frequency integral
# k-points/dim for equispace integration
npt = 100

# fully adaptive integration with innermost frequency integral
density = ElectronDensityIntegrand(h,  Σ, β, μ, lb(Σ), ub(Σ))
prob = IntegralProblem(density, bz)

# vary routines for the BZ integral
for alg in (IAI(), PTR(; npt=npt))
    @show typeof(alg)
    @time sol = solve(prob, alg; abstol=atol, reltol=rtol)
    @show sol.u
end

# put the frequency integral outside

function fermi_dos(ω, (β, h, bz, Σ, alg, abstol, reltol))
    dos = DOSIntegrand(h, (ω+μ)*I-Σ(ω))
    prob = IntegralProblem(dos, bz)
    AutoBZ.fermi(β*ω)*solve(prob, alg; abstol=atol, reltol=rtol).u
end
alg2 = IAI(; segbufs=alloc_segbufs(Float64, ntuple(n->Float64,3), ntuple(n->Float64,3), 3))
prob2 = IntegralProblem(fermi_dos, lb(Σ), ub(Σ), (β, h, bz, Σ, alg2, atol, rtol))
@time sol2 = solve(prob2, QuadGKJL(); abstol=atol, reltol=rtol)

dos_solver = IntegralSolver(DOSIntegrand(h, Σ), bz, alg2; abstol=atol, reltol=rtol)
fermi_dos_solver(ω, (β, dos)) = fermi(β*ω)*dos(ω)
prob3 = IntegralProblem(fermi_dos_solver, lb(Σ), ub(Σ), (β, dos_solver))
@time sol3 = solve(prob3, QuadGKJL(); abstol=atol, reltol=rtol)