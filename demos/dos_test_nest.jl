#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
=#

# using SymmetryReduceBZ # add package to use bz=IBZ()
using AutoBZ

seed = "svo"
prec = Float32
# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone
h, bz = load_wannier90_data(seed; precision=prec, interp=HamiltonianInterp, bz=CubicSymIBZ())

# Define problem parameters
ω = convert(prec, 0.0) # eV
η = convert(prec, 0.1) # eV
μ = convert(prec, 12.3958) # eV

shift!(h, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)


# set error tolerances
atol = 1e-3
rtol = 0.0
npt = 100

w = FourierSeriesEvaluators.workspace_allocate(h, FourierSeriesEvaluators.period(h), (1,1,4))
integrand = DOSIntegrand(w, Σ)

algs = (IAI(), TAI(), PTR(; npt=npt), AutoPTR())

for alg in algs
    @show nameof(typeof(alg))
    solver = IntegralSolver(integrand, bz, alg; abstol=atol, reltol=rtol)
    @time @show solver(ω=ω)
    println()
end
