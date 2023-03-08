#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
=#

# using SymmetryReduceBZ # add package to use bzkind=:ibz
using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
h, bz = load_wannier90_data("svo"; bzkind=:cubicsymibz)

# Define problem parameters
ω = 0.0 # eV
η = 0.1 # eV
μ = 12.3958 # eV

shift!(h, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)


# set error tolerances
atol = 1e-3
rtol = 0.0
npt = 100

integrand = DOSIntegrand(h, Σ)

for alg in (IAI(), TAI(), PTR(; npt=npt), AutoPTR())
    @show typeof(alg)
    solver = IntegralSolver(integrand, bz, alg; abstol=atol, reltol=rtol)
    @time @show solver(ω)
    println()
end
