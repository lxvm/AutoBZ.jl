#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
=#

# using SymmetryReduceBZ # add package to use bz=IBZ()
using AutoBZ

seed = "svo"
# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone
h, bz = load_wannier90_data(seed; interp=HamiltonianInterp, bz=InversionSymIBZ())

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


algs = (IAI(), TAI(), PTR(; npt=npt), AutoPTR())

for alg in algs
    @show nameof(typeof(alg))
    solver = DOSSolver(h, bz, alg; Σ, ω, abstol=atol, reltol=rtol)
    sol = @time solve!(solver)
    @show sol.value
    println()
end
