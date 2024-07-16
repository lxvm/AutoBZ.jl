#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using LinearAlgebra

# using SymmetryReduceBZ # add package to use bz=IBZ()
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

# use self energies
# Σ = load_self_energy("svo_self_energy_scalar.txt")

# set error tolerances
atol = 1e-2
rtol = 1e-3
npt = 25

falg = QuadGKJL() # adaptive algorithm for frequency integral

kalgs = (IAI(), PTR(; npt=npt), AutoPTR()) # BZ algorithms

# loop to test various routines with the frequency integral on the inside
for kalg in kalgs
    @show nameof(typeof(kalg))
    solver = OpticalConductivitySolver(hv, bz, kalg, Σ, falg; Ω, β, abstol=atol, reltol=rtol)
    @time @show solve!(solver).value
end

# loop to test various routines with the frequency integral on the outside
for kalg in kalgs
    @show nameof(typeof(kalg))
    solver = OpticalConductivitySolver(Σ, falg, hv, bz, kalg; Ω, β, abstol=atol, reltol=rtol)
    @time @show solve!(solver).value
end
