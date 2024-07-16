#=
In this script we compute the electron count at single chemical potential using the interface in AutoBZ.jl
=#
using LinearAlgebra

# using SymmetryReduceBZ # add package to use bz=IBZ()
using AutoBZ

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


kalgs = (IAI(), AutoPTR(), PTR(; npt=npt))
falg = QuadGKJL()
fdom = (AutoBZ.lb(Σ), AutoBZ.ub(Σ))

# loop to test various routines with the frequency integral on the inside
for kalg in kalgs
    @show nameof(typeof(kalg))
    solver = ElectronDensitySolver(h, bz, kalg, Σ, fdom, falg; β, μ, abstol=atol, reltol=rtol)
    @time @show solve!(solver).value
end

# loop to test various routines with the frequency integral on the outside
for kalg in kalgs
    @show nameof(typeof(kalg))
    solver = ElectronDensitySolver(Σ, fdom, falg, h, bz, kalg; β, μ, abstol=atol, reltol=rtol)
    @time @show solve!(solver).value
end
