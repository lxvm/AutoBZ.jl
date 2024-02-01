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


kalgs = (IAI(), TAI(), AutoPTR(), PTR(; npt=npt))
falg = QuadGKJL()


# loop to test various routines with the frequency integral on the inside
integrand = ElectronDensityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, h; Σ, β, abstol=atol/nsyms(bz), reltol=rtol)
for kalg in kalgs
    @show nameof(typeof(kalg))
    solver = IntegralSolver(integrand, bz, kalg; abstol=atol, reltol=rtol)
    @time @show solver(; μ)
end

# loop to test various routines with the frequency integral on the outside
for kalg in kalgs
    local integrand = ElectronDensityIntegrand(bz, kalg, h; Σ, β, abstol=atol/nsyms(bz), reltol=rtol)
    @show nameof(typeof(kalg))
    solver = IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg; abstol=atol, reltol=rtol)
    @time @show solver(; μ)
end
