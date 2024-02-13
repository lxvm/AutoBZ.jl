#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using LinearAlgebra

# using SymmetryReduceBZ # add package to use bz=IBZ()
using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone
# keywords
seed = "svo"

prec = Float32

hv, bz = load_wannier90_data(seed; interp=CovariantVelocityInterp, gauge=Wannier(), vcomp=Whole(), coord=Cartesian(), bz=CubicSymIBZ(), precision=prec)

# Define problem parameters
Ω = convert(prec, 0.0) # eV
η = convert(prec, 1.0) # eV
μ = convert(prec, 12.3958) # eV
β = convert(prec, inv(sqrt(η*8.617333262e-5*0.5*300/pi))) # eV # Fermi liquid scaling
# shift!(hv, μ)
Σ = EtaSelfEnergy(η)

# use self energies
# Σ = load_self_energy("svo_self_energy_scalar.txt", precision=prec)

# set error tolerances
atol = 1e-2
rtol = 1e-3
npt = 25

falg = QuadGKJL() # adaptive algorithm for frequency integral

kalgs = (IAI(), PTR(; npt=npt),)# AutoPTR()) # BZ algorithms

w = FourierSeriesEvaluators.workspace_allocate(hv, FourierSeriesEvaluators.period(hv), (1,1,4))

# loop to test various routines with the frequency integral on the inside
integrand = OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, w; Σ, β, abstol=atol/nsyms(bz), reltol=rtol)
for kalg in kalgs
    @show nameof(typeof(kalg))
    solver = IntegralSolver(integrand, bz, kalg; abstol=atol, reltol=rtol)
    @time @show solver(; Ω, μ)
end

# loop to test various routines with the frequency integral on the outside
for kalg in kalgs
    local integrand = OpticalConductivityIntegrand(bz, kalg, w; Σ, β, abstol=atol, reltol=rtol)
    @show nameof(typeof(kalg))
    solver = IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg; abstol=atol, reltol=rtol)
    @time @show solver(; Ω, μ)
end
