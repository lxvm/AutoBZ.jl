#=
In this script we compute optical conductivity at various frequencies.
Users have the following choices:
- load_wannier90_data: see documentation for parameters
- Self energies: Fermi liquid scaling, i.e. η = c*T^2 or FTPS data
- BZ integral algorithms: IAI, PTR, AutoPTR
- Order of integration: frequency then BZ integral or vice versa
=#

# using SymmetryReduceBZ # add package to use bz=IBZ()
using HDF5  # load before AutoBZ
using AutoBZ

seed = "svo"; μ = 12.3958 # eV
# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone
hv, bz = load_wannier90_data(seed; gauge=Wannier(), interp=CovariantVelocityInterp, coord=Cartesian(), vcomp=Whole(), bz=CubicSymIBZ())
shift!(hv, μ) # shift the Fermi energy to zero

# define problem parameters

Ωs = range(0, 1, length=10) # eV

# Use Fermi liquid scaling of self energy
η = 0.5 # eV
Σ = EtaSelfEnergy(η)

# Use self energies
# Σ = load_self_energy("svo_self_energy_scalar.txt")


# define constants
kB = 8.617333262e-5 # eV/K

T₀ = 300
Z  = 0.5
c = kB*pi/(Z*T₀)

# derived parameters
T = sqrt(η/c)
β = inv(kB*T)

# set error tolerances
rtol = 1e-3
atol = 1e-2

falg = QuadGKJL() # adaptive algorithm for frequency integral

# setup algorithm for Brillouin zone integral
npt = 15
kalg = PTR(; npt=npt)
# kalg = AutoPTR()
# kalg = IAI()
#= alternative algorithms that save work for IAI when requesting a reltol
kalg = AutoPTR_IAI(; ptr=PTR(; npt=npt), iai=IAI())
kalg = AutoPTR_IAI(; ptr=AutoPTR(), iai=IAI())
=#


# create integrand with bz integral on the inside
oc_integrand = OpticalConductivityIntegrand(bz, kalg, hv, Σ, β; abstol=atol, reltol=rtol)
oc_solver = IntegralSolver(oc_integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg; abstol=atol, reltol=rtol)

# create integrand with frequency integral on the inside
# oc_integrand = OpticalConductivityIntegrand(falg, hv, Σ, β; abstol=atol/nsyms(bz), reltol=rtol)
# oc_solver = IntegralSolver(oc_integrand, bz, kalg; abstol=atol, reltol=rtol)

# run calculation
nthreads = kalg isa AutoPTR ? 1 : Threads.nthreads() # kpt parallelization (default) is preferred for large k-grids

results = h5open("oc.h5", "w") do h5
    batchsolve(h5, oc_solver, Ωs; nthreads=nthreads)
end

# show kpts/dim of converged ptr grid
kalg isa AutoPTR && @show kalg.buffer.npt1[] kalg.buffer.npt2[]

nothing
