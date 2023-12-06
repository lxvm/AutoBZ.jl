#=
In this script we compute kinetic coefficients using the PTR algorithm with the
BZ integral on the inside and the frequency integral on the outside at various Ω
at a single η, where the temperature is inferred from a Fermi liquid scaling,
i.e. η = c*T^2
=#

using SymmetryReduceBZ  # load package to use bz=IBZ()
using HDF5              # load package for batchsolve
using AutoBZ

seed = "svo"; μ = 12.3958 # eV

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone
hv, bz = load_wannier90_data(seed; gauge=Hamiltonian(), interp=CovariantVelocityInterp, coord=Cartesian(), vcomp=Inter(), bz=IBZ())

# define problem parameters
Ωs = pushfirst!(10.0 .^ range(-2.5, 1.0, length=50), 0.0)
η = 0.5 # eV

shift!(hv, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)

# define constants
kB = 8.617333262e-5 # eV/K

T₀ = 25
Z  = 0.3
c = kB*pi/(Z*T₀)

# derived parameters
T = sqrt(η/c)
β = inv(kB*T)

# set error tolerances
rtol = 1e-3
atol = 1e-3

falg = QuadGKJL() # adaptive algorithm for frequency integral

# setup algorithm for Brillouin zone integral
npt = 15
kalg = PTR(; npt=npt)
# kalg = AutoPTR()
# kalg = IAI()
#= alternative algorithms that save work for IAI when requesting a reltol
kalg = PTR_IAI(; ptr=PTR(; npt=npt), iai=IAI())
kalg = AutoPTR_IAI(; ptr=AutoPTR(), iai=IAI())
=#

# create frequency integrand, which evaluates a bz integral at each frequency
kc_integrand = KineticCoefficientIntegrand(bz, kalg, hv; Σ, abstol=atol/nsyms(bz), reltol=rtol)
# construct solver
kc_solver = IntegralSolver(kc_integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg; abstol=atol, reltol=rtol)

#=
# create bz integrand, which evaluates a frequency integral at each kpt
kc_integrand = KineticCoefficientIntegrand(falg, hv, Σ; abstol=atol/nsyms(bz), reltol=rtol)
# construct solver
kc_solver = IntegralSolver(kc_integrand, bz, kalg; abstol=atol, reltol=rtol)
=#

# run calculation
nthreads = kalg isa AutoPTR ? 1 : Threads.nthreads() # kpt parallelization (default) is preferred for large k-grids
h5open("sro_tetra_oc_fl_ptr_eta$(η)_atol$(atol)_rtol$(rtol)_k$(npt).h5", "w") do h5
    kc_0 = create_group(h5, "kc_0")
    batchsolve(kc_0, kc_solver, paramproduct(Ω=Ωs, n=0, β=β); nthreads=nthreads)
    kc_1 = create_group(h5, "kc_1")
    batchsolve(kc_1, kc_solver, paramproduct(Ω=0.0, n=1, β=β); nthreads=nthreads)
end
# show the number of points used on the ibz
kalg isa AutoPTR && @show length.(kc_integrand.p[1].cacheval.cache)
# show npt/dim
kalg isa AutoPTR && bz.syms !== nothing && @show getproperty.(kc_integrand.p[1].cacheval.cache, :npt)

nothing
