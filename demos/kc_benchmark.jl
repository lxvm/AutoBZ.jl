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

# create frequency integrand, which evaluates a bz integral at each frequency
kc_solver = KineticCoefficientSolver(Σ, falg, hv, bz, kalg; n=0, β, Ω=first(Ωs), abstol=atol/nsyms(bz), reltol=rtol)

# create bz integrand, which evaluates a frequency integral at each kpt
# kc_solver = KineticCoefficientSolver(hv, bz, kalg, Σ, falg; n=0, β, Ω=first(Ω), abstol=atol/nsyms(bz), reltol=rtol)

# run calculation
h5open("sro_tetra_oc_fl_ptr_eta$(η)_atol$(atol)_rtol$(rtol)_k$(npt).h5", "w") do h5
    kc_0 = create_group(h5, "kc_0")
    kc_0["frequencies"] = collect(Ωs)
    kc_0_dat = create_dataset(kc_0, "conductivities", ComplexF64, (3,3,length(Ωs)))
    for (i, Ω) in enumerate(Ωs)
        AutoBZ.update_kc!(kc_solver; Ω, β, n=0)
        sol = solve!(kc_solver)
        kc_0_dat[:,:,i] = sol.value
    end
    kc_1 = create_group(h5, "kc_1")
    kc_1["frequencies"] = [0.0]
    kc_1_dat = create_dataset(kc_1, "conductivities", ComplexF64, (3,3,1))
    AutoBZ.update_kc!(kc_solver; Ω=0.0, β, n=1)
    sol = solve!(kc_solver)
    kc_1_dat[:,:,1] = sol.value
end
