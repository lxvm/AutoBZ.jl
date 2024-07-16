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


# create integrand with bz integral on the inside
oc_solver = OpticalConductivitySolver(Σ, falg, hv, bz, kalg; β, Ω=first(Ωs), abstol=atol, reltol=rtol)

# create integrand with frequency integral on the inside
# oc_solver = OpticalConductivitySolver(hv, bz, kalg, Σ, falg; β, Ω=first(Ωs), abstol=atol, reltol=rtol)

# run calculation
results = h5open("oc.h5", "w") do h5
    h5["frequencies"] = collect(Ωs)
    dat_oc = create_dataset(h5, "conductivities", ComplexF64, (3, 3, length(Ωs)))
    for (i, Ω) in enumerate(Ωs)
        AutoBZ.update_oc!(oc_solver; β, Ω)
        sol = solve!(oc_solver)
        dat_oc[:,:,i] = sol.value
    end
    dat_oc[:,:,:]
end
