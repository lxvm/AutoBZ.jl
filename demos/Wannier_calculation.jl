#=
In this script we compute the DOS of SrVO3 aacross a range of frequencies
=#

using LinearAlgebra

using AutoBZ
using AutoBZ.Jobs

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
H, FBZ = load_wannier90_data("svo")
μ = 12.3958 # eV
shift!(H, μ)

ibz_limits = AutoBZ.TetrahedralLimits(period(H)) # Cubic symmetries
IBZ = IrreducibleBZ(FBZ.a, FBZ.b, ibz_limits)

# define problem parameters
ωs = range(-1, 1, length=10) # eV

# define integrand
greens_integrand(H, ω, η=0.01) = AutoBZ.Jobs.tr_inv(complex(ω,η)*I-H)/(-pi)

# set error tolerances
atol = 1e-3
rtol = 1e-3


# run script
results = AutoBZ.Jobs.equispace_fourier_integration(greens_integrand, IBZ, H, ωs, 100)
results = AutoBZ.Jobs.automatic_equispace_fourier_integration(greens_integrand, IBZ, H, ωs, rtol, atol)
results = AutoBZ.Jobs.adaptive_fourier_integration(greens_integrand, IBZ, H, ωs, rtol, atol)
results = AutoBZ.Jobs.auto_fourier_integration(greens_integrand, IBZ, H, ωs, rtol, atol)
# AutoBZ.Jobs.write_nt_to_h5(results, "Wannier_calculation.h5")