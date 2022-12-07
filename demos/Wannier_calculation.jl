#=
In this script we compute the DOS of SrVO3 aacross a range of frequencies
=#

using LinearAlgebra

using AutoBZ
using AutoBZ.Applications

# define the periods of the axes of the Brillouin zone for example material
b = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
H = load_hamiltonian("svo_hr.dat"; period=b)

# define problem parameters
ωs = range(-1, 1, length=10) # eV

# define integrand
greens_integrand(H, ω; μ=12.3958, η=0.01) = AutoBZ.Applications.tr_inv(complex(ω+μ,η)*I-H)/(-pi)

# define limits of integration
lims = TetrahedralLimits(CubicLimits(period(H)))

# set error tolerances
atol = 1e-3
rtol = 1e-3


# run script
# results = AutoBZ.Jobs.run_wannier_equispace(greens_integrand, H, ωs, lims, 100)
# results = AutoBZ.Jobs.run_wannier_auto_equispace(greens_integrand, H, ωs, lims, rtol, atol)
# results = AutoBZ.Jobs.run_wannier_adaptive(greens_integrand, H, ωs, lims, rtol, atol)
results = AutoBZ.Jobs.run_wannier(greens_integrand, H, ωs, lims, rtol, atol)
# AutoBZ.Jobs.write_nt_to_h5(results, "Wannier_calculation.h5")