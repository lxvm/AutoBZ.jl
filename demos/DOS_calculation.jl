#=
In this script we compute the DOS of SrVO3 aacross a range of frequencies
=#

using AutoBZ
using AutoBZ.Applications

# define the periods of the axes of the Brillouin zone for example material
b = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
H = load_hamiltonian("svo_hr.dat"; period=b)

# define problem parameters
μ = 12.3958 # eV
ωs = range(-1, 1, length=10)
η = 0.01 # eV

# derived parameters
Σ = EtaSelfEnergy(η)

# set error tolerances
atol = 1e-3
rtol = 1e-3

# run script
results = AutoBZ.Jobs.run_dos(H, Σ, μ, ωs, rtol, atol)
# AutoBZ.Jobs.write_nt_to_h5(results, "DOS_calculation.h5")