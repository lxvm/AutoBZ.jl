#=
In this script we compute the DOS of SrVO3 aacross a range of frequencies
=#

using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
H, FBZ = load_wannier90_data("svo")

ibz_limits = AutoBZ.TetrahedralLimits(period(H)) # Cubic symmetries
IBZ = IrreducibleBZ(FBZ.a, FBZ.b, ibz_limits)

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
results = AutoBZ.Jobs.run_dos(shift!(H, μ), Σ, ωs, IBZ, rtol, atol)
# AutoBZ.Jobs.write_nt_to_h5(results, "DOS_calculation.h5")