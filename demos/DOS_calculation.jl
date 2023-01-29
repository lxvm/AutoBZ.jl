#=
In this script we compute the DOS of SrVO3 across a range of frequencies
=#

using AutoBZ.Jobs

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
H, FBZ = load_wannier90_data("svo")

IBZ = Jobs.cubic_sym_ibz(FBZ; atol=1e-5) # for lattices with cubic symmetry only

# define problem parameters
μ = 12.3958 # eV
ωs = range(-1, 1, length=10)
η = 0.01 # eV

shift!(H, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)

# set error tolerances
atol = 1e-3
rtol = 1e-3

# run script
results = run_dos_adaptive(H, Σ, ωs, IBZ, rtol, atol)
# AutoBZ.Jobs.write_nt_to_h5(results, "DOS_calculation.h5")