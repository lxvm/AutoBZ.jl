#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
=#

using AutoBZ
using AutoBZ.Jobs

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
H, FBZ = load_wannier90_data("svo")

IBZ = Jobs.cubic_sym_ibz(FBZ; atol=1e-5) # for lattices with cubic symmetry only


# Define problem parameters
ω = 0.0 # eV
η = 0.1 # eV
μ = 12.3958 # eV

shift!(H, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)

D = DOSIntegrand(H, Σ, ω)

# set error tolerances
atol = 1e-3
rtol = 0.0

@show atol rtol

# int_fbz, err_fbz = AutoBZ.iterated_integration(D, FBZ; atol=atol, rtol=rtol)
# inte_fbz, erre_fbz, pre_fbz = AutoBZ.automatic_equispace_integration(D, FBZ; atol=atol, rtol=rtol)

int_ibz, err_ibz = AutoBZ.iterated_integration(D, IBZ; atol=atol, rtol=rtol)
# inte_ibz, erre_ibz, pre_ibz = AutoBZ.automatic_equispace_integration(D, IBZ; atol=atol, rtol=rtol)

# @show int_fbz int_ibz inte_fbz inte_ibz

SD = SafeDOSIntegrand(H, Σ, ω)
#=
safe_int_fbz, safe_err_fbz = AutoBZ.iterated_integration(SD, FBZ; atol=atol, rtol=rtol)
safe_inte_fbz, safe_erre_fbz, safe_pre_fbz = AutoBZ.automatic_equispace_integration(SD, FBZ; atol=atol, rtol=rtol)

safe_int_ibz, safe_err_ibz = AutoBZ.iterated_integration(SD, IBZ; atol=atol, rtol=rtol)
safe_inte_ibz, safe_erre_ibz, safe_pre_ibz = AutoBZ.automatic_equispace_integration(SD, IBZ; atol=atol, rtol=rtol)

@show safe_int_fbz safe_int_ibz safe_inte_fbz safe_inte_ibz
=#
nothing