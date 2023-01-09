#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
=#

using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
H, FBZ = load_wannier90_data("svo")

ibz_limits = AutoBZ.TetrahedralLimits(period(H)) # Cubic symmetries
IBZ = IrreducibleBZ(FBZ.a, FBZ.b, ibz_limits)

# Define problem parameters
ω = 0.0 # eV
η = 0.1 # eV
μ = 12.3958 # eV

shift!(H, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)

D = DOSIntegrand(H, ω, Σ)

# set error tolerances
atol = 1e-3
rtol = 0.0

int, err = AutoBZ.iterated_integration(D, IBZ; atol=atol, rtol=rtol)
inte, erre, other = AutoBZ.automatic_equispace_integration(D, IBZ; atol=atol, rtol=rtol)
