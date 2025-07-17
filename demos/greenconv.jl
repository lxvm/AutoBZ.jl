using AutoBZ
using AutoBZ.StaticArrays


seed = "svo"
# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone
h, bz = load_wannier90_data(seed; interp=HamiltonianInterp, bz=FBZ(), gauge=Hamiltonian())

# Define problem parameters
ω = 0.0 # eV
η = 0.1 # eV
μ = 12.3958 # eV

shift!(h, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)


# set error tolerances
atol = 1e-3
rtol = 0.0
npt = 100
alg = IAI()

q = SVector(0.0, 0.0, 0.0)
ω = 0.0

solver = TwoGreensBZConvolutionSolver(Σ, h, bz, IAI(); q, ω, abstol=atol, reltol=rtol)

sol = solve!(solver)