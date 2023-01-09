using Plots

using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series
HV, FBZ = load_wannier90_data("svo"; velocity_kind=:orbital)

ibz_limits = AutoBZ.TetrahedralLimits(period(HV)) # Cubic symmetries
IBZ = IrreducibleBZ(FBZ.a, FBZ.b, ibz_limits)

# Define problem parameters
Ωs = range(0,2, length=1000) # eV
η = 0.001 # eV
Σ = EtaSelfEnergy(η)
μ = 12.3958 # eV
shift!(HV, μ)
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling

# set error tolerances
atol = 1e-3 # TODO, set the atol from rtol*norm(I_ptr)
rtol = 0.0

kx = 1.0
ky = 0.2
kz = 3.0

contract!(HV, kz, 3)
contract!(HV, ky, 2)
contract!(HV, kx, 1)

Ω = 0.9
l = fermi_window_limits(Ω, β)
ωs = range(AutoBZ.limits(l,1)..., length=1000)
A = KineticIntegrand(HV, Σ, β, 0, Ω)
plot(ωs, [real(first(A(ω))) for ω in ωs])

# results = AutoBZ.Jobs.test_run_kinetic(HV, Σ, β, 0, Ωs, rtol, atol, kx, ky, kz)
