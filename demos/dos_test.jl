#=
In this script we compute DOS at single point using the interface in AutoBZ.jl
=#

using AutoBZ

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
h, bz = load_wannier90_data("svo/svo")

bz = AutoBZ.cubic_sym_ibz(bz; atol=1e-5) # for lattices with cubic symmetry only


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

dos = DOSIntegrand(h, Σ(ω))
# dos_integrand(h_k, Σ, ω) = SVector{1,Float64}(tr(imag(inv(ω*I-h_k - Σ(ω))))/(-pi))
# dos = FourierIntegrand(dos_integrand, h, Σ, ω)
prob = IntegralProblem(dos, bz)

for alg in (IAI(), TAI(), PTR(; npt=npt), AutoPTR())
    @show typeof(alg)
    @time sol = solve(prob, alg; abstol=atol, reltol=rtol)
    @show sol
    println()
end

dos_solver = IntegralSolver(DOSIntegrand(h, Σ), bz, IAI(); abstol=atol, reltol=rtol)
# dos_solver = IntegralSolver(FourierIntegrand(dos_integrand, h, Σ), bz, IAI(); abstol=atol, reltol=rtol)
@time dos_solver(ω)