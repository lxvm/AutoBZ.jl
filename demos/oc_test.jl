#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using LinearAlgebra
using AutoBZ
using QuadGK: alloc_segbuf
using StaticArrays:SMatrix

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone
# keywords
# gauge: can only be :Wannier (:Hamiltonian needs to be reimplemented)
# vkind: can be :none for just the Hamiltonian, or :gradient for just the
# Hamiltonian gradient, or :covariant to include the Berry connection
# vcomp: can only be :whole (:inter and :intra need to be reimplemented)
hv, bz = load_wannier90_data("svo/svo"; gauge=:Wannier, vkind=:covariant, vcomp=:whole)

bz = AutoBZ.cubic_sym_ibz(bz; atol=1e-5) # for lattices with cubic symmetry only


# Define problem parameters
Ω = 0.0 # eV
η = 1.0 # eV
μ = 12.3958 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling
shift!(hv, μ)
Σ = EtaSelfEnergy(η)

# set error tolerances
atol = 1.0
rtol = 1e-3
quad_kw = (atol=atol/nsyms(bz), rtol=rtol) # error tolerance for innermost frequency integral
# k-points/dim for equispace integration
npt = 25

# put the frequency integral on the inside
## Note that this integrand calculates the frequency integration limits on the fly
falg = QuadGKJL()
oc = OpticalConductivityIntegrand(hv, Σ, β, Ω; alg=falg, abstol=atol/nsyms(bz), reltol=rtol)
prob = IntegralProblem(oc, bz)

# loop to test various routines
for alg in (PTR(; npt=npt), )
    @show typeof(alg)
    @time sol = solve(prob, alg; abstol=atol, reltol=rtol)
    @show sol.u
end

# put the frequency integral on the outside
function fermi_transport(ω, (hv, bz, Σ, β, Ω, alg, atol, rtol))
    ω₁ = ω; ω₂ = ω+Ω
    transport = TransportDistributionIntegrand(hv, ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂))
    prob = IntegralProblem(transport, bz)
    Γ = solve(prob, alg; abstol=atol, reltol=rtol).u
    β*fermi_window(β*ω, β*Ω)*Γ
end
alg2 = PTR(; npt=npt)
prob2 = IntegralProblem(fermi_transport, a, b, (hv, bz, Σ, β, Ω, alg2, atol, rtol))
@time sol2 = solve(prob2, QuadGKJL(); abstol=atol, reltol=rtol)


transport_solver = IntegralSolver(TransportDistributionIntegrand(hv, Σ), bz, alg2)
fermi_transport_solver(ω, (β, Ω, Γ)) = β*fermi_window(β*ω, β*Ω)*Γ((ω, ω+Ω))
prob3 = IntegralProblem(fermi_transport_solver, a, b, (β, Ω, transport_solver))
@time sol3 = solve(prob3, QuadGKJL(); abstol=atol, reltol=rtol)

# to run a sweep over frequencies, construct an integrator (pick one from below)
oc_integrand = OpticalConductivityIntegrand(hv, Σ, β; alg=falg, abstol=atol/nsyms(bz), reltol=rtol)
oc_solver = IntegralSolver(oc_integrand, bz, PTR(; npt=npt))
# then define the frequency sweep
Ωs = range(0, 2, length=50)
# then collect the results (and change from lattice basis to Cartesian basis)
results = map(Ω -> bz.A * oc_solver(Ω) * bz.A', Ωs)
# then plot them
using Plots
iA = inv(bz.A)
lat_results = map(σ -> iA * σ * iA', results)
plot(; xguide="Ω", yguide="σ_{αβ}", title="SRO")
plot!(Ωs, map(σ -> real(σ[1,1]), results); label="σ_{xx}")
plot!(Ωs, map(σ -> real(σ[2,2]), results); label="σ_{yy}")
plot!(Ωs, map(σ -> real(σ[3,3]), results); label="σ_{zz}")