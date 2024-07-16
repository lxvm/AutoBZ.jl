using Test
using AutoBZ

for d in 1:3
    C = integer_lattice(d)
    bz = load_bz(CubicSymIBZ(d))
    for hv in [
        let h=coeffs2FourierHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), LAPACKEigen(); gauge=Hamiltonian()); end,
        let h=coeffs2HermitianHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), JLEigen(); gauge=Hamiltonian()); end,
        # let h=coeffs2RealSymmetricHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), JLEigen(); gauge=Hamiltonian()); end,
    ]
        η = 1.0
        Σ = EtaSelfEnergy(η)
        ω₁= 0.0
        ω₂= 0.1
        μ = 0.0
        abstol=1e-2
        reltol=0.0
        for alg in (PTR(; npt=50), IAI(), AutoPTR(a=η))
            solver = TransportDistributionSolver(Σ, hv, bz, alg; ω₁, ω₂, μ, abstol, reltol)
            sol1 = solve!(solver)
            AutoBZ.update_td!(solver; ω₁=ω₂, ω₂=ω₁, μ)
            sol2 = solve!(solver)
            @test sol1.value ≈ transpose(sol2.value) atol=abstol rtol=reltol
        end
    end
end
