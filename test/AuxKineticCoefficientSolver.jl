using Test
using AutoBZ

for d in 1:3
    C = integer_lattice(d)
    bz = load_bz(CubicSymIBZ(d))
    for h in [
        let h=coeffs2FourierHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), LAPACKEigen(); gauge=Wannier()); end,
        let h=coeffs2HermitianHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), JLEigen(); gauge=Wannier()); end,
        let h=coeffs2RealSymmetricHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), JLEigen(); gauge=Wannier()); end,
    ]
        η = 1.0
        Σ = EtaSelfEnergy(η)
        β = 1.0
        μ = 0.1
        Ω = 0.0
        n = 0
        falg = QuadGKJL()
        abstol=1e-2
        reltol=0.0
        for alg in (PTR(; npt=50), IAI(), AutoPTR(a=η))
            solver = AuxKineticCoefficientSolver(Σ, falg, h, bz, alg; β, μ, Ω, n, abstol, reltol)
            sol1 = solve!(solver)
            AutoBZ.update_auxkc!(solver; β, μ, Ω, n)
            sol2 = solve!(solver)
            # display(sol2.value)
            # @test sol1.value-0.5 ≈ -(sol2.value-0.5) atol=abstol rtol=reltol
            solver = AuxKineticCoefficientSolver(h, bz, alg, Σ, falg; β, μ, Ω, n, abstol, reltol)
            sol3 = solve!(solver)
            AutoBZ.update_auxkc!(solver; β, μ, Ω, n)
            sol4 = solve!(solver)
            # # @test sol3.value-0.5 ≈ -(sol4.value-0.5) atol=abstol rtol=reltol
            @test sol1.value.val ≈ sol3.value.val atol=abstol rtol=reltol
            @test sol2.value.val ≈ sol4.value.val atol=abstol rtol=reltol

        end
    end
end
