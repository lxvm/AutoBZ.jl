using Test
using AutoBZ

for d in 1:3
    C = integer_lattice(d)
    bz = load_bz(CubicSymIBZ(d))
    for hv in [
        let h=coeffs2FourierHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), LAPACKEigen(); gauge=Hamiltonian()); end,
        let h=coeffs2HermitianHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), JLEigen(); gauge=Hamiltonian()); end,
        let h=coeffs2RealSymmetricHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), JLEigen(); gauge=Hamiltonian()); end,
    ]
        β = 1.0
        μ = 0.1
        abstol=1e-2
        reltol=0.0
        for alg in (PTR(; npt=50), IAI(), AutoPTR(a=inv(β)))
            solver = TransportFunctionSolver(hv, bz, alg; β, μ, abstol, reltol)
            sol1 = solve!(solver)
            AutoBZ.update_tf!(solver; β, μ=-μ)
            sol2 = solve!(solver)
            @test sol1.value ≈ sol2.value atol=abstol rtol=reltol
        end
    end
end
