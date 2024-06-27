using Test
using AutoBZ

for d in 1:3
    C = integer_lattice(d)
    bz = load_bz(CubicSymIBZ(d))
    for h in [
        coeffs2FourierHamiltonian(C),
        coeffs2HermitianHamiltonian(C),
        coeffs2RealSymmetricHamiltonian(C),
    ]
        η = 1.0
        Σ = EtaSelfEnergy(η)
        ω = 0.1
        μ = 0.0
        abstol=1e-2
        reltol=0.0
        for alg in (PTR(; npt=50), IAI(), AutoPTR(a=η))
            solver = TrGloc2Solver(h, bz, alg; Σ, ω, μ, abstol, reltol)
            sol1 = solve!(solver)
            AutoBZ.update!(solver; Σ, ω=-ω, μ)
            sol2 = solve!(solver)
            @test sol1.value ≈ -conj(sol2.value) atol=abstol rtol=reltol
        end
    end
end