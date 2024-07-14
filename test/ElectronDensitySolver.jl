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
        β = 1.0
        μ = 0.1
        fdom = (-Inf, Inf)
        falg = QuadGKJL()
        abstol=1e-2
        reltol=0.0
        for alg in (PTR(; npt=50), IAI(), AutoPTR(a=η))
            solver = ElectronDensitySolver(Σ, fdom, falg, h, bz, alg; β, μ, abstol, reltol)
            sol1 = solve!(solver)
            AutoBZ.update_density!(solver; β, μ=-μ)
            sol2 = solve!(solver)
            @test sol1.value-0.5 ≈ -(sol2.value-0.5) atol=abstol rtol=reltol
            solver = ElectronDensitySolver(h, bz, alg, Σ, fdom, falg; β, μ, abstol, reltol)
            sol3 = solve!(solver)
            AutoBZ.update_density!(solver; β, μ=-μ)
            sol4 = solve!(solver)
            @test sol3.value-0.5 ≈ -(sol4.value-0.5) atol=abstol rtol=reltol
            @test sol1.value ≈ sol3.value atol=abstol rtol=reltol
            @test sol2.value ≈ sol4.value atol=abstol rtol=reltol

        end
    end
end
