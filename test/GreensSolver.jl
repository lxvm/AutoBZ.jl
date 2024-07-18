using Test
using AutoBZ

for d in 1:3
    C = integer_lattice(d)
    bz = load_bz(FBZ(d))
    for h in [
        coeffs2FourierHamiltonian(C),
        coeffs2HermitianHamiltonian(C; gauge=Hamiltonian()),
        # coeffs2RealSymmetricHamiltonian(C),
    ]
        η = 1.0
        Σ = EtaSelfEnergy(η)
        ω = 0.1
        μ = 0.0
        abstol=1e-2
        reltol=0.0
        for bzalg in (PTR(; npt=50), IAI(), AutoPTR(a=η))
            for (fun, invsym) in ((identity, (-) ∘ adjoint), (AutoBZ.spectral_function, identity))
                for alg in (JLInv(), JLTrInv())
                    solver = AutoBZ._GreensSolver(fun, Σ, h, bz, bzalg, alg; ω, μ, abstol, reltol)
                    sol1 = solve!(solver)
                    AutoBZ.update_greens!(solver; ω=-ω, μ)
                    sol2 = solve!(solver)
                    @test sol1.value ≈ invsym(sol2.value) atol=abstol rtol=reltol
                end
            end
        end
    end
end
