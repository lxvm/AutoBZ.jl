using AutoBZ
using AutoBZCore: IntegralProblem, solve

using Test

@testset "apps" for dims in (1, 2), alg in (IAI(), AutoPTR())
    h = HamiltonianInterp(integer_lattice(dims))
    ibz = load_bz(CubicSymIBZ(dims))
    fbz = load_bz(FBZ(dims))
    @testset "Green's function integrands" for integrand in (GlocIntegrand, DiagGlocIntegrand, TrGlocIntegrand, DOSIntegrand)
        h = h
        f = integrand(h)
        r = map((fbz, ibz)) do bz
            prob = IntegralProblem(f, bz)
            gint = IntegralSolver(prob, alg; abstol=1e-1)
            Σ = EtaSelfEnergy(1.0); ω = 1.0; μ = 0.1
            return @inferred(gint(; Σ, ω, μ))
        end
        # the Green's function integrands have unknown symmetry
        # representations but we are only integrating a scalar Hamiltonian
        # wrapped as a matrix
        @test r[1] ≈ r[2] atol=1e-1
    end
    @testset "TransportFunctionIntegrand" begin
        hv = GradientVelocityInterp(h, I, gauge=Hamiltonian())
        f = TransportFunctionIntegrand(hv)
        r = map((fbz, ibz)) do bz
            prob = IntegralProblem(f, bz)
            tfun = IntegralSolver(prob, alg; abstol=1e-1)
            # compare the various interfaces
            β = 1.0; μ = 0.1
            return @inferred(tfun(; β, μ))
        end
        @test r[1] ≈ r[2] atol=1e-1
    end
    @testset "TransportDistributionIntegrand" begin
        hv = GradientVelocityInterp(h, I, gauge=Hamiltonian())
        f = TransportDistributionIntegrand(hv)
        r = map((fbz, ibz)) do bz
            prob = IntegralProblem(f, bz)
            tdis = IntegralSolver(prob, alg; abstol=1e-1)
            # compare the various interfaces
            Σ = EtaSelfEnergy(1.0); ω₁ = 1.0; ω₂ = 2.0; μ = 0.1
            return @inferred(tdis(; Σ, ω₁, ω₂, μ))
        end
        @test r[1] ≈ r[2] atol=1e-1
    end
    @testset "KineticCoefficientIntegrand" for Ω in (0.0, 2.0)
        β = 1.0; μ = 0.1; n = 0
        hv = GradientVelocityInterp(h, I, gauge=Hamiltonian())
        Σ = EtaSelfEnergy(1.0)
        falg = QuadGKJL()
        r = map((fbz, ibz)) do bz
            # test both orders of integration
            # 1. bz integral inside
            f = KineticCoefficientIntegrand(bz, alg, hv; abstol=1e-2)
            prob = IntegralProblem(f, AutoBZ.lb(Σ), AutoBZ.ub(Σ))
            kcof = IntegralSolver(prob, falg; abstol=1e-1)
            anskw = @inferred(kcof(; Σ, n, β, Ω, μ))
            # 2. frequency integral inside
            f = KineticCoefficientIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, hv; abstol=1e-2)
            prob = IntegralProblem(f, bz)
            kcof = IntegralSolver(prob, alg; abstol=1e-1)
            answk = @inferred(kcof(; Σ, n, β, Ω, μ))
            return anskw, answk
        end
        @test r[1][1] ≈ r[1][2] atol=1e-1
        @test r[1][1] ≈ r[2][1] atol=1e-1
        @test r[1][1] ≈ r[2][2] atol=1e-1
    end
    @testset "ElectronDensityIntegrand" begin
        hv = h
        Σ = EtaSelfEnergy(1.0)
        β = 1.0; μ = 0.1
        falg = QuadGKJL()
        r = map((fbz, ibz)) do bz
            # test both orders of integration
            # 1. bz integral inside
            f = ElectronDensityIntegrand(bz, alg, hv; abstol=1e-2)
            prob = IntegralProblem(f, AutoBZ.lb(Σ), AutoBZ.ub(Σ))
            dens = IntegralSolver(prob, falg; abstol=1e-1)
            anskw = @inferred(dens(; Σ, β, μ))
            # 2. frequency integral inside
            f = ElectronDensityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, hv; abstol=1e-2)
            prob = IntegralProblem(f, bz)
            kcof = IntegralSolver(prob, alg; abstol=1e-1)
            answk = @inferred(dens(; Σ, β, μ))
            return anskw, answk
        end
        @test r[1][1] ≈ r[1][2] atol=1e-1
        @test r[1][1] ≈ r[2][1] atol=1e-1
        @test r[1][1] ≈ r[2][2] atol=1e-1
    end
    @testset "AuxTransportDistributionIntegrand" begin
        hv = GradientVelocityInterp(h, I, gauge=Hamiltonian())
        Σ = EtaSelfEnergy(1.0)
        ω₁ = 1.0; ω₂ = 2.0; μ = 0.1
        f = AuxTransportDistributionIntegrand(hv)
        r = map((fbz, ibz)) do bz
            prob = IntegralProblem(f, bz)
            tdis = IntegralSolver(prob, alg; abstol=1e-1)
            return @inferred(tdis(; Σ, ω₁, ω₂, μ))
        end
        @test r[1] ≈ r[2] atol=1e-1
    end
    @testset "AuxKineticCoefficientIntegrand" for Ω in (0.0, 2.0)
        hv = GradientVelocityInterp(h, I, gauge=Hamiltonian())
        Σ = EtaSelfEnergy(1.0)
        β = 1.0; μ = 0.1; n = 0
        falg = QuadGKJL()
        r = map((fbz, ibz)) do bz
            # test both orders of integration
            # 1. bz integral inside
            f = AuxKineticCoefficientIntegrand(bz, alg, hv; abstol=1e-2)
            prob = IntegralProblem(f, AutoBZ.lb(Σ), AutoBZ.ub(Σ))
            kcof = IntegralSolver(prob, falg; abstol=1e-1)
            anskw = @inferred(kcof(; Σ, n, β, Ω, μ))
            # 2. frequency integral inside
            f = AuxKineticCoefficientIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, hv; abstol=1e-2)
            prob = IntegralProblem(f, bz)
            kcof = IntegralSolver(prob, alg; abstol=1e-1)
            answk = @inferred(kcof(; Σ, n, β, Ω, μ))
            return anskw, answk
        end
        @test r[1][1] ≈ r[1][2] atol=1e-1
        @test r[1][1] ≈ r[2][1] atol=1e-1
        @test r[1][1] ≈ r[2][2] atol=1e-1
    end
end
