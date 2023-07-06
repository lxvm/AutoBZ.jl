using Test
using LinearAlgebra

using StaticArrays

using OffsetArrays
using SymmetryReduceBZ
using AutoBZ

function integer_lattice(n)
    C = OffsetArray(zeros(SMatrix{1,1,Float64,1},ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k ≈ i ? j : 0, n))] = [1/2n;;]
    end
    FourierSeries(C, period=1)
end

# run these tests with multiple threads to check multithreading works
@testset "AutoBZ" begin

#=
    @testset "linalg" begin
        # diag_inv ≈ diag ∘ inv
        # tr_inv ≈ tr ∘ inv
        # commutator
        # herm
    end

    @testset "fermi" begin

    end

    @testset "interp begin
        @testset "HamiltonianInterp" begin

        end

        @testset "GradientVelocityInterp" begin
            # check that it evaluates to the same as 4 independent fourier series
        end

        @testset "CovariantVelocityInterp" begin
            # check that it evaluates to the same as 4 independent fourier series
            # check for gauge covariance (eigenvalues of velocities the same in both gauges)
        end
    end
=#
    @testset "apps" for dims in (1, 2), alg in (IAI(), AutoPTR(), IAI(parallels=Parallel(dims)), AutoPTR(parallel=true))
        h = HamiltonianInterp(integer_lattice(dims))
        lims = TetrahedralLimits(fill(0.5, dims))
        syms = vec(collect(AutoBZ.cube_automorphisms(Val{dims}())))
        ibz = SymmetricBZ(2pi*I(dims), I(dims), lims, syms)
        fbz = FullBZ(2pi*I(dims))
        @testset "Green's function integrands" for integrand in (GlocIntegrand, DiagGlocIntegrand, TrGlocIntegrand, DOSIntegrand)
            h = h
            Σ = EtaSelfEnergy(1.0)
            f = integrand(h, Σ)
            r = map((fbz, ibz)) do bz
                prob = IntegralProblem(f, bz)
                gint = IntegralSolver(prob, alg; abstol=1e-1)
                # compare the various interfaces
                ω = 1.0; μ = 0.1
                ans = @inferred(gint(ω, μ))
                @test ans ≈ gint(ω, μ=μ) ≈ gint(ω=ω, μ=μ)
                return ans
            end
            # the Green's function integrands have unknown symmetry
            # representations but we are only integrating a scalar Hamiltonian
            # wrapped as a matrix
            if eltype(r) <: AutoBZCore.TrivialRepType
                @test r[1] ≈ r[2] atol=1e-1
            else
                @test only(r[1]) ≈ AutoBZCore.symmetrize(f, ibz, only(r[2])) atol=1e-1
            end
        end
        @testset "TransportFunctionIntegrand" begin
            hv = GradientVelocityInterp(h, I, gauge=Hamiltonian())
            f = TransportFunctionIntegrand(hv)
            r = map((fbz, ibz)) do bz
                prob = IntegralProblem(f, bz)
                tfun = IntegralSolver(prob, alg; abstol=1e-1)
                # compare the various interfaces
                β = 1.0; μ = 0.1
                ans = @inferred(tfun(β, μ))
                @test ans ≈ tfun(β, μ=μ) ≈ tfun(β=β, μ=μ)
                return ans
            end
            @test r[1] ≈ r[2] atol=1e-1
        end
        @testset "TransportDistributionIntegrand" begin
            hv = GradientVelocityInterp(h, I, gauge=Hamiltonian())
            Σ = EtaSelfEnergy(1.0)
            f = TransportDistributionIntegrand(hv, Σ)
            r = map((fbz, ibz)) do bz
                prob = IntegralProblem(f, bz)
                tdis = IntegralSolver(prob, alg; abstol=1e-1)
                # compare the various interfaces
                ω₁ = 1.0; ω₂ = 2.0; μ = 0.1
                ans = @inferred(tdis(ω₁, ω₂, μ))
                @test ans ≈ tdis(ω₁, ω₂, μ=μ) ≈ tdis(ω₁=ω₁, ω₂=ω₂, μ=μ)
                return ans
            end
            @test r[1] ≈ r[2] atol=1e-1
        end
        @testset "KineticCoefficientIntegrand" for Ω in (0.0, 2.0)
            β = 1.0; μ = 0.1
            hv = GradientVelocityInterp(h, I, gauge=Hamiltonian())
            Σ = EtaSelfEnergy(1.0)
            falg = QuadGKJL()
            r = map((fbz, ibz)) do bz
                # test both orders of integration
                # 1. bz integral inside
                f = KineticCoefficientIntegrand(bz, alg, hv, Σ, 0; abstol=1e-2)
                prob = IntegralProblem(f, AutoBZ.lb(Σ), AutoBZ.ub(Σ))
                kcof = IntegralSolver(prob, falg; abstol=1e-1)
                anskw = @inferred(kcof(β, Ω, μ))
                # compare the various interfaces
                @test anskw ≈ kcof(β, Ω, μ=μ) ≈ kcof(β, Ω=Ω, μ=μ) ≈ kcof(β=β, Ω=Ω, μ=μ)
                # 2. frequency integral inside
                f = KineticCoefficientIntegrand(falg, hv, Σ, 0; abstol=1e-2)
                prob = IntegralProblem(f, bz)
                kcof = IntegralSolver(prob, alg; abstol=1e-1)
                answk = @inferred(kcof(β, Ω, μ))
                # compare the various interfaces
                @test answk ≈ kcof(β, Ω, μ=μ) ≈ kcof(β, Ω=Ω, μ=μ) ≈ kcof(β=β, Ω=Ω, μ=μ)
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
                f = ElectronDensityIntegrand(bz, alg, hv, Σ; abstol=1e-2)
                prob = IntegralProblem(f, AutoBZ.lb(Σ), AutoBZ.ub(Σ))
                dens = IntegralSolver(prob, falg; abstol=1e-1)
                anskw = @inferred(dens(β, μ))
                # compare the various interfaces
                @test anskw ≈ dens(β, μ=μ) ≈ dens(β=β, μ=μ)
                # 2. frequency integral inside
                f = ElectronDensityIntegrand(falg, hv, Σ; abstol=1e-2)
                prob = IntegralProblem(f, bz)
                kcof = IntegralSolver(prob, alg; abstol=1e-1)
                answk = @inferred(dens(β, μ))
                # compare the various interfaces
                @test answk ≈ dens(β, μ=μ) ≈ dens(β=β, μ=μ)
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
            f = AuxTransportDistributionIntegrand(hv, Σ)
            r = map((fbz, ibz)) do bz
                prob = IntegralProblem(f, bz)
                tdis = IntegralSolver(prob, alg; abstol=1e-1)
                ans = @inferred(tdis(ω₁, ω₂, μ))
                # compare the various interfaces
                @test ans ≈ tdis(ω₁, ω₂, μ=μ) ≈ tdis(ω₁=ω₁, ω₂=ω₂, μ=μ)
                return ans
            end
            @test r[1] ≈ r[2] atol=1e-1
        end
        @testset "AuxKineticCoefficientIntegrand" for Ω in (0.0, 2.0)
            hv = GradientVelocityInterp(h, I, gauge=Hamiltonian())
            Σ = EtaSelfEnergy(1.0)
            β = 1.0; μ = 0.1
            falg = QuadGKJL()
            r = map((fbz, ibz)) do bz
                # test both orders of integration
                # 1. bz integral inside
                f = AuxKineticCoefficientIntegrand(bz, alg, hv, Σ, 0; abstol=1e-2)
                prob = IntegralProblem(f, AutoBZ.lb(Σ), AutoBZ.ub(Σ))
                kcof = IntegralSolver(prob, falg; abstol=1e-1)
                anskw = @inferred(kcof(β, Ω, μ))
                # compare the various interfaces
                @test anskw ≈ kcof(β, Ω, μ=μ) ≈ kcof(β, Ω=Ω, μ=μ) ≈ kcof(β=β, Ω=Ω, μ=μ)
                # 2. frequency integral inside
                f = AuxKineticCoefficientIntegrand(falg, hv, Σ, 0; abstol=1e-2)
                prob = IntegralProblem(f, bz)
                kcof = IntegralSolver(prob, alg; abstol=1e-1)
                answk = @inferred(kcof(β, Ω, μ))
                # compare the various interfaces
                @test answk ≈ kcof(β, Ω, μ=μ) ≈ kcof(β, Ω=Ω, μ=μ) ≈ kcof(β=β, Ω=Ω, μ=μ)
                return anskw, answk
            end
            @test r[1][1] ≈ r[1][2] atol=1e-1
            @test r[1][1] ≈ r[2][1] atol=1e-1
            @test r[1][1] ≈ r[2][2] atol=1e-1
        end
    end
end

# @testset "SymmetryReduceBZExt" begin
#     include("test_ibz.jl")
# end

# @testset "BrillouinPlotlyExt" begin
# end
