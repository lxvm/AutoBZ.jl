using Test
using LinearAlgebra

using StaticArrays

using OffsetArrays
using SymmetryReduceBZ
using AutoBZ

function integer_lattice(n)
    C = OffsetArray(zeros(ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = 1/2n
    end
    FourierSeries(C, period=1)
end

@testset "AutoBZ" begin

#=
    @testset "linalg" begin
        # diag_inv == diag ∘ inv
        # tr_inv == tr ∘ inv
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
    @testset "apps" begin
        @testset "DOSIntegrand" begin
            dims = 1
            h = HamiltonianInterp(integer_lattice(dims))
            Σ = EtaSelfEnergy(1.0)
            f = DOSIntegrand(h, Σ)
            bz= FullBZ(2pi*I(dims))
            prob = IntegralProblem(f, bz)
            dos = IntegralSolver(prob, AuxIAI())
            # compare the various interfaces
            ω = 1.0; μ = 0.1
            @test @inferred(Float64, dos(ω, μ)) == dos(ω, μ=μ) == dos(ω=ω, μ=μ)
        end
        @testset "TransportDistributionIntegrand" begin
            
        end
        @testset "KineticCoefficientIntegrand" begin
            
        end
        @testset "ElectronDensityIntegrand" begin
            
        end
        @testset "AuxTransportDistributionIntegrand" begin
            
        end
        @testset "AuxKineticCoefficientIntegrand" begin
            
        end
    end
end

@testset "SymmetryReduceBZExt" begin
    include("test_ibz.jl")
end

@testset "BrillouinPlotlyExt" begin
end
