using AutoBZ
using Test
using StaticArrays

@testset "AutoBZ.jl" begin
    @testset "IAI validation" begin
        n = 5
        @test iterated_integration(x -> x .^ (1:n), zeros(SVector{n}), ones(SVector{n}))[1] ≈ [1/(i+1) for i in 1:n]
        # test integration on more complicated domains
        # check that IteratedIntegrand and other are working
    end

    @testset "PTR validation" begin
        # check for spectral convergence on FBZ and IBZ
    end

    @testset "Fourier series" begin
        # Check Fourier series and derivatives match a reference
    end

    @testset "Integrator" begin
        # Check that the interface works
    end
end

@testset "AutoBZ.Jobs" begin
    # check that the integrators work and that routines dont fail
end
