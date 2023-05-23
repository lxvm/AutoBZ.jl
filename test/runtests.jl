using Test
using LinearAlgebra

using StaticArrays

using SymmetryReduceBZ
using AutoBZ

include("test_ibz.jl")

@testset "AutoBZ" begin
    # 3D integer lattice Hamiltonian
    C = zeros(3,3,3)
    C[1,2,2] = C[3,2,2] = C[2,1,2] = C[2,3,2] = C[2,2,1] = C[2,2,3] = 0.5


    @testset "linalg" begin
        # diag_inv == diag ∘ inv
        # tr_inv == tr ∘ inv
        # commutator
        # herm
    end

    @testset "fermi" begin

    end

    @testset "Hamiltonian" begin

    end

    @testset "HamiltonianVelocity" begin
        # check that it evaluates to the same as 4 independent fourier series
    end

    @testset "CovariantHamiltonianVelocity" begin
        # check that it evaluates to the same as 4 independent fourier series
        # check for gauge covariance (eigenvalues of velocities the same in both gauges)
    end
end
