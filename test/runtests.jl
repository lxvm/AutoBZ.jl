using Test
using LinearAlgebra

using StaticArrays

using AutoBZ

@testset "AutoBZ" begin
    C = zeros(3,3,3)
    C[1,2,2] = C[3,2,2] = C[2,1,2] = C[2,3,2] = C[2,2,1] = C[2,2,3] = 0.5
    H = Jobs.FourierSeries3D(C)
    HV = Jobs.BandEnergyVelocity3D(C)

    @testset "linalg" begin
    
    end

    @testset "fermi" begin
        
    end

    @testset 
end
