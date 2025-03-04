using Test
using LinearAlgebra

using StaticArrays

using OffsetArrays
using AutoBZ

using Aqua

function integer_lattice(n)
    C = OffsetArray(zeros(SMatrix{1,1,Float64,1},ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k ≈ i ? j : 0, n))] = [1/2n;;]
    end
    return C
end
function graphene()
    H = OffsetArray(zeros(SMatrix{2,2,ComplexF64,4},3,3),-1:1,-1:1)
    H[0,0] = [0 -1; -1 0]
    H[0,-1] = H[-1,0] = [0 -1; 0 0]
    H[0,1]  = H[1,0]  = [0 0; -1 0]
    return H
end
function coeffs2FourierHamiltonian(C; gauge=Wannier(), eigalg=JLEigen())
    prob = gauge isa Wannier ? nothing : EigenProblem(zero(eltype(C)))
    alg = gauge isa Wannier ? nothing : eigalg
    return HamiltonianInterp(AutoBZ.Freq2RadSeries(FourierSeries(C; period=2pi)), prob, alg; gauge)
end
function coeffs2HermitianHamiltonian(C; gauge=Wannier(), eigalg=JLEigen())
    prob = gauge isa Wannier ? nothing : EigenProblem(zero(eltype(C)))
    alg = gauge isa Wannier ? nothing : eigalg
    return HamiltonianInterp(AutoBZ.Freq2RadSeries(HermitianFourierSeries(FourierSeries(C; period=2pi))), prob, alg; gauge)
end
#=
function coeffs2RealSymmetricHamiltonian(C; gauge=Wannier(), eigalg=JLEigen())
    prob = gauge isa Wannier ? nothing : EigenProblem(zero(eltype(C)))
    alg = gauge isa Wannier ? nothing : eigalg
    return HamiltonianInterp(AutoBZ.Freq2RadSeries(RealSymmetricFourierSeries(FourierSeries(C; period=2pi))), prob, alg; gauge)
end
=#

# run these tests with multiple threads to check multithreading works
@testset "AutoBZ" begin
    @testset "aqua" Aqua.test_all(AutoBZ)
    @testset "utils" include("utils.jl")
    @testset "linearsystem" include("linearsystem.jl")
    @testset "eigen" include("eigen.jl")
    @testset "trinv" include("trinv.jl")
    @testset "GlocSolver,TrGlocSolver,DOSSolver" include("GreensSolver.jl")
    @testset "TransportFunctionSolver" include("TransportFunctionSolver.jl")
    @testset "TransportDistributionSolver" include("TransportDistributionSolver.jl")
    @testset "AuxTransportDistributionSolver" include("AuxTransportDistributionSolver.jl")
    @testset "ElectronDensitySolver" include("ElectronDensitySolver.jl")
    @testset "KineticCoefficientSolver" include("KineticCoefficientSolver.jl")
    @testset "AuxKineticCoefficientSolver" include("AuxKineticCoefficientSolver.jl")
    # TODO: validate linalg, soc, interpolation, fermi functions, io
    @testset "AutoBZLehmannExt" include("lehmann.jl")
    @testset "selfenergies" include("selfenergies.jl")
end

# @testset "BrillouinPlotlyExt" begin
# end
