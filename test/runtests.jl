using Test
using LinearAlgebra

using StaticArrays

using OffsetArrays
using AutoBZ
using AutoBZ: Freq2RadSeries

using Aqua

function integer_lattice(n)
    C = OffsetArray(zeros(SMatrix{1,1,Float64,1},ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k ≈ i ? j : 0, n))] = [1/2n;;]
    end
    return Freq2RadSeries(FourierSeries(C, period=2pi))
end

# run these tests with multiple threads to check multithreading works
@testset "AutoBZ" begin
    @testset "aqua" Aqua.test_all(AutoBZ)
    @testset "utils" include("utils.jl")
    @testset "linearsystem" include("linearsystem.jl")
    @testset "eigen" include("eigen.jl")
    @testset "trinv" include("trinv.jl")
    @testset "apps" include("apps.jl")
    # TODO: validate linalg, soc, interpolation, fermi functions, self energies, io
end

# @testset "BrillouinPlotlyExt" begin
# end
