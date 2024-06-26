using Test
using LinearAlgebra
using StaticArrays
using AutoBZ

for alg in (JLTrInv(), LinearSystemTrInv(LUFactorization()), LinearSystemTrInv(JLInverse()), EigenTrInv(JLEigen()))
    for A in [
        rand(40, 40),
        rand(ComplexF64, 40, 40),
        rand(SMatrix{3,3,Float64,9}),
        rand(SMatrix{3,3,ComplexF64,9}),
        rand(SHermitianCompact{3,Float64,6}),
        rand(SHermitianCompact{3,ComplexF64,6}),
    ]
        A isa StaticArray && alg isa LinearSystemTrInv{LUFactorization} && continue
        A isa StaticArray && alg isa EigenTrInv && continue
        prob = TraceInverseProblem(A)
        solver = init(prob, alg)
        sol = solve!(solver)
        @test sol.value ≈ tr(inv(A))
    end
end