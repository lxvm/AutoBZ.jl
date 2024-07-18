using Test
using LinearAlgebra
using StaticArrays
using AutoBZ

for alg in (LUFactorization(), JLInv())
    for A in [
        rand(40, 40),
        rand(ComplexF64, 40, 40),
        rand(SMatrix{3,3,Float64,9}),
        rand(SMatrix{3,3,ComplexF64,9}),
        rand(SHermitianCompact{3,Float64,6}),
        rand(SHermitianCompact{3,ComplexF64,6}),
    ]
        prob = LinearSystemProblem(A)
        solver = init(prob, alg)
        sol = solve!(solver)
        ismutable(A) && @test inv(sol.value) ≈ inv(A)
        vecb = zeros(eltype(A), size(A, 1))
        vecx = zeros(eltype(A), size(A, 2))
        for i in 1:size(A, 1)
            vecb[i] = 1
            vecs = ismutable(A) ? ldiv!(vecx, sol.value, vecb) : sol.value \ vecb
            @test vecs ≈ A\vecb
            vecb[i] = 0
        end
    end
end
