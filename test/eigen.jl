using Test
using LinearAlgebra
using StaticArrays
using AutoBZ

function test_vectors(A, values, vectors; kws...)
    for (val, vec) in zip(values, eachcol(vectors))
        isapprox(val*vec, A*vec; kws...) || return false
    end
    return true
end

@testset "eigen - general" for alg in (LAPACKEigen(), JLEigen())
    for A in [rand(40, 40), rand(ComplexF64, 40, 40)]
        prob = EigenProblem(A, false)
        solver = init(prob, alg)
        sol = solve!(solver)
        @test sol.value isa AbstractVector
        @test sol.value ≈ eigvals(A)
        prob = EigenProblem(A)
        solver = init(prob, alg)
        sol = solve!(solver)
        @test sol.value isa Eigen
        eig = eigen(A)
        @test sol.value.values ≈ eig.values
        @test test_vectors(A, sol.value.values, eig.vectors)
        @test test_vectors(A, eig.values, sol.value.vectors)
    end
end


@testset "eigen - hermitian" for alg in (LAPACKEigenH(), JLEigen())
    for A in [
        Symmetric(rand(40, 40)),
        Hermitian(rand(ComplexF64, 40, 40)),
        begin x = rand(ComplexF64, 40, 40); (x+x')/2; end,
        begin x = rand(SHermitianCompact{3,Float64,6}); (x+x')/2; end,
        begin x = rand(SHermitianCompact{3,ComplexF64,6}); (x+x')/2; end,
    ]
        prob = EigenProblem(A, false)
        solver = init(prob, alg)
        sol = solve!(solver)
        @test sol.value isa AbstractVector
        @test sol.value ≈ eigvals(A)
        prob = EigenProblem(A)
        solver = init(prob, alg)
        sol = solve!(solver)
        @test sol.value isa Eigen
        eig = eigen(A)
        @test sol.value.values ≈ eig.values
        @test test_vectors(A, sol.value.values, eig.vectors)
        @test test_vectors(A, eig.values, sol.value.vectors)
    end
end
