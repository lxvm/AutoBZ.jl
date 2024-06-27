abstract type LinearSystemAlgorithm end

struct LinearSystemProblem{A,Tl,Tr,K}
    A::A
    Pl::Tl
    Pr::Tr
    kwargs::K
end


"""
    LinearSystemProblem(A, [Pl=identity, Pr=identity]; abstol, reltol)

Constructor for a linear system of equations, represented by the matrix operator `A`, whose
solution is the matrix inverse. The solution does not need to be the inverse, but instead a
representation that can be used to solve ``Ax=b``. The solution, `sol.value`, should
implement `x = sol.value\\b`, `ldiv!(x, sol.value, b)`, and `inv`.
"""
LinearSystemProblem(A, Pl=identity, Pr=identity; kws...) = LinearSystemProblem(A, Pl, Pr, NamedTuple(kws))

mutable struct LinearSystemSolver{A,Tl,Tr,AA,C,K}
    A::A
    Pl::Tl
    Pr::Tr
    alg::AA
    cacheval::C
    kwargs::K
end

struct LinearSystemSolution{X,S}
    value::X
    retcode::ReturnCode
    stats::S
end

function init(prob::LinearSystemProblem, alg::LinearSystemAlgorithm; kws...)

    (; A, Pl, Pr, kwargs) = prob

    cacheval = init_linear_cacheval(A, Pl, Pr, alg)
    return LinearSystemSolver(A, Pl, Pr, alg, cacheval, merge(kwargs, NamedTuple(kws)))
end
function solve!(solver::LinearSystemSolver)
    do_linear_solve!(solver.A, solver.Pl, solver.Pr, solver.alg, solver.cacheval; solver.kwargs...)
end


"""
    LUFactorization()

Construct the LU factorization of the matrix so that the inverse can be efficiently
calculated for a small number of right-hand sides. Intended for mutable matrices
"""
struct LUFactorization <: LinearSystemAlgorithm end

function init_linear_cacheval(A, Pl, Pr, alg::LUFactorization)
    return ismutable(A) ? (LUWs(A), similar(A)) : nothing
end
function do_linear_solve!(A, Pl, Pr, alg::LUFactorization, ws; kws...)
    value = ismutable(A) ? LU(LAPACK.getrf!(ws[1], copy!(ws[2], A))...) : lu(A)
    retcode = LinearAlgebra.issuccess(value) ? Success : Failure
    stats = (;)
    return LinearSystemSolution(value, retcode, stats)
end

"""
    JLInv()

Solve a linear system by explicitly constructing the inverse. This should only be used for
very small matrices, such as 3x3, or when the full matrix is desired. Intended for immutable matrices.
"""
struct JLInv <: LinearSystemAlgorithm end

function init_linear_cacheval(A, Pl, Pr, alg::JLInv)
    # TODO allocate workspace for LAPACK.getri!
    return ismutable(A) ? init_linear_cacheval(A, Pl, Pr, LUFactorization()) : nothing
end
function do_linear_solve!(A, Pl, Pr, alg::JLInv, cacheval; kws...)
    invA = if ismutable(A)
        sol = do_linear_solve!(A, Pl, Pr, LUFactorization(), cacheval; kws...)
        LAPACK.getri!(sol.value.factors, sol.value.ipiv)
    else
        _inv(A)
    end
    value = JLInvMatrix(invA)
    retcode = Success
    stats = (;)
    return LinearSystemSolution(value, retcode, stats)
end

struct JLInvMatrix{A}
    A::A
end
Base.inv(A::JLInvMatrix) = A.A
Base.:\(A::JLInvMatrix, b) = A.A * b
LinearAlgebra.ldiv!(x, A::JLInvMatrix, b) = mul!(x, A.A, b)
