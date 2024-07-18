abstract type TraceInverseAlgorithm end

struct TraceInverseProblem{A,K}
    A::A
    kwargs::K
end

TraceInverseProblem(A; kws...) = TraceInverseProblem(A, kws)

mutable struct TraceInverseSolver{A,K,G,C}
    A::A
    kwargs::K
    alg::G
    cacheval::C
end

struct TraceInverseSolution{V,S}
    value::V
    retcode::ReturnCode
    stats::S
end

function init(prob::TraceInverseProblem, alg::TraceInverseAlgorithm; kws...)
    cacheval = init_cacheval(prob, alg)
    kwargs = (; kws..., prob.kwargs...)
    return TraceInverseSolver(prob.A, kwargs, alg, cacheval)
end

function solve!(solver::TraceInverseSolver)
    return do_trinv(solver.A, solver.alg, solver.cacheval; solver.kwargs...)
end

struct JLTrInv <: TraceInverseAlgorithm end

init_cacheval(::TraceInverseProblem, ::JLTrInv) = nothing
function do_trinv(A, ::JLTrInv, cacheval; kws...)
    value = tr_inv(A)
    retcode = Success
    stats = (;)
    return TraceInverseSolution(value, retcode, stats)
end

struct LinearSystemTrInv{A<:LinearSystemAlgorithm} <: TraceInverseAlgorithm
    alg::A
end

function init_cacheval(prob::TraceInverseProblem, alg::LinearSystemTrInv)
    return init(LinearSystemProblem(prob.A), alg.alg)
end
function do_trinv(A, ::LinearSystemTrInv, solver; kws...)
    solver.A = A
    sol = solve!(solver)
    return TraceInverseSolution(tr(inv(sol.value)), sol.retcode, sol.stats)
end

struct EigenTrInv{A<:EigenAlgorithm} <: TraceInverseAlgorithm
    alg::A
end
function init_cacheval(prob::TraceInverseProblem, alg::EigenTrInv)
    return init(EigenProblem(prob.A, false), alg.alg)
end
function do_trinv(A, ::EigenTrInv, solver; kws...)
    solver.A = A
    sol = solve!(solver)
    return TraceInverseSolution(sum(inv, sol.value), sol.retcode, sol.stats)
end

# TODO another algorithm for large matrices is QR