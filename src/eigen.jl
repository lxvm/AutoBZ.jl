abstract type EigenAlgorithm end

eigsortby(λ::Real) = λ
eigsortby(λ::Complex) = (real(λ),imag(λ))
function sorteig!(p, tmp, λ::AbstractVector, X::AbstractMatrix, sortby::Union{Function,Nothing}=eigsortby)
    if sortby !== nothing && !issorted(λ, by=sortby)
        sortperm!(p, λ; alg=QuickSort, by=sortby)
        copy!(λ, tmp .= getindex.(Ref(λ), p))
        Base.permutecols!!(X, p)
    end
    return λ, X
end
sorteig!(λ::AbstractVector, sortby::Union{Function,Nothing}=eigsortby) = sortby === nothing ? λ : sort!(λ, by=sortby)

struct EigenProblem{A,S,K}
    A::A
    sortby::S
    kwargs::K
end
EigenProblem(A::AbstractMatrix, sortby=eigsortby; kws...) = EigenProblem(A, sortby, kws)

mutable struct EigenSolver{A,S,K,G,C}
    A::A
    sortby::S
    kwargs::K
    alg::G
    cacheval::C
end

struct EigenSolution{V,S}
    value::V
    retcode::ReturnCode
    stats::S
end

function init(prob::EigenProblem, alg::EigenAlgorithm; kws...)
    cacheval = init_cacheval(prob.A, alg)
    kwargs = (; kws..., prob.kwargs...)
    return EigenSolver(prob.A, prob.sortby, kwargs, alg, cacheval)
end

function solve!(solver::EigenSolver)
    return do_eigen(solver.A, solver.sortby, solver.alg, solver.cacheval)
end

struct LAPACKEigen <: EigenAlgorithm
    balanc::Char
    jobvl::Char
    jobvr::Char
    sense::Char
end
function LAPACKEigen(;
    permute=true, scale=true, balanc = permute ? (scale ? 'B' : 'P') : (scale ? 'S' : 'N'),
    vecs=true, jobvl=vecs ? 'V' : 'N', jobvr=vecs ? 'V' : 'N',
    sense='N')
    return LAPACKEigen(balanc, jobvl, jobvr, sense)
end

function init_cacheval(A, alg::LAPACKEigen)
    ishermitian(A) && @warn "Hermitian matrix detected. Consider using LAPACKEigenH"
    Atmp = Matrix{typeof(complex(one(eltype(A))))}(undef, size(A)...)
    copy!(Atmp, A)
    ws = EigenWs(Atmp; lvecs = alg.jobvl == 'V', rvecs = alg.jobvl == 'R', sense = alg.sense != 'N')
    perm = Vector{Int}(undef, checksquare(A))
    return A[:,begin], _ustrip(A), Atmp, Atmp[:,1], ws, perm
end

function do_eigen(A, sortby::S, alg::LAPACKEigen, (Avec, Amat, Atmp, tmp, ws, perm)) where S
    _ucopy!(Atmp, A)
    t = LAPACK.geevx!(ws, alg.balanc, alg.jobvl, alg.jobvr, alg.sense, Atmp)
    values = t[2]
    vectors = t[eltype(Atmp) isa Real ? 5 : 4]
    if sortby !== nothing
        sorteig!(perm, tmp, values, vectors, sortby)
    end
    E = LinearAlgebra.Eigen(_ofutype(Avec, values), _oftype(Amat, vectors))
    retcode = Success
    stats = (;) # TODO populate stats
    return EigenSolution(E, retcode, stats)
end

struct LAPACKEigenH{T} <: EigenAlgorithm
    jobz::Char
    range::Char
    uplo::Char
    vl::T
    vu::T
    il::Int
    iu::Int
    work::Bool
end
function LAPACKEigenH(;
    vecs=true, jobz=vecs ? 'V' : 'N',
    range='A', uplo='U',
    vl=0, vu=0, il=0, iu=0,
    work=true)
    return LAPACKEigenH(jobz, range, uplo, vl, vu, il, iu, work)
end

function init_cacheval(A, alg::LAPACKEigenH)
    ishermitian(A) || @warn "Non-hermitian matrix detected. Results may be incorrect"
    Atmp = Matrix{typeof(one(eltype(A)))}(undef, size(A)...)
    ws = HermitianEigenWs(Atmp; vecs = alg.jobz == 'V', work = alg.work)
    perm = Vector{Int}(undef, checksquare(A))
    return real(parent(A)[:,begin]), _ustrip(parent(A)), Atmp, Atmp[:,1], ws, perm
end

function do_eigen(A, sortby::S, alg::LAPACKEigenH, (Avec, Amat, Atmp, tmp, ws, perm); abstol=-1.0) where S
    _ucopy!(Atmp, A)
    a = real(zero(eltype(Atmp)))
    t = LAPACK.syevr!(ws, alg.jobz, alg.range, alg.uplo, Atmp, oftype(a, alg.vl), oftype(a, alg.vu), alg.il, alg.iu, oftype(a, abstol))
    values, vectors = t
    if sortby !== nothing
        sorteig!(perm, tmp, values, vectors, sortby)
    end
    E = LinearAlgebra.Eigen(_ofutype(Avec, values), _oftype(Amat, vectors))
    retcode = Success
    stats = (;) # TODO populate stats
    return EigenSolution(E, retcode, stats)
end

struct LAPACKEigvals <: EigenAlgorithm
    alg::LAPACKEigen
end
function LAPACKEigvals(; kws...)
    return LAPACKEigvals(LAPACKEigen(; kws..., jobvl = 'N', jobvr = 'N', sense = 'N'))
end
function init_cacheval(A, alg::LAPACKEigvals)
    return init_cacheval(A, alg.alg)
end
function do_eigen(A, sortby, alg::LAPACKEigvals, ws)
    sol = do_eigen(A, sortby, alg.alg, ws)
    return EigenSolution(sol.value.values, sol.retcode, sol.stats)
end

struct LAPACKEigvalsH{T} <: EigenAlgorithm
    alg::LAPACKEigenH{T}
end
function LAPACKEigvalsH(; kws...)
    return LAPACKEigvalsH(LAPACKEigenH(; kws..., jobz = 'N'))
end
function init_cacheval(A, alg::LAPACKEigvalsH)
    return init_cacheval(A, alg.alg)
end
function do_eigen(A, sortby, alg::LAPACKEigvalsH, ws)
    sol = do_eigen(A, sortby, alg.alg, ws)
    return EigenSolution(sol.value.values, sol.retcode, sol.stats)
end
