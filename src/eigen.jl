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

struct EigenProblem{vecs,A,S,K}
    vecs::Val{vecs}
    A::A
    sortby::S
    kwargs::K
end

"""
    EigenProblem(A::AbstractMatrix, [vecs::Bool=true, sortby]; kws...)

Define an eigenproblem for the matrix `A` with the option `vecs` to return
the spectrum with the eigenvectors (`true` by default) or only the spectrum
(`false`). A comparison function `sortby(λ)` can be provided to sort the
eigenvalues, which by default are sorted lexicographically. `sortby=nothing`
will leave the eigenvalues in an arbitrary order.
Additonal keywords are forwarded to the solver.

Aims to provide a non-allocating interface to `LinearAlgebra.eigen`.

When `vecs` is `true`, the value of the solution will be a `LinearAlgebra.Eigen`
factorization object. When `vecs` is false, the solution will be a vector
containing the eigenvalues. The function [`hasvecs`](@ref) will return `vecs`.
"""
function EigenProblem(A::AbstractMatrix, vecs::Bool=true, sortby=eigsortby; kws...)
    return EigenProblem(Val(vecs), A, sortby, kws)
end

"""
    hasvecs(::EigenProblem{vecs}) where {vecs} = vecs

Test whether the eigenproblem also needs the eigenvectors
"""
hasvecs(::EigenProblem{vecs}) where {vecs} = vecs

mutable struct EigenSolver{vecs,A,S,K,G,C}
    vecs::Val{vecs}
    A::A
    sortby::S
    kwargs::K
    alg::G
    cacheval::C
end

"""
    hasvecs(::EigenSolver{vecs}) where {vecs} = vecs

Test whether the eigensolver also calculates the eigenvectors
"""
hasvecs(::EigenSolver{vecs}) where {vecs} = vecs

struct EigenSolution{vecs,V,S}
    vecs::Val{vecs}
    value::V
    retcode::ReturnCode
    stats::S
end

"""
    hasvecs(sol::EigenSolution{vecs}) where {vecs} = vecs

Test whether the eigensolution contains the eigenvectors.
If `true`, `sol.value` will be a `LinearAlgebra.Eigen` and otherwise a vector
containing the spectrum.
"""
hasvecs(::EigenSolution{vecs}) where {vecs} = vecs

function init(prob::EigenProblem, alg::EigenAlgorithm; kws...)
    cacheval = init_cacheval(prob, alg)
    kwargs = (; kws..., prob.kwargs...)
    return EigenSolver(prob.vecs, prob.A, prob.sortby, kwargs, alg, cacheval)
end

function solve!(solver::EigenSolver)
    return do_eigen(solver.vecs, solver.A, solver.sortby, solver.alg, solver.cacheval)
end

struct LAPACKEigen <: EigenAlgorithm
    balanc::Char
    sense::Char
end
function LAPACKEigen(; balanc = 'B', sense='N')
    return LAPACKEigen(balanc, sense)
end

function init_cacheval(prob::EigenProblem, alg::LAPACKEigen)
    A = prob.A
    ishermitian(A) && @warn "Hermitian matrix detected. Consider using LAPACKEigenH"
    Atmp = Matrix{typeof(complex(one(eltype(A))))}(undef, size(A)...)
    copy!(Atmp, A)
    ws = EigenWs(Atmp; lvecs = false, rvecs = hasvecs(prob), sense = alg.sense != 'N')
    perm = Vector{Int}(undef, checksquare(A))
    return complex(A[:,begin]), complex(_ustrip(A)), Atmp, Atmp[:,1], ws, perm
end

function do_eigen(v::Val{vecs}, A, sortby::S, alg::LAPACKEigen, (Avec, Amat, Atmp, tmp, ws, perm)) where {vecs,S}
    _ucopy!(Atmp, A)
    t = LAPACK.geevx!(ws, alg.balanc, 'N', vecs ? 'V' : 'N', alg.sense, Atmp)
    values = t[2]
    vectors = t[eltype(Atmp) isa Real ? 5 : 4]
    if sortby !== nothing
        if vecs
            sorteig!(perm, tmp, values, vectors, sortby)
        else
            sorteig!(values, sortby)
        end
    end
    E = vecs ? LinearAlgebra.Eigen(_ofutype(Avec, values), _oftype(Amat, vectors)) : values
    retcode = Success
    stats = (;) # TODO populate stats
    return EigenSolution(v, E, retcode, stats)
end

struct LAPACKEigenH{T} <: EigenAlgorithm
    range::Char
    uplo::Char
    vl::T
    vu::T
    il::Int
    iu::Int
    work::Bool
end
function LAPACKEigenH(; range='A', uplo='U', vl=0, vu=0, il=0, iu=0, work=true)
    return LAPACKEigenH(range, uplo, vl, vu, il, iu, work)
end

function init_cacheval(prob::EigenProblem, alg::LAPACKEigenH)
    A = prob.A
    ishermitian(A) || @warn "Non-hermitian matrix detected. Results may be incorrect"
    Atmp = Matrix{typeof(one(eltype(A)))}(undef, size(A)...)
    ws = HermitianEigenWs(Atmp; vecs = hasvecs(prob), work = alg.work)
    perm = Vector{Int}(undef, checksquare(A))
    return real(parent(A)[:,begin]), _ustrip(parent(A)), Atmp, Atmp[:,1], ws, perm
end

function do_eigen(v::Val{vecs}, A, sortby::S, alg::LAPACKEigenH, (Avec, Amat, Atmp, tmp, ws, perm); abstol=-1.0) where {vecs,S}
    _ucopy!(Atmp, A)
    a = real(zero(eltype(Atmp)))
    t = LAPACK.syevr!(ws, vecs ? 'V' : 'N', alg.range, alg.uplo, Atmp, oftype(a, alg.vl), oftype(a, alg.vu), alg.il, alg.iu, oftype(a, abstol))
    values, vectors = t
    if sortby !== nothing
        if vecs
            sorteig!(perm, tmp, values, vectors, sortby)
        else
            sorteig!(values, sortby)
        end
    end
    E = vecs ? LinearAlgebra.Eigen(_ofutype(Avec, values), _oftype(Amat, vectors)) : values
    retcode = Success
    stats = (;) # TODO populate stats
    return EigenSolution(v, E, retcode, stats)
end

struct JLEigen <: EigenAlgorithm
    permute::Bool
    scale::Bool
end
JLEigen(; permute=true, scale=true) = JLEigen(permute, scale)

init_cacheval(::EigenProblem, alg::JLEigen) = nothing
function do_eigen(v::Val{vecs}, A, sortby::S, alg::JLEigen, cacheval; kws...) where {vecs,S}
    E = vecs ? _eigen(A; permute=alg.permute, scale=alg.scale, sortby) : _eigvals(A; permute=alg.permute, scale=alg.scale, sortby)
    retcode = Success
    stats = (;) # TODO populate stats
    return EigenSolution(v, E, retcode, stats)
end

_eigen(args...; kws...) = eigen(args...; kws...)
_eigen(A::Union{<:Hermitian,<:Symmetric}, args...; sortby, kws...) = eigen(A, args...; sortby)
_eigen(A::StaticArray, args...; permute, scale, kws...) = eigen(A, args...; permute, scale)
_eigvals(args...; kws...) = eigvals(args...; kws...)
_eigvals(A::Union{<:Hermitian,<:Symmetric}, args...; sortby, kws...) = eigvals(A, args...; sortby)
_eigvals(A::StaticArray, args...; permute, scale, kws...) = eigvals(A, args...; permute, scale)