
"""
    SOC(A)
Wrapper for a matrix A that should be used as a block-diagonal matrix
    [A 0
     0 A]
as when dealing with spin-orbit coupling
"""
struct SOC{N,T,M} <: StaticMatrix{N,N,T}
    A::M
    SOC(A::M) where {N,T,M<:StaticMatrix{N,N,T}} = new{2*N,T,M}(A)
end

function _check_square_parameters(::Val{N}) where N
    d, r = divrem(N, 2)
    iszero(r) || throw(ArgumentError("Size mismatch in SOC. Got matrix linear dimension of $N"))
    d
end

@generated function SOC{N_,T}(data::Tuple) where {N_,T}
    N = _check_square_parameters(Val(N_))
    expr = Vector{Expr}(undef, N^2)
    m = 0
    for i in 0:N-1, j in 1:N
        idx = j + i*N_
        expr[m += 1] = :(data[$idx])
    end
    quote
        Base.@_inline_meta
        @inbounds return SOC(SMatrix{$N,$N,T,$(N^2)}(tuple($(expr...))))
    end
end

@generated function Base.Tuple(a::SOC{N_,T}) where {N_,T}
    z = zero(T)
    N = div(N_, 2)
    exprs = [ i == j ? :(a.A[$n,$m]) : :($z) for n in 1:N, i in 1:2, m in 1:N, j in 1:2 ]
    quote
        Base.@_inline_meta
        tuple($(exprs...))
    end
end

# TODO, we should be able to reduce the cost of matrix multiplication by half by
# avoiding multiplication of the empty off-diagonal blocks, however at least
# this performance matches dense matrix multiplication and we still benefit from
# reduced storage and the optimizations below

@generated function _soc_indices(::Val{N}) where N
    # return a Tuple{Pair{Int,Bool}} idx such that for linear index i
    # idx[i][1] is the linear index into the A field of SOC
    # idx[i][2] is true iff i is an index into a block diagonal of the SOC
    indexmat = Matrix{Pair{Int,Bool}}(undef, N, N)
    M = div(N, 2)
    for j in 0:1, m in 1:M, i in 0:1, n in 1:M
        idx = n + (m-1)*M
        indexmat[n + M*i, m + j*M] = if i == j
            idx => true
        else
            idx => false
        end
    end
    quote
        StaticArrays.@_inline_meta
        return $(tuple(indexmat...))
    end
end

Base.@propagate_inbounds function Base.getindex(a::SOC{N_,T}, i::Int) where {N_,T}
    idx = _soc_indices(Val(N_))
    j, isdiag = idx[i]
    isdiag ? @inbounds(a.A[j]) : zero(T)
end

@inline Base.:+(a::SOC{N}, b::SOC{N}) where N = SOC(a.A + b.A)
@inline Base.:-(a::SOC{N}, b::SOC{N}) where N = SOC(a.A - b.A)
@inline Base.:*(a::SOC{N}, b::SOC{N}) where N = SOC(a.A * b.A)

for op in (:*, :/)
    @eval @inline Base.$op(a::SOC, b::Number) = SOC($op(a.A, b))
    @eval @inline Base.$op(a::Number, b::SOC) = SOC($op(a, b.A))
end

for op in (:+, :-)
    @eval @inline Base.$op(a::SOC, b::UniformScaling) = SOC($op(a.A, b))
    @eval @inline Base.$op(a::UniformScaling, b::SOC) = SOC($op(a, b.A))
end

for op in (:zero, :one, :oneunit)
    @eval Base.$op(::Type{T}) where {T<:SOC} = SOC($op(fieldtype(T, :A)))
end

@inline LinearAlgebra.inv(a::SOC) = SOC(inv(a.A))
@inline LinearAlgebra.tr(a::SOC) = 2tr(a.A)


# based on eq. 3 of https://doi.org/10.1103/PhysRevB.98.205128
function t2g_coupling(λ::Number=1)
    CT = typeof(complex(λ))
    σˣ = SMatrix{2,2,CT,4}(( 0, 1, 1, 0))
    σʸ = SMatrix{2,2,CT,4}(( 0, 1,-1, 0))*im
    σᶻ = SMatrix{2,2,CT,4}(( 1, 0, 0,-1))
    ϵˣ = SMatrix{3,3,CT,9}(( 0, 0, 0, 0, 0,-1, 0, 1, 0))
    ϵʸ = SMatrix{3,3,CT,9}(( 0, 0, 1, 0, 0, 0,-1, 0, 0))
    ϵᶻ = SMatrix{3,3,CT,9}(( 0,-1, 0, 1, 0, 0, 0, 0, 0))
    return (kron(σˣ, ϵˣ) + kron(σʸ, ϵʸ) + kron(σᶻ, ϵᶻ)) * λ * im // 2
end

struct SOCHamiltonianInterp{G,N,T,F,L} <: AbstractHamiltonianInterp{G,N,T}
    f::F
    λ::L
    SOCHamiltonianInterp{G}(f::F, λ::L) where {G,F<:FourierSeries,L} =
        new{G,ndims(f),eltype(f),F,L}(f,λ)
end

function SOCHamiltonianInterp(f, λ::AbstractMatrix; gauge=GaugeDefault(SOCHamiltonianInterp))
    return SOCHamiltonianInterp{gauge}(f, λ)
end

contract(h::SOCHamiltonianInterp, x::Number, ::Val{d}) where d =
    SOCHamiltonianInterp{gauge(h)}(contract(h.f, x, Val(d)), h.λ)

period(h::SOCHamiltonianInterp) = period(h.f)

evaluate(h::SOCHamiltonianInterp, x::NTuple{1}) =
    to_gauge(h, evaluate(h.f, x) + h.λ)

GaugeDefault(::Type{<:SOCHamiltonianInterp}) = Wannier()

coefficients(h::SOCHamiltonianInterp) = coefficients(h.f)
