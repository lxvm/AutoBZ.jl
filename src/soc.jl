
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

# we overload A * b::SVector{<:SOC} to deal with a constructor error
# this could be optimized more (should be 2x faster than full matrix) but is already 10% faster than full
function Base.:*(A::StaticMatrix{N,N,<:Number}, b::SVector{N,<:SOC}) where {N}
    @assert N > 0
    vals = ntuple(Val(N)) do n
        ntuple(m -> A[n,m]*b[m], Val(N))
    end

    return SVector(ntuple(n -> +(vals[n]...), Val(N)))
end


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

struct WrapperFourierSeries{W,S,N,iip,C,A,T,F} <: AbstractFourierSeries{N,T,iip}
    w::W
    s::FourierSeries{S,N,iip,C,A,T,F}
end

Base.parent(s::WrapperFourierSeries) = s.s

period(s::WrapperFourierSeries) = period(parent(s))
frequency(s::WrapperFourierSeries) = frequency(parent(s))
allocate(s::WrapperFourierSeries, x, dim) = allocate(parent(s), x, dim)
function contract!(cache, s::WrapperFourierSeries, x, dim)
    return WrapperFourierSeries(s.w, contract!(cache, parent(s), x, dim))
end
evaluate!(cache, s::WrapperFourierSeries, x) = s.w(evaluate!(cache, parent(s), x))
nextderivative(s::WrapperFourierSeries, dim) = WrapperFourierSeries(s.w, nextderivative(parent(s), dim))

show_dims(s::WrapperFourierSeries) = show_dims(parent(s))
show_details(s::WrapperFourierSeries) = show_details(parent(s))

function shift!(s::WrapperFourierSeries, λ)
    shift!(parent(s), λ)
    return s
end


wrap_soc(A) = SOC(A)

struct SOCHamiltonianInterp{G,N,T,iip,S<:Freq2RadSeries{N,T,iip,<:WrapperFourierSeries{typeof(wrap_soc)}},L} <: AbstractHamiltonianInterp{G,N,T,iip}
    s::S
    λ::L
    SOCHamiltonianInterp{G}(s::Freq2RadSeries{N,T,iip,<:WrapperFourierSeries{typeof(wrap_soc)}}, λ) where {G,N,T,iip} =
        new{G,N,T,iip,typeof(s),typeof(λ)}(s,λ)
end

function SOCHamiltonianInterp(s, λ; gauge=GaugeDefault(SOCHamiltonianInterp))
    return SOCHamiltonianInterp{gauge}(s, λ)
end

GaugeDefault(::Type{<:SOCHamiltonianInterp}) = Wannier()
parentseries(h::SOCHamiltonianInterp) = h.s

period(h::SOCHamiltonianInterp) = period(parent(h))
frequency(h::SOCHamiltonianInterp) = frequency(parent(h))
allocate(h::SOCHamiltonianInterp, x, dim) = allocate(parent(h), x, dim)
function contract!(cache, h::SOCHamiltonianInterp, x, dim)
    return SOCHamiltonianInterp{gauge(h)}(contract!(cache, parent(h), x, dim), h.λ)
end
evaluate!(cache, h::SOCHamiltonianInterp, x) = to_gauge(h, evaluate!(cache, parent(h), x) + h.λ)
nextderivative(h::SOCHamiltonianInterp, dim) = nextderivative(parent(h), dim)
