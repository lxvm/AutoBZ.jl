"""
    SSymmetricCompact{N, T, L} <: StaticMatrix{N, N, T}

A `StaticArray` subtype that represents a Symmetric matrix. Unlike
`LinearAlgebra.Symmetric`, `SSymmetricCompact` stores only the lower triangle
of the matrix (as an `SVector`). The lower triangle is stored in column-major order.
For example, for an `SSymmetricCompact{3}`, the indices of the stored elements
can be visualized as follows:

```
┌ 1 ⋅ ⋅ ┐
| 2 4 ⋅ |
└ 3 5 6 ┘
```

Type parameters:
* `N`: matrix dimension;
* `T`: element type for lower triangle;
* `L`: length of the `SVector` storing the lower triangular elements.

Note that `L` is always the `N`th [triangular number](https://en.wikipedia.org/wiki/Triangular_number).

An `SSymmetricCompact` may be constructed either:

* from an `AbstractVector` containing the lower triangular elements; or
* from a `Tuple` containing both upper and lower triangular elements in column major order; or
* from another `StaticMatrix`.

For the latter two cases, only the lower triangular elements are used; the upper triangular
elements are ignored.
"""
struct SSymmetricCompact{N, T, L} <: StaticMatrix{N, N, T}
    lowertriangle::SVector{L, T}

    @inline function SSymmetricCompact{N, T, L}(lowertriangle::SVector{L}) where {N, T, L}
        _check_hermitian_parameters(Val(N), Val(L))
        new{N, T, L}(lowertriangle)
    end
end

@inline function _check_hermitian_parameters(::Val{N}, ::Val{L}) where {N, L}
    if 2 * L !== N * (N + 1)
        throw(ArgumentError("Size mismatch in SSymmetricCompact parameters. Got dimension $N and length $L."))
    end
end

Base.@pure triangularnumber(N::Int) = div(N * (N + 1), 2)
Base.@pure triangularroot(L::Int) = div(isqrt(8 * L + 1) - 1, 2) # from quadratic formula

lowertriangletype(::Type{SSymmetricCompact{N, T, L}}) where {N, T, L} = SVector{L, T}
lowertriangletype(::Type{SSymmetricCompact{N, T}}) where {N, T} = SVector{triangularnumber(N), T}
lowertriangletype(::Type{SSymmetricCompact{N}}) where {N} = SVector{triangularnumber(N)}

@inline SSymmetricCompact{N, T}(lowertriangle::SVector{L}) where {N, T, L} = SSymmetricCompact{N, T, L}(lowertriangle)
@inline SSymmetricCompact{N}(lowertriangle::SVector{L, T}) where {N, T, L} = SSymmetricCompact{N, T, L}(lowertriangle)

@inline function SSymmetricCompact(lowertriangle::SVector{L, T}) where {T, L}
    N = triangularroot(L)
    SSymmetricCompact{N, T, L}(lowertriangle)
end

@generated function SSymmetricCompact{N, T, L}(a::Tuple) where {N, T, L}
    _check_hermitian_parameters(Val(N), Val(L))
    expr = Vector{Expr}(undef, L)
    i = 0
    for col = 1 : N, row = col : N
        index = N * (col - 1) + row
        expr[i += 1] = :(a[$index])
    end
    quote
        StaticArrays.@_inline_meta
        @inbounds return SSymmetricCompact{N, T, L}(SVector{L, T}(tuple($(expr...))))
    end
end

@inline function SSymmetricCompact{N, T}(a::Tuple) where {N, T}
    L = triangularnumber(N)
    SSymmetricCompact{N, T, L}(a)
end

@inline (::Type{SSC})(a::SSymmetricCompact) where {SSC <: SSymmetricCompact} = SSC(a.lowertriangle)

@inline (::Type{SSC})(a::AbstractVector) where {SSC <: SSymmetricCompact} = SSC(convert(lowertriangletype(SSC), a))

# disambiguation
@inline (::Type{SSC})(a::StaticArray{<:Tuple,<:Any,1}) where {SSC <: SSymmetricCompact} = SSC(convert(SVector, a))

@generated function _symmetric_compact_indices(::Val{N}) where N
    # Returns a Tuple{Int} I such that for linear index i,
    # * I[i] is the index into the lowertriangle field of an SSymmetricCompact{N};
    indexmat = Matrix{Int}(undef, N, N)
    i = 0
    for col = 1 : N, row = 1 : N
        indexmat[row, col] = if row >= col
            (i += 1)
        else
            indexmat[col, row]
        end
    end
    quote
        StaticArrays.@_inline_meta
        return $(tuple(indexmat...))
    end
end

Base.@propagate_inbounds function Base.getindex(a::SSymmetricCompact{N}, i::Int) where {N}
    I = _symmetric_compact_indices(Val(N))
    j = I[i]
    @inbounds value = a.lowertriangle[j]
    return value
end

Base.@propagate_inbounds function Base.setindex(a::SSymmetricCompact{N, T, L}, x, i::Int) where {N, T, L}
    I = _symmetric_compact_indices(Val(N))
    j = I[i]
    value = x
    return SSymmetricCompact{N}(setindex(a.lowertriangle, value, j))
end

# needed because it is used in convert.jl and the generic fallback is slow
@generated function Base.Tuple(a::SSymmetricCompact{N}) where N
    exprs = [:(a[$i]) for i = 1 : N^2]
    quote
        StaticArrays.@_inline_meta
        tuple($(exprs...))
    end
end

LinearAlgebra.ishermitian(a::SSymmetricCompact) = true
LinearAlgebra.issymmetric(a::SSymmetricCompact) = true

# TODO: factorize?

@inline Base.:(==)(a::SSymmetricCompact, b::SSymmetricCompact) = a.lowertriangle == b.lowertriangle
@generated function _map(f, a::SSymmetricCompact...)
    S = Size(a[1])
    N = S[1]
    L = triangularnumber(N)
    exprs = Vector{Expr}(undef, L)
    for i ∈ 1:L
        tmp = [:(a[$j].lowertriangle[$i]) for j ∈ 1:length(a)]
        exprs[i] = :(f($(tmp...)))
    end
    return quote
        StaticArrays.@_inline_meta
        same_size(a...)
        @inbounds return SSymmetricCompact(SVector(tuple($(exprs...))))
    end
end

@inline Base.:*(a::Number, b::SSymmetricCompact) = SSymmetricCompact(a * b.lowertriangle)
@inline Base.:*(a::SSymmetricCompact, b::Number) = SSymmetricCompact(a.lowertriangle * b)

@inline Base.:/(a::SSymmetricCompact, b::Number) = SSymmetricCompact(a.lowertriangle / b)
@inline Base.:\(a::Number, b::SSymmetricCompact) = SSymmetricCompact(a \ b.lowertriangle)

@generated function _plus_uniform(::Size{S}, a::SSymmetricCompact{N, T, L}, λ) where {S, N, T, L}
    @assert S[1] == N
    @assert S[2] == N
    exprs = Vector{Expr}(undef, L)
    i = 0
    for col = 1 : N, row = col : N
        i += 1
        exprs[i] = row == col ? :(a.lowertriangle[$i] + λ) : :(a.lowertriangle[$i])
    end
    return quote
        StaticArrays.@_inline_meta
        R = promote_type(eltype(a), typeof(λ))
        SSymmetricCompact{N, R, L}(SVector{L, R}(tuple($(exprs...))))
    end
end

@generated function _one(::Size{S}, ::Type{SSC}) where {S, SSC <: SSymmetricCompact}
    N = S[1]
    L = triangularnumber(N)
    T = eltype(SSC)
    if T == Any
        T = Float64
    end
    exprs = Vector{Expr}(undef, L)
    i = 0
    for col = 1 : N, row = col : N
        exprs[i += 1] = row == col ? :(one($T)) : :(zero($T))
    end
    quote
        StaticArrays.@_inline_meta
        return SSymmetricCompact(SVector(tuple($(exprs...))))
    end
end

@inline _scalar_matrix(s::Size{S}, t::Type{SSC}) where {S, SSC <: SSymmetricCompact} = _one(s, t)

# _fill covers fill, zeros, and ones:
@generated function _fill(val, ::Size{s}, ::Type{SSC}) where {s, SSC <: SSymmetricCompact}
    N = s[1]
    L = triangularnumber(N)
    v = [:val for i = 1:L]
    return quote
        StaticArrays.@_inline_meta
        $SSC(SVector(tuple($(v...))))
    end
end
#=
@generated function LinearAlgebra.transpose(a::SSymmetricCompact{N, T, L}) where {N, T, L}
    # To conform with LinearAlgebra, the transpose should be recursive.
    # For this compact representation of a Hermitian matrix, that means that
    # we should recursively transpose of the diagonal elements, but only
    # conjugate the off-diagonal elements:
    # [A  Bᴴ]ᵀ  =  [Aᵀ  Bᵀ]  =  [Aᵀ      Bᵀ]
    # [B  C ]      [Bᴴᵀ Cᵀ]     [conj(B) Cᵀ]
    exprs = Vector{Expr}(undef, L)
    i = 0
    for col = 1 : N, row = col : N
        i += 1
        exprs[i] = row == col ? :(transpose(a.lowertriangle[$i])) : :(conj(a.lowertriangle[$i]))
    end
    return quote
        StaticArrays.@_inline_meta
        SSymmetricCompact{N}(SVector{L}(tuple($(exprs...))))
    end
end

@generated function _rand(randfun, rng::AbstractRNG, ::Type{SSC}) where {N, SSC <: SSymmetricCompact{N}}
    T = eltype(SSC)
    if T == Any
        T = Float64
    end
    L = triangularnumber(N)
    v = [:(randfun(rng, $T)) for i = 1:L]
    return quote
        StaticArrays.@_inline_meta
        $SSC(SVector(tuple($(v...))))
    end
end

@inline Random.rand(rng::AbstractRNG, ::Type{SSC}) where {SSC <: SSymmetricCompact} = _rand(rand, rng, SSC)
@inline Random.randn(rng::AbstractRNG, ::Type{SSC}) where {SSC <: SSymmetricCompact} = _rand(randn, rng, SSC)
@inline Random.randexp(rng::AbstractRNG, ::Type{SSC}) where {SSC <: SSymmetricCompact} = _rand(randexp, rng, SSC)
=#
