export AbstractLimits, limits, domain_type
export CubicLimits, PolyhedralLimits, CompositeLimits

"""
    AbstractLimits{d,T<:AbstractFloat}

Represents a set of integration limits over `d` variables of type `T`.
Realizations of this type should implement `lower` and `upper`, which return the
lower and upper limits of integration along some dimension, `rescale` which
represents the number of symmetries of the BZ which are used by the realization
to reduce the BZ (the integrand over the limits gets multiplied by this factor),
and a functor that accepts a single numeric argument and returns another
realization of that type (in order to do nested integration). Thus the
realization is also in control of the order of variables of integration and must
coordinate this behavior with their integrand.
Instances should also be static structs.
"""
abstract type AbstractLimits{d,T<:AbstractFloat} end

# Interface for AbstractLimits types

"""
    limits(::AbstractLimits, dim)

Return a tuple with the lower and upper limit of the `dim`th variable of
integration.
"""
function limits end

# Generic methods for AbstractLimits

"""
    setindex(l::AbstractLimits, x, dims)

Compute the outermost `length(x)` variables of integration at the given `dims`.
Realizations of type `T<:AbstractLimits` only have to implement a method with
signature `setindex(::T, ::Number, ::Int)`.
"""
Base.setindex(l::AbstractLimits, x::SVector{N}, dims::NTuple{N,Int}) where N = setindex(setindex(l, last(x), dims[N]), pop(x), dims[1:N-1])
Base.setindex(l::AbstractLimits, ::SVector{0}, ::Tuple{}) = l

"""
    domain_type(x::AbstractLimits)

Returns the type of the coordinate coefficients for the domain
"""
domain_type(x) = domain_type(typeof(x))
domain_type(::Type{<:AbstractLimits{d,T}}) where {d,T} = T

Base.ndims(::T) where {T<:AbstractLimits} = ndims(T)
Base.ndims(::Type{<:AbstractLimits{d}}) where {d} = d

# Implementations of AbstractLimits

"""
    CubicLimits(a, [b])

Store integration limit information for a hypercube with vertices `a` and `b`.
If `b` is not passed as an argument, then the lower limit defaults to `zero(a)`.
"""
struct CubicLimits{d,T} <: AbstractLimits{d,T}
    l::SVector{d,T}
    u::SVector{d,T}
end
function CubicLimits(l::SVector{d,Tl}, u::SVector{d,Tu}) where {d,Tl<:Real,Tu<:Real}
    T = float(promote_type(Tl, Tu))
    CubicLimits{d,T}(SVector{d,T}(l), SVector{d,T}(u))
end
CubicLimits(l::Tl, u::Tu) where {Tl<:Real,Tu<:Real} = CubicLimits(SVector{1,Tl}(l), SVector{1,Tu}(u))
CubicLimits(u::SVector) = CubicLimits(zero(u), u)
CubicLimits(u::NTuple{N,T}) where {N,T} = CubicLimits(SVector{N,T}(u))

function limits(c::CubicLimits{d}, dim) where d
    1 <= dim <= d || throw(ArgumentError("pick dim=$(dim) in 1:$d"))
    return (c.l[dim], c.u[dim])
end
Base.setindex(c::CubicLimits, ::Number, dim::Int) = CubicLimits(deleteat(c.l, dim), deleteat(c.u, dim))

"""
    PolyhedralLimits(::Hull)

Integration limits from a convex hull.
"""
struct PolyhedralLimits{d,T,P<:Hull{T}} <: AbstractLimits{d,T}
    ch::P
    PolyhedralLimits{d}(ch::P) where {d,P} = new{d,coefficient_type(ch),P}(ch)
end
PolyhedralLimits(ch) = PolyhedralLimits{fulldim(ch)}(ch)
Base.setindex(l::PolyhedralLimits{d}, x, dim=d) where d =
    PolyhedralLimits{d-1}(intersect(l.ch, HyperPlane(setindex!(zeros(fulldim(l.ch)), 1, dim), x)))
limits(l::PolyhedralLimits{d}, dim=d) where d =
    (minimum(x -> x[dim], points(l.ch)), maximum(x -> x[dim], points(l.ch)))


"""
    CompositeLimits(lims::AbstractLimits...)
    CompositeLimits(::Tuple{Vararg{AbstractLimits}})

Construct a collection of limits which yields the first limit followed by the
second, and so on.
"""
struct CompositeLimits{d,T,L<:Tuple{Vararg{AbstractLimits}}} <: AbstractLimits{d,T}
    lims::L
end
CompositeLimits(lims::AbstractLimits...) = CompositeLimits(lims)
function CompositeLimits(lims::L) where {L<:Tuple{Vararg{AbstractLimits}}}
    CompositeLimits{mapreduce(ndims, +, lims; init=0),mapreduce(eltype, promote_type, lims),L}(lims)
end
CompositeLimits{d,T}(lims::AbstractLimits...) where {d,T} = CompositeLimits{d,T}(lims)
CompositeLimits{d,T}(lims::L) where {d,T,L} = CompositeLimits{d,T,L}(lims)
Base.setindex(l::CompositeLimits{d,T,L}, x::Number, dim) where {d,T,L} =
    CompositeLimits{d-1,T}(first(l.lims)(x), Base.rest(l.lims, 2)...)
Base.setindex(l::CompositeLimits{d,T,L}, ::Number, dim) where {d,T,L<:Tuple{<:AbstractLimits{1},Vararg{AbstractLimits}}} =
    CompositeLimits{d-1,T}(Base.rest(l.lims, 2)...)
