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
    endpoints(::AbstractLimits{d}, [dim=d]) where d

Return a tuple with the lower and upper limit of the `dim`th variable of
integration, which is the outermost by default. This is equivalent to projecting
the integration domain onto one dimension.
"""
endpoints(l::AbstractLimits{d}) where d = endpoints(l, d)

"""
    fixandeliminate(l::AbstractLimits, x)

Fix the outermost variable of integration and return the inner limits.

!!! note "For developers"
    Realizations of type `T<:AbstractLimits` only have to implement a method
    with signature `fixandeliminate(::T, ::Number)`. The result must also have
    dimension one less than the input, and this should only be called when ndims
    >= 1
"""
function fixandeliminate end

# Generic methods for AbstractLimits

Base.ndims(::AbstractLimits{d}) where {d} = d

"""
    coefficient_type(x::AbstractLimits)

Returns the type of the coordinate coefficients for the domain
"""
coefficient_type(x) = coefficient_type(typeof(x))
coefficient_type(::Type{<:AbstractLimits{d,T}}) where {d,T} = T

# Implementations of AbstractLimits

"""
    CubicLimits(a, b)

Store integration limit information for a hypercube with vertices `a` and `b`.
which can be can be real numbers, tuples, or `AbstractVector`s.
The outermost variable of integration corresponds to the last entry.
"""
struct CubicLimits{d,T,L} <: AbstractLimits{d,T}
    a::L
    b::L
    CubicLimits{d,T}(a::L, b::L) where {d,T,L<:NTuple{d,T}} = new{d,T,L}(a, b)
end
function CubicLimits(a::NTuple{d,A}, b::NTuple{d,B}) where {d,A<:Real,B<:Real}
    T = float(promote_type(A, B))
    CubicLimits{d,T}(
        ntuple(n -> T(a[n]), Val{d}()),
        ntuple(n -> T(b[n]), Val{d}()),
    )
end
CubicLimits(a::Real, b::Real) = CubicLimits((a,), (b,))
CubicLimits(a::AbstractVector, b::AbstractVector) = CubicLimits(Tuple(a), Tuple(b))

function endpoints(c::CubicLimits{d}, dim) where d
    1 <= dim <= d || throw(ArgumentError("pick dim=$(dim) in 1:$d"))
    return (c.a[dim], c.b[dim])
end
fixandeliminate(c::CubicLimits, _) = CubicLimits(Base.front(c.a), Base.front(c.b))

"""
    TetrahedralLimits(a::NTuple)

A parametrization of the integration limits for a tetrahedron whose vertices are
the origin and the unit coordinate vectors rescaled by the components of `a`.
"""
struct TetrahedralLimits{d,T,A} <: AbstractLimits{d,T}
    a::A
    TetrahedralLimits(a::A) where {d,T,A<:NTuple{d,T}} = new{d,T,A}(a)
end
TetrahedralLimits(a::AbstractVector) = TetrahedralLimits(Tuple(a))

endpoints(t::TetrahedralLimits) =
    (zero(coefficient_type(T)), t.a[ndims(t)])
fixandeliminate(t::TetrahedralLimits, x) =
    TetrahedralLimits(Base.front(setindex(t.a, convert(coefficient_type(t), x)/t.a[ndims(t)], ndims(t)-1)))

"""
    ProductLimits(lims::AbstractLimits...)
    ProductLimits(::Tuple{Vararg{AbstractLimits}})

Construct a collection of limits which yields the first limit followed by the
second, and so on. The inner limits are not allowed to depend on the outer ones.
The outermost variable of integration should be placed first, i.e.
``\\int_{\\Omega} \\int_{\\Gamma}`` should be `ProductLimits(Ω, Γ)`.
"""
struct ProductLimits{d,T,L<:Tuple{Vararg{AbstractLimits}}} <: AbstractLimits{d,T}
    lims::L
end
ProductLimits(lims::AbstractLimits...) = ProductLimits(lims)
function ProductLimits(lims::L) where {L<:Tuple{Vararg{AbstractLimits}}}
    ProductLimits{mapreduce(ndims, +, lims; init=0),mapreduce(coefficient_type, promote_type, lims),L}(lims)
end
ProductLimits{d,T}(lims::AbstractLimits...) where {d,T} = ProductLimits{d,T}(lims)
ProductLimits{d,T}(lims::L) where {d,T,L} = ProductLimits{d,T,L}(lims)

endpoints(l::ProductLimits) = endpoints(l.lims[1])

fixandeliminate(l::ProductLimits{d,T}, x::Number) where {d,T} =
    ProductLimits{d-1,T}(Base.setindex(l.lims, fixandeliminate(l.lims[1], x), 1))
fixandeliminate(l::ProductLimits{d,T,<:Tuple{<:AbstractLimits{1},Vararg{AbstractLimits}}}, x::Number) where {d,T} =
    ProductLimits{d-1,T}(Base.setindex(l.lims, fixandeliminate(l.lims[1], x), 1))
