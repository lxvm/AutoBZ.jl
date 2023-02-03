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

fixandeliminate(c::CubicLimits{d,T}, _) where {d,T} =
    CubicLimits{d-1,T}(Base.front(c.a), Base.front(c.b))

"""
    TetrahedralLimits(a::NTuple{d}) where d

A parametrization of the integration limits for a tetrahedron whose vertices are
```
( 0.0,  0.0, ...,  0.0)
( 0.0,  0.0, ..., a[d])
…
( 0.0, a[2], ..., a[d])
(a[1], a[2], ..., a[d])
```
"""
struct TetrahedralLimits{d,T,A} <: AbstractLimits{d,T}
    a::A
    s::T
    TetrahedralLimits{d,T}(a::A, s::T) where {d,T,A<:NTuple{d,T}} = new{d,T,A}(a, s)
end
TetrahedralLimits(a::NTuple{d,T}) where {d,T} =
    TetrahedralLimits{d,float(T)}(ntuple(n -> float(a[n]), Val{d}()), one(T))
TetrahedralLimits(a::Tuple) = TetrahedralLimits(promote(a...))
TetrahedralLimits(a::AbstractVector) = TetrahedralLimits(Tuple(a))

endpoints(t::TetrahedralLimits{d,T}) where {d,T} = (zero(T), t.a[d]*t.s)

fixandeliminate(t::TetrahedralLimits{d,T}, x) where {d,T} =
    TetrahedralLimits{d-1,T}(Base.front(t.a), convert(T, x)/t.a[d])


function corners(t::AbstractLimits)
    a, b = endpoints(t)
    ndims(t) == 1 && return [(a,), (b,)]
    ta = corners(fixandeliminate(t, a))
    tb = corners(fixandeliminate(t, b))
    unique((map(x -> (x..., a), ta)..., map(x -> (x..., b), tb)...))
end


"""
    ProductLimits(lims::AbstractLimits...)

Construct a collection of limits which yields the first limit followed by the
second, and so on. The inner limits are not allowed to depend on the outer ones.
The outermost variable of integration should be placed first, i.e.
``\\int_{\\Omega} \\int_{\\Gamma}`` should be `ProductLimits(Ω, Γ)`.
"""
struct ProductLimits{d,T,L} <: AbstractLimits{d,T}
    lims::L
    ProductLimits{d,T}(lims::L) where {d,T,L<:Tuple{Vararg{AbstractLimits}}} =
        ProductLimits{d,T,L}(lims)
end
ProductLimits(lims::AbstractLimits...) = ProductLimits(lims)
function ProductLimits(lims::L) where {L<:Tuple{Vararg{AbstractLimits}}}
    d = mapreduce(ndims, +, lims; init=0)
    T = mapreduce(coefficient_type, promote_type, lims)
    ProductLimits{d,T}(lims)
end

endpoints(l::ProductLimits) = endpoints(l.lims[1])

fixandeliminate(l::ProductLimits{d,T}, x) where {d,T} =
    ProductLimits{d-1,T}(Base.setindex(l.lims, fixandeliminate(l.lims[1], x), 1))
fixandeliminate(l::ProductLimits{d,T,<:Tuple{<:AbstractLimits{1},Vararg{AbstractLimits}}}, x) where {d,T} =
    ProductLimits{d-1,T}(Base.front(l.lims))


"""
    TranslatedLimits(lims::AbstractLimits{d}, t::NTuple{d}) where d

Returns the limits of `lims` translated by offsets in `t`.
"""
struct TranslatedLimits{d,C,L,T} <: AbstractLimits{d,C}
    l::L
    t::T
    TranslatedLimits{d,C}(l::L, t::T) where {d,C,L<:AbstractLimits{d,C},T<:NTuple{d,C}} =
        new{d,C,L,T}(l, t)
end
TranslatedLimits(l::AbstractLimits{d,C}, t::NTuple{d}) where {d,C} =
    TranslatedLimits{d,C}(l, map(x -> convert(C, x), t))

endpoints(t::TranslatedLimits) =
    map(x -> x + t.t[ndims(t)], endpoints(t.l))
fixandeliminate(t::TranslatedLimits{d,C}, x) where {d,C} =
    TranslatedLimits{d-1,C}(fixandeliminate(t.l, convert(C, x) - t.t[ndims(t)]), Base.front(t.t))

# More ideas for transformed limits requiring linear programming
# RotatedLimits
# AffineLimits