export IntegrationLimits, limits, box, vol, nsyms, symmetrize, symmetries,
    CubicLimits, CompositeLimits

"""
    IntegrationLimits{d}

Represents a set of integration limits over `d` variables.
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
abstract type IntegrationLimits{d} end

# Interface for IntegrationLimits types

"""
    box(::IntegrationLimits)

Return an iterator of tuples that for each dimension returns a tuple with the
lower and upper limits of the integration domain without symmetries applied.
"""
function box end

"""
    limits(::IntegrationLimits, dim)

Return a tuple with the lower and upper limit of the `dim`th variable of
integration.
"""
function limits end

"""
    nsyms(::IntegrationLimits)

Return the number of symmetries that the parametrization has used to reduce the
volume of the integration domain.
"""
function nsyms end

"""
    symmetries(::IntegrationLimits)

Return an iterator over the symmetry transformations that the parametrization
has used to reduce the volume of the integration domain.
"""
function symmetries end

"""
    discretize_equispace(::IntegrationLimits, ::Integer)

Return an iterator of 2-tuples containing integration nodes and weights that
correspond to an equispace integration grid with the symmetry transformations
applied to it.
"""
function discretize_equispace end

# Generic methods for IntegrationLimits

"""
    (::IntegrationLimits)(x::SVector)

Compute the outermost `length(x)` variables of integration.
Realizations of type `T<:IntegrationLimits` only have to implement a method with
signature `(::T)(::Number)`.
"""
(l::IntegrationLimits)(x, dim) = l(x)
(l::IntegrationLimits)(x::SVector) = l(last(x))(pop(x))
(l::IntegrationLimits)(::SVector{0}) = l

"""
    symmetrize(::IntegrationLimits, x)
    symmetrize(::IntegrationLimits, xs...)

Transform `x` by the symmetries of the parametrization used to reduce the
domain, thus mapping the value of `x` on the parametrization to the full domain.
When the integrand is a scalar, this is equal to `nsyms(l)*x`.
When the integrand is a vector, this is `sum(S*x for S in symmetries(l))`.
When the integrand is a matrix, this is `sum(S*x*S' for S in symmetries(l))`.
"""
symmetrize(l::IntegrationLimits, xs...) = map(x -> symmetrize(l, x), xs)
symmetrize(l::IntegrationLimits, x) = symmetrize_(x, nsyms(l), symmetries(l))
symmetrize_(x::Number, nsym, syms) = nsym*x
symmetrize_(x::AbstractArray{<:Any,0}, nsym, syms) = symmetrize_(only(x), nsym, syms)
function symmetrize_(x::AbstractVector, nsym, syms)
    r = zero(x)
    for S in syms
        r += S * x
    end
    r
end
function symmetrize_(x::AbstractMatrix, nsym, syms)
    r = zero(x)
    for S in syms
        r += S * x * S'
    end
    r
end

"""
    vol(::IntegrationLimits)

Return the volume of the full domain without the symmetries applied
"""
vol(l::IntegrationLimits) = prod(u-l for (l, u) in box(l))

"""
    ndims(::IntegrationLimits{d})

Returns `d`. This is a type-based rule.
"""
Base.ndims(::T) where {T<:IntegrationLimits} = ndims(T)
Base.ndims(::Type{<:IntegrationLimits{d}}) where {d} = d

# Implementations of IntegrationLimits

"""
    CubicLimits(a, [b])

Store integration limit information for a hypercube with vertices `a` and `b`.
If `b` is not passed as an argument, then the lower limit defaults to `zero(a)`.
"""
struct CubicLimits{d,T<:Real} <: IntegrationLimits{d}
    l::SVector{d,T}
    u::SVector{d,T}
end
function CubicLimits(l::SVector{d,Tl}, u::SVector{d,Tu}) where {d,Tl<:Real,Tu<:Real}
    T = float(promote_type(Tl, Tu))
    CubicLimits(SVector{d,T}(l), SVector{d,T}(u))
end
CubicLimits(l::Tl, u::Tu) where {Tl<:Real,Tu<:Real} = CubicLimits(SVector{1,Tl}(l), SVector{1,Tu}(u))
CubicLimits(u::SVector) = CubicLimits(zero(u), u)
CubicLimits(u::NTuple{N,T}) where {N,T} = CubicLimits(SVector{N,T}(u))

"""
    limits(::CubicLimits, dim)

Returns a tuple with the lower and upper limit of the outermost variable of
integration.
"""
function limits(c::CubicLimits{d}, dim) where d
    1 <= dim <= d || throw(ArgumentError("pick dim=$(dim) in 1:$d"))
    return (c.l[dim], c.u[dim])
end

"""
    (::CubicLimits)(x::Number)

Return a CubicLimits of lower dimension with the outermost variable of
integration removed.
"""
(c::CubicLimits)(::Number) = CubicLimits(pop(c.l), pop(c.u))

domain_type(::Type{CubicLimits{d,T}}) where {d,T} = T


"""
    TetrahedralLimits(a::SVector)
    TetrahedralLimits(a::CubicLimits)
    TetrahedralLimits(a, p)

A parametrization of the integration limits for a tetrahedron generated from the
automorphism group of the hypercube whose corners are `-a*p` and `a*p`. By
default, `p=0.5` which gives integration over the irreducible Brillouin zone of
the cube. If the entries of `a` vary then this implies the hypercube is
rectangular.
"""
struct TetrahedralLimits{d,T<:AbstractFloat} <: IntegrationLimits{d}
    a::SVector{d,T}
    p::T
end
TetrahedralLimits(a::SVector{d,T}) where {d,T<:AbstractFloat} = TetrahedralLimits(a, one(T)/2)
TetrahedralLimits(a::NTuple{d,T}) where {d,T} = TetrahedralLimits(SVector{d,T}(a))
TetrahedralLimits(c::CubicLimits) = TetrahedralLimits(c.u-c.l)
(t::TetrahedralLimits)(x::Number) = TetrahedralLimits(pop(t.a), x/last(t.a))
domain_type(::Type{TetrahedralLimits{d,T}}) where {d,T} = T

struct PolyhedronLimits{d,T,P<:Polyhedra.Hull{T}} <: IntegrationLimits{d}
    ch::P
    PolyhedronLimits{d}(ch::P) where {d,P} = new{d,Polyhedra.coefficient_type(ch),P}(ch)
end
PolyhedronLimits(ch) = PolyhedronLimits{fulldim(ch)}(ch)
(l::PolyhedronLimits{d})(x, dim=d) where d =
    PolyhedronLimits{d-1}(intersect(l.ch, HyperPlane(setindex!(zeros(fulldim(l.ch)), 1, dim), x)))
limits(l::PolyhedronLimits{d}, dim=d) where d =
    (minimum(x -> x[dim], points(l.ch)), maximum(x -> x[dim], points(l.ch)))
domain_type(::Type{<:PolyhedronLimits{d,T}}) where {d,T} = T



"""
    CompositeLimits(lims::IntegrationLimits...)
    CompositeLimits(::Tuple{Vararg{IntegrationLimits}})

Construct a collection of limits which yields the first limit followed by the
second, and so on.
"""
struct CompositeLimits{d,T,L<:Tuple{Vararg{IntegrationLimits}}} <: IntegrationLimits{d}
    lims::L
end
CompositeLimits(lims::IntegrationLimits...) = CompositeLimits(lims)
function CompositeLimits(lims::L) where {L<:Tuple{Vararg{IntegrationLimits}}}
    CompositeLimits{mapreduce(ndims, +, lims; init=0),mapreduce(eltype, promote_type, lims),L}(lims)
end
CompositeLimits{d,T}(lims::IntegrationLimits...) where {d,T} = CompositeLimits{d,T}(lims)
CompositeLimits{d,T}(lims::L) where {d,T,L} = CompositeLimits{d,T,L}(lims)
(l::CompositeLimits{d,T,L})(x::Number) where {d,T,L} = CompositeLimits{d-1,T}(first(l.lims)(x), Base.rest(l.lims, 2)...)
(l::CompositeLimits{d,T,L})(::Number) where {d,T,L<:Tuple{<:IntegrationLimits{1},Vararg{IntegrationLimits}}} = CompositeLimits{d-1,T}(Base.rest(l.lims, 2)...)
domain_type(::Type{<:CompositeLimits{d,T}}) where {d,T} = T
