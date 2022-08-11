export IntegrationLimits, lower, upper, nsym, symmetrize, symmetries,
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
    lower(::IntegrationLimits)

Return the lower limit of the next variable of integration. If a vector is
returned, then the integration routine may attempt to multidimensional
integration.
"""
function lower end

"""
    upper(::IntegrationLimits)

Return the upper limit of the next variable of integration. If a vector is
returned, then the integration routine may attempt to multidimensional
integration.
"""
function upper end

"""
    nsym(::IntegrationLimits)

Return the number of symmetries that the parametrization has used to reduce the
volume of the integration domain.
"""
function nsym end

"""
    symmetries(::IntegrationLimits)

Return an iterator over the symmetry transformations that the parametrization
has used to reduce the volume of the integration domain.
"""
function symmetries end

# Generic methods for IntegrationLimits

"""
    (::IntegrationLimits)(x::SVector)

Compute the outermost `length(x)` variables of integration.
Realizations of type `T<:IntegrationLimits` only have to implement a method with
signature `(::T)(::Number)`.
"""
(l::IntegrationLimits)(x::SVector) = l(last(x))(pop(x))
(l::IntegrationLimits)(::SVector{0}) = l

"""
    symmetrize(::IntegrationLimits, x)
    symmetrize(::IntegrationLimits, xs...)

Transform `x` by the symmetries of the parametrization used to reduce the
domain, thus mapping the value of `x` on the parametrization to the full domain.
When the integrand is a scalar, this is equal to `nsym(l)*x`.
When the integrand is a vector, this is `sum(S*x for S in symmetries(l))`.
When the integrand is a matrix, this is `sum(S*x*S' for S in symmetries(l))`.
"""
symmetrize(l::IntegrationLimits, xs...) = map(x -> symmetrize(l, x), xs)
symmetrize(l::IntegrationLimits, x) = symmetrize_(x, nsym(l), symmetries(l))
symmetrize_(x::Number, nsyms, syms) = nsyms*x
symmetrize_(x::AbstractArray{<:Any,0}, nsyms, syms) = symmetrize_(only(x), nsyms, syms)
function symmetrize_(x::AbstractVector, nsyms, syms)
    r = zero(x)
    for S in syms
        r += S * x
    end
    r
end
function symmetrize_(x::AbstractMatrix, nsyms, syms)
    r = zero(x)
    for S in syms
        r += S * x * S'
    end
    r
end

"""
    ndims(::IntegrationLimits{d})

Returns `d`. This is a type-based rule.
"""
Base.ndims(::T) where {T<:IntegrationLimits} = ndims(T)
Base.ndims(::Type{<:IntegrationLimits{d}}) where {d} = d

# Implementations of IntegrationLimits

"""
    CubicLimits(a, b)

Store integration limit information for a hypercube with vertices `a` and `b`.
"""
struct CubicLimits{d,Tl<:Number,Tu<:Number} <: IntegrationLimits{d}
    l::SVector{d,Tl}
    u::SVector{d,Tu}
end

"""
    lower(::CubicLimits)

Returns the lower limit of the outermost variable of integration.
"""
lower(c::CubicLimits) = last(c.l)

"""
    upper(::CubicLimits)

Returns the upper limit of the outermost variable of integration.
"""
upper(c::CubicLimits) = last(c.u)

"""
    nsym(::CubicLimits)

Returns 1 because the only symmetry applied to the cube is the identity.
"""
nsym(::CubicLimits) = 1

"""
    symmetries(::CubicLimits)

Return an identity matrix.
"""
symmetries(::CubicLimits) = tuple(I)

"""
    (::CubicLimits)(x::Number)

Return a CubicLimits of lower dimension with the outermost variable of
integration removed.
"""
(c::CubicLimits)(::Number) = CubicLimits(pop(c.l), pop(c.u))

"""
    CompositeLimits(::Tuple{Vararg{IntegrationLimits}})

Construct a collection of limits which yields the first limit followed by the
second, and so on.
"""
struct CompositeLimits{T<:Tuple{Vararg{IntegrationLimits}},d} <: IntegrationLimits{d}
    lims::T
    function CompositeLimits(lims::T) where {T<:Tuple{Vararg{IntegrationLimits}}}
        new{T, mapreduce(ndims, +, T.parameters; init=0)}(lims)
    end
end
CompositeLimits(lims::IntegrationLimits...) = CompositeLimits(lims)
(l::CompositeLimits)(x::Number) = CompositeLimits(first(l.lims)(x), Base.rest(l.lims, 2)...)
(l::CompositeLimits{T})(x::Number) where {T<:Tuple{<:IntegrationLimits}} = CompositeLimits(first(l.lims)(x))
(l::CompositeLimits{T})(::Number) where {T<:Tuple{<:IntegrationLimits{1},Vararg{IntegrationLimits}}} = CompositeLimits(Base.rest(l.lims, 2)...)

lower(l::CompositeLimits) = lower(first(l.lims))
upper(l::CompositeLimits) = upper(first(l.lims))
nsym(l::CompositeLimits) = prod(nsym, l.lims)
function symmetrize(l::CompositeLimits, x)
    r = x
    for lim in l.lims
        r = symmetrize(lim, r)
    end
    r
end

