export IntegrationLimits, lower, upper, rescale, CubicLimits

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
rescale(::IntegrationLimits)

Return the number of symmetries used to reduce the volume of the integration
domain. This is called only once when computing the integral to rescale the
final result.
"""
function rescale end

"""
    (::IntegrationLimits)(x::SVector)

Compute the outermost `length(x)` variables of integration.
Realizations of type `T<:IntegrationLimits` only have to implement a method with
signature `(::T)(::Number)`.
"""
(l::IntegrationLimits)(x::SVector) = l(last(x))(pop(x))
(l::IntegrationLimits)(::SVector{0}) = l

"""
    CubicLimits{N,T}

Store integration limit information for a hypercube.
"""
struct CubicLimits{d,Tl<:Number,Tu<:Number} <: IntegrationLimits{d}
    l::SVector{d,Tl}
    u::SVector{d,Tu}
end

"""
    lower(::CubicLimits)

Returns the lower limit of the outermost variable of integration (this choice is
made to help preserve memory continguity for tensor contractions).
"""
lower(l::CubicLimits) = last(l.l)

"""
    lower(::CubicLimits)

Returns the lower limit of the outermost variable of integration (this choice is
made to help preserve memory contiguity for tensor contractions).
"""
upper(l::CubicLimits) = last(l.u)

"""
    rescale(::CubicLimits)

Returns 1 because the integration region is exactly the cube.
"""
rescale(::CubicLimits) = 1

"""
    (::CubicLimits)(x::Number)

Return a CubicLimits of lower dimension with the outermost variable of
integration removed.
"""
(l::CubicLimits)(::Number) = CubicLimits(pop(l.l), pop(l.u))