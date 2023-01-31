"""
    AbstractIntegrator{F} <: Function

Supertype of integration routines of a (collection of) function `F`.
"""
abstract type AbstractIntegrator{F} <: Function end

# interface

function limits end # return the limits object(s) used by the routine
function quad_integrand end # return the integrand used by the quadrature routine, optionally with arguments
quad_routine(f::AbstractIntegrator) = f.routine # integration routine
quad_args(f::AbstractIntegrator, ps...) = (quad_integrand(f.f, ps...), limits(f)...) # integration routine arguments
quad_kwargs(f::AbstractIntegrator; kwargs...) = (f.kwargs..., kwargs...) # keyword arguments to routine

# abstract methods

(f::AbstractIntegrator)(ps...; kwargs...) =
    quad_routine(f)(quad_args(f, ps...)...; quad_kwargs(f; kwargs...)...)

# implementations

"""
    IteratedIntegrator(f, l, p...; ps=0.0, routine=iterated_integration, args=(), kwargs=(;))

!!! warning "Experimental"
    Intended to integrate all kinds of [`AutoBZ.AbstractIntegrand`](@ref)
"""
struct IteratedIntegrator{F<:AbstractIteratedIntegrand,L<:AbstractLimits,P<:Tuple,R,K<:NamedTuple} <: AbstractIntegrator{F}
    f::F
    l::L
    p::P
    routine::R
    kwargs::K
end
function IteratedIntegrator(f::F, l, p...; ps=0.0, routine=iterated_integration, kwargs...) where F
    test = IteratedIntegrand{F}(p..., ps...)
    IteratedIntegrator(f, l, p, routine, quad_kwargs(routine, test, l; kwargs...))
end

limits(f::IteratedIntegrator) = f.l
IteratedIntegrator{F}(args...; kwargs...) where {F<:Function} =
    IteratedIntegrator(F.instance, args...; kwargs...)


"""
    FourierIntegrator(f, bz, s, p...; ps=0.0, routine=iterated_integration, args=(), kwargs=(;))
    FourierIntegrator{F}(args...; kwargs...) where {F<:Function}

Composite type that stores parts of a [`AutoBZ.FourierIntegrand`](@ref),
including the user-defined integrand `f`, the [`AutoBZ.AbstractBZ`] `bz`,
a [`AutoBZ.AbstractFourierSeries`] `s`, and additional parameters for the
integrand `ps`. Uses the integration routine specified by the caller.
Please provide a sample input `ps` (possibly a tuple of inputs) to the
constructor so that it can infer the type of outcome.

DEV notes: When called as a functor, this integrator concatenates `ps` followed
by the input `xs...` to the functor call to construct the integrand like so
`FourierIntegrand{typeof(f)}(s, ps..., xs...)`, which means that the arguments
passed to `f` which will be passed to the integrator must go in the last
positions. This mostly applies to aliases that may have specialized behavior
while also having an interface compatible with other routines (e.g. interpolation).
"""
struct FourierIntegrator{F,S<:AbstractFourierSeries,P<:Tuple,BZ<:AbstractBZ,R,K<:NamedTuple} <: AbstractIntegrator{F}
    f::F
    bz::BZ
    s::S
    p::P
    routine::R
    kwargs::K
end

function FourierIntegrator(f::F, bz, s, p...; ps=0.0, routine=iterated_integration, kwargs...) where F
    check_period_match(s, bz)
    test = FourierIntegrand{F}(s, p..., ps...)
    FourierIntegrator(f, bz, s, p, routine, quad_kwargs(routine, test, bz; kwargs...))
end


FourierIntegrator{F}(args...; kwargs...) where {F<:Function} =
    FourierIntegrator(F.instance, args...; kwargs...)
FourierIntegrator{F}(args...; kwargs...) where {F<:Tuple{Vararg{Function}}} =
    FourierIntegrator(tuple(map(f -> f.instance, F.parameters)...), args...; kwargs...)

limits(f::FourierIntegrator) = f.bz

build_integrand(f::FourierIntegrator{F}, ps...) where {F<:Tuple} = IteratedFourierIntegrand{F}(f.s, f.p..., ps...)


"""
    quad_args(routine, f, lims, [...])

Return the tuple of arguments needed by the quadrature `routine` other than the
test integrand  `f`, depending on the test integrand `f` and `lims`.
"""
quad_args(::typeof(quadgk), f, segs::T...) where T = segs
quad_args(::typeof(quadgk), f, lims::NTuple{N,T}) where {N,T} = lims
quad_args(::typeof(quadgk), f, lims::CubicLimits{1}) = limits(lims, 1)
quad_args(::typeof(iterated_integration), f, lims::CubicLimits{1}) = (lims,)

"""
    quad_kwargs(routine, f, lims, kwargs::NamedTuple)

Supplies the default keyword arguments to the given integration `routine`
without over-writing those already provided in `kwargs`
"""
quad_kwargs(::typeof(quadgk), f, lims; kwargs...) = quad_kwargs(quadgk, f, quad_args(quadgk, f, lims)...; kwargs...)
function quad_kwargs(::typeof(quadgk), f, segs::T...;
    atol=zero(coefficient_type(lims)), rtol=iszero(atol) ? sqrt(eps(coefficient_type(lims))) : zero(atol),
    order=7, maxevals=10^7, norm=norm, segbuf=nothing) where T
    F = Base.promote_op(f, T)
    segbuf_ = segbuf === nothing ? alloc_segbuf(T, F, Base.promote_op(norm, F)) : segbuf
    (rtol=rtol, atol=atol, order=order, maxevals=maxevals, norm=norm, segbuf=segbuf_)
end
quad_kwargs(::typeof(iterated_integration), f, l::AbstractLimits; kwargs...) =
    iterated_integration_kwargs(f, l; kwargs...)
quad_kwargs(::typeof(iterated_integration), f, bz::AbstractBZ; kwargs...) =
    iterated_integration_kwargs(f, limits(bz); kwargs...)
quad_kwargs(::typeof(automatic_equispace_integration), f, bz; kwargs...) =
    automatic_equispace_integration_kwargs(f, bz; kwargs...)
quad_kwargs(::typeof(equispace_integration), f, bz; kwargs...) =
    equispace_integration_kwargs(f, bz; kwargs...)