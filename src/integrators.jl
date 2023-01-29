export FourierIntegrator

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
struct FourierIntegrator{F,S<:AbstractFourierSeries,P<:Tuple,BZ<:AbstractBZ,R,K<:NamedTuple} <: Function
    f::F
    bz::BZ
    s::S
    p::P
    routine::R
    kwargs::K
end

function FourierIntegrator(f::F, bz, s, p...; ps=0.0, routine=iterated_integration, kwargs...) where F
    check_period_match(s, bz)
    FourierIntegrator(f, bz, s, p, routine, default_kwargs(routine, FourierIntegrand{F}(s, p..., ps...), bz; kwargs...))
end

# allows dispatch by aliases to accommodate broader user interface
FourierIntegrator{F}(args...; kwargs...) where {F<:Function} =
    FourierIntegrator(F.instance, args...; kwargs...) 
FourierIntegrator{F}(args...; kwargs...) where {F<:Tuple{Vararg{Function}}} =
    FourierIntegrator(tuple(map(f -> f.instance, F.parameters)...), args...; kwargs...)

(f::FourierIntegrator{F})(ps...) where F =
    first(f.routine(FourierIntegrand{F}(f.s, f.p..., ps...), f.bz,; f.kwargs...))

(f::FourierIntegrator{F})(ps...) where {F<:Tuple} =
    first(f.routine(IteratedFourierIntegrand{F}(f.s, f.p..., ps...), f.bz; f.kwargs...))

"""
    default_kwargs(routine, f, bz, kwargs::NamedTuple)

Supplies the default keyword arguments to the given integration `routine`
without over-writing those already provided in `kwargs`
"""
default_kwargs(::typeof(iterated_integration), f, bz; kwargs...) =
    iterated_integration_kwargs(f, limits(bz); kwargs...)
default_kwargs(::typeof(automatic_equispace_integration), f, bz; kwargs...) =
    automatic_equispace_integration_kwargs(f, bz; kwargs...)
default_kwargs(::typeof(equispace_integration), f, bz; kwargs...) =
    equispace_integration_kwargs(f, bz; kwargs...)