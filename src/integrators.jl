export FourierIntegrator

"""
    FourierIntegrator(f, l, s, ps...; order=4, atol=0.0, rtol=sqrt(eps()), norm=norm)
    FourierIntegrator{F}(args...; kwargs...) where {F<:Function}

Composite type that stores parts of a [`AutoBZ.FourierIntegrand`](@ref),
including the user-defined integrand `f`, the [`AutoBZ.IntegrationLimits`] `l`,
a [`AutoBZ.AbstractFourierSeries`] `s`, and additional parameters for the
integrand `ps`.

DEV notes: When called as a functor, this integrator concatenates `ps` followed
by the input `xs...` to the functor call to construct the integrand like so
`FourierIntegrand{typeof(f)}(s, ps..., xs...)`, which means that the arguments
passed to `f` which will be passed to the integrator must go in the last
positions. This mostly applies to aliases that may have specialized behavior
while also having an interface compatible with other routines (e.g. interpolation).
"""
struct FourierIntegrator{F,S<:AbstractFourierSeries,P<:Tuple,L<:IntegrationLimits,N} <: Function
    f::F
    l::L
    s::S
    p::P
    order::Int
    atol::Float64
    rtol::Float64
    norm::N
end

FourierIntegrator{F}(args...; kwargs...) where {F<:Function} =
    FourierIntegrator(F.instance, args...; kwargs...) # allows dispatch by aliases

FourierIntegrator(f, l, s, ps...; order=4, atol=0.0, rtol=sqrt(eps()), norm=norm) =
    FourierIntegrator(f, l, s, ps, order, atol, rtol, norm)

(f::FourierIntegrator{F})(ps...) where F =
    first(iterated_integration(FourierIntegrand{F}(f.s, f.p..., ps...), f.l; order=f.order, atol=f.atol, rtol=f.rtol))