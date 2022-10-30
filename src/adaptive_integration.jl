export tree_integration, iterated_integration
# TODO: rename callback to pre_eval
"""
    tree_integration(f, a, b)
    tree_integration(f, ::CubicLimits)

Calls `HCubature` to perform multi-dimensional integration of `f` over a cube.
"""
tree_integration(f, a, b; callback=nothing, kwargs...) = hcubature(f, a, b; kwargs...)
tree_integration(f, c::CubicLimits; kwargs...) = tree_integration(f, c.l, c.u; kwargs...)

"Choose a new set of error tolerances for the next inner integral."
default_error_callback(atol, rtol) = (atol, rtol)

"""
    iterated_integration(f, ::IntegrationLimits)

Calls `HCubature` to perform iterated 1D integration of `f` over a domain
parametrized by `IntegrationLimits`.
Accepts a callback function whose arguments are `f` and the evaluation point,
`x`, as a keyword argument. The callback can return a modified integrand to the
next inner integral, but the default is `thunk` which delays the computation to
the innermost integral.
"""
iterated_integration(f, a, b; kwargs...) = iterated_integration(f, CubicLimits(a, b); kwargs...)
function iterated_integration(f, l::IntegrationLimits; callback=thunk, atol=0.0, rtol=sqrt(eps()), error_callback=default_error_callback, kwargs...)
    int, err = iterated_integration_(f, l, callback, atol/nsyms(l), rtol, error_callback; kwargs...)
    symmetrize(l, int, err)
end

function iterated_integration_(f, l::IntegrationLimits{1}, callback, atol, rtol, error_callback; kwargs...) 
    hcubature(f, SVector(lower(l)), SVector(upper(l)); atol=atol, rtol=rtol, kwargs...)
end
function iterated_integration_(f, l::IntegrationLimits, callback, atol, rtol, error_callback; kwargs...)
    hcubature(SVector(lower(l)), SVector(upper(l)); atol=atol, rtol=rtol, kwargs...) do x
        first(iterated_integration_(callback(f, x), l(x), callback, error_callback(atol, rtol)..., error_callback; kwargs...))
    end
end
#= replacements with other quadrature routines
# TODO: think about how to make an interface to easily switch between routines
# TODO: write arbitrary order quadrature routine with a speedup like hcubature

# slower because wraps integrand with (x -> f(x[1])) and calls promote_type
# is it faster to write f ∘ only ?
iterated_integration_(f, l::IntegrationLimits{1}, callback; kwargs...) = hquadrature(f, lower(l), upper(l); kwargs...)
function iterated_integration_(f, l::IntegrationLimits, callback; kwargs...)
    hquadrature(lower(l), upper(l); kwargs...) do x
        first(iterated_integration_(callback(f, x), l(x), callback; kwargs...))
    end
end

# slower or may not even run ... not sure why ... maybe commit below helps
# https://github.com/JuliaMath/QuadGK.jl/commit/298f76e71be8a36d6e3f16715f601c3d22c2241c
# https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing
iterated_integration_(f, l::IntegrationLimits{1}, callback; kwargs...) = quadgk(f, lower(l), upper(l); kwargs...)
function iterated_integration_(f, l::IntegrationLimits, callback; kwargs...)
    quadgk(lower(l), upper(l); kwargs...) do x
        first(iterated_integration_(callback(f, x), l(x), callback; kwargs...))
    end
end
=#