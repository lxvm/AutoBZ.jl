export tree_integration, iterated_integration

"""
    tree_integration(f, a, b)
    tree_integration(f, ::CubicLimits)

Calls `HCubature` to perform multi-dimensional integration of `f` over a cube.
"""
tree_integration(f, a, b; callback=nothing, kwargs...) = hcubature(f, a, b; kwargs...)
tree_integration(f, c::CubicLimits; kwargs...) = tree_integration(f, c.l, c.u; kwargs...)

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
function iterated_integration(f, L::IntegrationLimits; kwargs...)
    int, err = iterated_integration_(f, L; kwargs...)
    rescale(L)*int, err
end

iterated_integration_(f, L::IntegrationLimits{1}; callback=nothing, kwargs...) = hcubature(f, SVector(lower(L)), SVector(upper(L)); kwargs...)
function iterated_integration_(f, L::IntegrationLimits; callback=thunk, kwargs...)
    hcubature(SVector(lower(L)), SVector(upper(L)); kwargs...) do x
        g = callback(f, x)
        L′ = L
        first(iterated_integration_(g, L′(x); callback=callback, kwargs...))
    end
end
#= replacements with other quadrature routines
iterated_integration_(f, L::IntegrationLimits{1}; callback=nothing, kwargs...) = hquadrature(f, lower(L), upper(L); kwargs...)
function iterated_integration_(f, L::IntegrationLimits; callback=thunk, kwargs...) where {N}
    hquadrature(lower(L), upper(L); kwargs...) do x
        g = callback(f, x)
        L′ = L
        first(iterated_integration_(g, L′(x); callback=callback, kwargs...))
    end
end

iterated_integration_(f, L::IntegrationLimits{1}; callback=nothing, kwargs...) = quadgk(f, lower(L), upper(L); kwargs...)
function iterated_integration_(f, L::IntegrationLimits; callback=thunk, kwargs...) where {N}
    quadgk(lower(L), upper(L); kwargs...) do x
        g = callback(f, x)
        L′ = L
        first(iterated_integration_(g, L′(x); callback=callback, kwargs...))
    end
end
=#