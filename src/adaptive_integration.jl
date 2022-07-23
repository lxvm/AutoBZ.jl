export tree_integration, iterated_integration

tree_integration(f, a, b; callback=nothing, kwargs...) = hcubature(f, a, b; kwargs...)

"""
Accepts a callback
"""
iterated_integration(f, a, b; kwargs...) = iterated_integration(f, CubicLimits(a, b); kwargs...)
function iterated_integration(f, L::IntegrationLimits; kwargs...)
    int, err = _iterated_integration(f, L; kwargs...)
    rescale(L)*int, err
end

_iterated_integration(f, L::IntegrationLimits{1}; callback=nothing, kwargs...) = hcubature(f, SVector(lower(L)), SVector(upper(L)); kwargs...)
function _iterated_integration(f, L::IntegrationLimits; callback=thunk, kwargs...)
    hcubature(SVector(lower(L)), SVector(upper(L)); kwargs...) do x
        g = callback(f, x)
        L′ = L
        first(_iterated_integration(g, L′(x); callback=callback, kwargs...))
    end
end
#= replacements with other quadrature routines
_iterated_integration(f, L::IntegrationLimits{1}; callback=nothing, kwargs...) = hquadrature(f, lower(L), upper(L); kwargs...)
function _iterated_integration(f, L::IntegrationLimits; callback=thunk, kwargs...) where {N}
    hquadrature(lower(L), upper(L); kwargs...) do x
        g = callback(f, x)
        L′ = L
        first(_iterated_integration(g, L′(x); callback=callback, kwargs...))
    end
end

_iterated_integration(f, L::IntegrationLimits{1}; callback=nothing, kwargs...) = quadgk(f, lower(L), upper(L); kwargs...)
function _iterated_integration(f, L::IntegrationLimits; callback=thunk, kwargs...) where {N}
    quadgk(lower(L), upper(L); kwargs...) do x
        g = callback(f, x)
        L′ = L
        first(_iterated_integration(g, L′(x); callback=callback, kwargs...))
    end
end
=#