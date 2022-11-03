export tree_integration, iterated_integration,
    iterated_tol_update, iterated_pre_eval

"""
    tree_integration(f, a, b)
    tree_integration(f, ::CubicLimits)

Calls `HCubature` to perform multi-dimensional integration of `f` over a cube.
"""
tree_integration(f, a, b; callback=nothing, kwargs...) = hcubature(f, a, b; kwargs...)
tree_integration(f, c::CubicLimits; kwargs...) = tree_integration(f, c.l, c.u; kwargs...)

"""
    iterated_tol_update(f, l, atol, rtol)

Choose a new set of error tolerances for the next inner integral. By default
returns `(atol, rtol)` unchanged.
"""
iterated_tol_update(f, l, atol, rtol) = (atol, rtol)

"""
    iterated_pre_eval(f, x)

Perform a precomputation
"""
iterated_pre_eval(f, x) = thunk(f, x)

"""
    iterated_integration(f, ::IntegrationLimits)

Calls `HCubature` to perform iterated 1D integration of `f` over a domain
parametrized by `IntegrationLimits`.
"""
iterated_integration(f, a, b; kwargs...) = iterated_integration(f, CubicLimits(a, b); kwargs...)
function iterated_integration(f, l::IntegrationLimits;atol=0.0, rtol=sqrt(eps()), kwargs...)
    int, err = iterated_integration_(f, l, atol/nsyms(l), rtol; kwargs...)
    symmetrize(l, int, err)
end

function iterated_integration_(f, l::IntegrationLimits{1}, atol, rtol; kwargs...) 
    hcubature(f, SVector(lower(l)), SVector(upper(l)); atol=atol, rtol=rtol, kwargs...)
end
function iterated_integration_(f, l::IntegrationLimits, atol, rtol; kwargs...)
    hcubature(SVector(lower(l)), SVector(upper(l)); atol=atol, rtol=rtol, kwargs...) do x
        first(iterated_integration_(iterated_pre_eval(f, x), l(x), iterated_tol_update(f, l, atol, rtol)...; kwargs...))
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