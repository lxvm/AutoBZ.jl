export tree_integration, iterated_integration

tree_integration(f, a, b; kwargs...) = hcubature(f, a, b; kwargs...)

iterated_integration(f, a, b; kwargs...) = iterated_integration(f, SVector(a), SVector(b); kwargs...)
iterated_integration(f, a::SVector{1}, b::SVector{1}; kwargs...) = hcubature(f, a, b; kwargs...)
function iterated_integration(f, a::SVector, b::SVector; kwargs...)
    hcubature(SVector(last(a)), SVector(last(b)); kwargs...) do x
        g = contract(f, x)
        first(iterated_integration(g, pop(a), pop(b); kwargs...))
    end
end
#=
iterated_integration(f, a::SVector{1}, b::SVector{1}; kwargs...) = hquadrature(f, first(a), first(b); kwargs...)
function iterated_integration(f, a::SVector, b::SVector; kwargs...)
    hquadrature(last(a), last(b); kwargs...) do x
        g = contract(f, x)
        first(iterated_integration(g, pop(a), pop(b); kwargs...))
    end
end

iterated_integration(f, a::SVector{1}, b::SVector{1}; kwargs...) = quadgk(f, first(a), first(b); kwargs...)
function iterated_integration(f, a::SVector, b::SVector; kwargs...)
    quadgk(last(a), last(b); kwargs...) do x
        g = contract(f, x)
        first(iterated_integration(g, pop(a), pop(b); kwargs...))
    end
end
=#
function iterated_integration(f, L::IntegrationLimits; kwargs...)
    int, err = _iterated_integration(f, L; kwargs...)
    rescale(L)*int, err
end
_iterated_integration(f::Integrand{1}, L::IntegrationLimits; kwargs...) = hcubature(f, SVector(lower(L)), SVector(upper(L)); kwargs...)
function _iterated_integration(f, L::IntegrationLimits; kwargs...)
    hcubature(SVector(lower(L)), SVector(upper(L)); kwargs...) do x
        g = contract(f, first(x))
        L′ = L
        first(_iterated_integration(g, L′(first(x)); kwargs...))
    end
end