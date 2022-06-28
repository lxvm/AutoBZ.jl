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