export lower, upper, IntegrationLimits, CubicLimits, TetrahedralLimits

abstract type IntegrationLimits end

"""
    CubicLimits{N,T}
A wrapper for `SVector{N,T}` that contains integration limit information for a cube
"""
struct CubicLimits{N,T} <: IntegrationLimits
    x::SVector{N,T}
end
CubicLimits() = CubicLimits(SVector())
(l::CubicLimits)(x::Number) = CubicLimits(SVector(promote(x, l.x...)))

lower(l::CubicLimits{0}) = 0.0
lower(l::CubicLimits{1}) = 0.0
lower(l::CubicLimits{2}) = 0.0
upper(l::CubicLimits{0}) = 1.0
upper(l::CubicLimits{1}) = 1.0
upper(l::CubicLimits{2}) = 1.0

"""
    TetrahedralLimits{N,T}
A wrapper for `SVector{N,T}` that contains integration limit information for a tetrahedron
"""
struct TetrahedralLimits{N,T} <: IntegrationLimits
    x::SVector{N,T}
end
TetrahedralLimits() = TetrahedralLimits(SVector()) 
(l::TetrahedralLimits)(x::Number) = TetrahedralLimits(SVector(promote(x, l.x...)))

lower(l::TetrahedralLimits{0}) = 0.0
lower(l::TetrahedralLimits{1}) = 0.0
lower(l::TetrahedralLimits{2}) = 0.0
upper(l::TetrahedralLimits{0}) = 0.5
upper(l::TetrahedralLimits{1}) = last(l.x)
upper(l::TetrahedralLimits{2}) = first(l.x)

iterated_integration(f::Union{DOSIntegrand{1},FourierSeries{1}}, L::IntegrationLimits; kwargs...) = hcubature(f, SVector(lower(L)), SVector(upper(L)); kwargs...)
function iterated_integration(f, L::IntegrationLimits; kwargs...)
    hcubature(SVector(lower(L)), SVector(upper(L)); kwargs...) do x
        g = contract(f, first(x))
        L′ = L
        first(iterated_integration(g, L′(first(x)); kwargs...))
    end
end