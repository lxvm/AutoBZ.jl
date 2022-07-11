export lower, upper, IntegrationLimits, CubicLimits, TetrahedralLimits
# have a separate file for integration limits
# put the non-symmetrization code in other files
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
rescale(::CubicLimits) = 1

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
rescale(::TetrahedralLimits) = 48