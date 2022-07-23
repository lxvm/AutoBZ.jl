export TetrahedralLimits, FBZOpticalConductivity

"""
    TetrahedralLimits(d, s)

A parametrization of the integration limits for a tetrahedron generated from the
automorphism group of the cube on [0,s]^d
"""
struct TetrahedralLimits{d,N} <: IntegrationLimits{d}
    x::SVector{N,Float64}
    s::Float64
    TetrahedralLimits{d}(x::SVector{N}, s) where {d,N} = new{d,N}(x, s)
end
TetrahedralLimits(d::Int, s) = TetrahedralLimits{d}(SVector{0,Float64}(), s)
(l::TetrahedralLimits{d,N})(x::Number) where {d,N} = TetrahedralLimits{d-1}(vcat(x, l.x), l.s)

lower(::TetrahedralLimits) = 0.0
upper(l::TetrahedralLimits{d,0}) where {d} = 0.5l.s
upper(l::TetrahedralLimits) = first(l.x)
rescale(::TetrahedralLimits{d}) where {d} = 2^d * factorial(d)

"""
    FBZOpticalConductivity

These limits are designed for integrating over the cubic FBZ first, then over ω
restricted to the domain where the fermi window is not converging to zero
"""
struct FBZOpticalConductivity{d,d_,Tl,Tu} <: IntegrationLimits{d}
    c::CubicLimits{d_,Tl,Tu}
    Ω::Float64
    β::Float64
    μ::Float64
    cutoff::Float64
    function FBZOpticalConductivity(c::CubicLimits{d_,Tl,Tu}, Ω, β, μ, cutoff) where {d_,Tl,Tu}
        new{d_+1,d_,Tl,Tu}(c, Ω, β, μ, cutoff)
    end
end 

lower(l::FBZOpticalConductivity) = lower(l.c)
lower(l::FBZOpticalConductivity{1}) = _lower_limit_fermi_window(l.Ω, l.β, l.μ, l.cutoff)
upper(l::FBZOpticalConductivity) = upper(l.c)
upper(l::FBZOpticalConductivity{1}) = _upper_limit_fermi_window(l.Ω, l.β, l.μ, l.cutoff)
rescale(l::FBZOpticalConductivity) = rescale(l.c)
(l::FBZOpticalConductivity)(x::Number) = FBZOpticalConductivity(l.c(x), l.Ω, l.β, l.μ, l.cutoff)

_lower_limit_fermi_window(Ω, β, μ, cutoff) = μ-Ω-10inv(β)
_upper_limit_fermi_window(Ω, β, μ, cutoff) = μ+10inv(β)