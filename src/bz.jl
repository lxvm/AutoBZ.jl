export AbstractBZ, FullBZ, IrreducibleBZ

abstract type AbstractBZ{d} <: IntegrationLimits{d} end

struct FullBZ{d} <: AbstractBZ{d}
    a::Vector{SVector{d,Float64}}
    b::Vector{SVector{d,Float64}}
end

(f::FullBZ)(x) = f
function (l::CompositeLimits{d,T,<:Tuple{FullBZ{dBZ},Vararg{IntegrationLimits}}})(x, dim) where {d,T,dBZ}
    if 0 < dim <= (d-dBZ+1)
        return CompositeLimits{d-dBZ,T}(Base.rest(l.lims, 2)...)
    elseif (d-dBZ+1) < dim <= d
        return l
    else
        throw(ErrorException("skipped FBZ"))
    end
end
function limits(f::FullBZ{d}, dim) where d
    1 <= dim <= d || throw(ArgumentError("pick dim=$(dim) in 1:$d"))
    @inbounds b = norm(f.b[dim])
    half_b = b/2
    return (-half_b, half_b)
end
box(f::FullBZ{d}) where d = [limits(f,dim) for dim in 1:d]

"""
    nsyms(::FullBZ)

Returns 1 because the only symmetry applied to the cube is the identity.
"""
nsyms(::FullBZ) = 1

"""
    symmetries(::FullBZ)

Return an identity matrix.
"""
symmetries(::FullBZ) = tuple(I)
Base.eltype(::Type{FullBZ{d}}) where d = Float64

struct IrreducibleBZ{d,L<:IntegrationLimits{d},d_} <: AbstractBZ{d}
    a::Vector{SVector{d_,Float64}}
    b::Vector{SVector{d_,Float64}}
    lims::L
end

(l::IrreducibleBZ)(x) = IrreducibleBZ(l.a, l.b, l.lims(x))
limits(l::IrreducibleBZ, dim) = limits(l.lims, dim)
box(l::IrreducibleBZ) = box(l.lims)
nsyms(l::IrreducibleBZ) = nsyms(l.lims)
symmetries(l::IrreducibleBZ) = symmetries(l.lims)
Base.eltype(::Type{IrreducibleBZ{d,L,d_}}) where {d,L,d_} = eltype(L)

struct AffineTransform{d,T,L}
    R::SMatrix{d,d,T,L}
    t::SVector{d,T}
    AffineTransform{d,T}(R, t) where {d,T} = new{d,T,d^2}(R,t)
end
AffineTransform(R::SMatrix{d,d,T}, t::SVector{d,T}) where {d,T} = AffineTransform{d,T}(R,t)
