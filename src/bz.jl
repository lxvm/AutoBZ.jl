export AbstractBZ, FullBZ, IrreducibleBZ

abstract type AbstractBZ{d} end

# interface
function symmetries end
function nsyms end
function domain_type end
function limits end

# abstract methods
Base.ndims(::AbstractBZ{d}) where d = d

# implementations
struct FullBZ{d,T,L,C<:CubicLimits{d}} <: AbstractBZ{d}
    A::SMatrix{d,d,T,L}
    B::SMatrix{d,d,T,L}
    lims::C
end
function FullBZ(A::SMatrix{d,d,T}, B::SMatrix{d,d,T}; atol=sqrt(eps()))
    norm(A'B - 2pi*I) < atol || throw("Real and reciprocal Bravais lattice non-orthogonal to tolerance $atol")
    half_b = SVector{d}(ntuple(n -> norm(fbz.B[:,n])/2, Val{d}()))
    lims = CubicLimits(-half_b, half_b)
    FullBZ(A, B, lims)
end

nsyms(::FullBZ) = 1
symmetries(::FullBZ) = tuple(I)
domain_type(::Type{FullBZ{d,T,L,C}}) where {d,T,L,C} = domain_type(C)
limits(bz::FullBZ) = bz.lims

struct IrreducibleBZ{d,L<:IntegrationLimits{d},S,T,d2} <: AbstractBZ{d}
    A::SMatrix{d,d,T,d2}
    B::SMatrix{d,d,T,d2}
    lims::L
    syms::S
    function IrreducibleBZ(A::SMatrix{d,d,T}, B::SMatrix{d,d,T}, lims::L, syms::S; atol=sqrt(eps())) where {d,T,L,S}
        norm(A'B - 2pi*I) < atol || throw("Real and reciprocal Bravais lattice non-orthogonal to tolerance $atol")
        new{d,L,S,T,d2}(A,B,lims,syms)
    end
end

nsyms(l::IrreducibleBZ) = length(l.syms)
symmetries(l::IrreducibleBZ) = l.syms
domain_type(::Type{IrreducibleBZ{d,L,S,T,d2}}) where {d,L,S,T,d2} = domain_type(L)
limits(bz::IrreducibleBZ) = bz.lims



# iterated integration methods

function iterated_integration(f, bz::AbstractBZ; atol=nothing, kwargs...)
    atol = something(atol, zero(domain_type(bz)))/nsyms(bz) # rescaling by symmetries
    int, err = iterated_integration(f, limits(bz); atol=atol, kwargs...)
    symmetrize(bz, int, err)
end
alloc_segbufs(f, bz::AbstractBZ) = alloc_segbufs(f, limits(bz))
