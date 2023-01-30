"""
    AbstractBZ{d,T,basis}

Abstract supertype for Brillouin zones of dimension `d` and domain type `T` in
the basis `:Cartesian` or `:lattice`.
"""
abstract type AbstractBZ{d,T,basis} end

# interface
function symmetries end
function nsyms end
function domain_type end
function limits end

# TODO: incorporate rotations to Cartesian basis due to lattice vectors 
"""
    symmetrize(::AbstractBZ, x)
    symmetrize(::AbstractBZ, xs...)

Tranform `x` by the symmetries of the parametrization used to reduce the
domain, thus mapping the value of `x` on the parametrization to the full domain.
When the integrand is a scalar, this is equal to `nsyms(l)*x`.
When the integrand is a vector, this is `sum(S*x for S in symmetries(l))`.
When the integrand is a matrix, this is `sum(S*x*S' for S in symmetries(l))`.
"""
symmetrize(l::AbstractBZ, xs...) = map(x -> symmetrize(l, x), xs)
symmetrize(l::AbstractBZ, x) = symmetrize_(x, nsyms(l), symmetries(l))
symmetrize_(x::Number, nsym, syms) = nsym*x
symmetrize_(x::AbstractArray{<:Any,0}, nsym, syms) = symmetrize_(only(x), nsym, syms)
function symmetrize_(x::AbstractVector, nsym, syms)
    r = zero(x)
    for S in syms
        r += S * x
    end
    r
end
function symmetrize_(x::AbstractMatrix, nsym, syms)
    r = zero(x)
    for S in syms
        r += S * x * S'
    end
    r
end

"""
    vol(::AbstractLimits)

Return the volume of the full domain without the symmetries applied
"""
vol(bz::AbstractBZ) = det(bz.B)
# vol(bz::AbstractBZ) = prod(u-l for (l, u) in boundingbox(bz))

"""
    boundingbox(::AbstractBZ)

Return a tuple of the endpoints of the BZ in each lattice coordinate
"""
function boundingbox(bz::AbstractBZ)
    c = limits(convert(FullBZ{basis(bz)}, bz))
    ntuple(i -> limits(c, i),  ndims(bz))
end

# abstract methods
Base.ndims(::AbstractBZ{d}) where d = d
domain_type(::Type{<:AbstractBZ{d,T}}) where {d,T} = T
basis(::AbstractBZ{d,T,b}) where {d,T,b} = b

# utilities
function basis_to_limits(B::SMatrix{d,d}) where d
    half_b = SVector{d}(ntuple(n -> norm(B[:,n])/2, Val{d}()))
    CubicLimits(-half_b, half_b)
end
function check_bases_canonical(A::AbstractMatrix, B::AbstractMatrix, atol)
    norm(A'B - 2pi*I) < atol || throw("Real and reciprocal Bravais lattice bases non-orthogonal to tolerance $atol")
end
canonical_reciprocal_basis(A::T) where {T<:AbstractMatrix} = A' \ (2pi*one(T))


# implementations

struct FullBZ{basis,d,T,L,C<:CubicLimits{d},Td} <: AbstractBZ{d,Td,basis}
    A::SMatrix{d,d,T,L}
    B::SMatrix{d,d,T,L}
    lims::C
    FullBZ{basis}(A::M, B::M, lims::C) where {basis,d,T,L,M<:SMatrix{d,d,T,L},C<:CubicLimits} =
        new{basis,d,T,L,C,domain_type(lims)}(A, B, lims)
end
function FullBZ(A::SMatrix{d,d,T}, B::SMatrix{d,d,T}; atol=sqrt(eps()), basis=:lattice) where {d,T}
    check_bases_canonical(A, B, atol)
    FullBZ{basis}(A, B, basis_to_limits(B))
end
FullBZ(A; kwargs...) = FullBZ(A, canonical_reciprocal_basis(A); kwargs...)

nsyms(::FullBZ) = 1
symmetries(::FullBZ) = tuple(I)
limits(bz::FullBZ) = bz.lims
Base.convert(::Type{<:FullBZ}, fbz::FullBZ) = fbz

"""
    IrreducibleBZ(A, B, lims, syms)

Data type representing a Brillouin zone reduced by a set of symmetries, with
integration limits `lims`
"""
struct IrreducibleBZ{basis,d,L<:AbstractLimits{d},S,T,d2,Td} <: AbstractBZ{d,Td,basis}
    A::SMatrix{d,d,T,d2}
    B::SMatrix{d,d,T,d2}
    lims::L
    syms::S
    IrreducibleBZ{basis}(A::M, B::M, lims::L, syms::S) where {basis,d,T,d2,M<:SMatrix{d,d,T,d2},L<:AbstractLimits,S} =
        new{basis,d,L,S,T,d2,domain_type(lims)}(A, B, lims, syms)
end

function IrreducibleBZ(A::SMatrix{d,d,T}, B::SMatrix{d,d,T}, lims, syms::Vector; atol=sqrt(eps()), basis=:lattice) where {d,T}
    check_bases_canonical(A, B, atol)
    IrreducibleBZ{basis}(A,B,lims,syms)
end

nsyms(l::IrreducibleBZ) = length(l.syms)
symmetries(l::IrreducibleBZ) = l.syms
limits(bz::IrreducibleBZ) = bz.lims
Base.convert(::Type{<:FullBZ{basis}}, ibz::IrreducibleBZ) where basis = FullBZ{basis}(ibz.A, ibz.B, basis_to_limits(ibz.B))



# iterated integration customizations

function iterated_integration(f, bz::AbstractBZ; atol=nothing, kwargs...)
    atol = something(atol, zero(domain_type(bz)))/nsyms(bz) # rescaling by symmetries
    int, err = iterated_integration(f, limits(bz); atol=atol, kwargs...)
    symmetrize(bz, int, err)
end
function iterated_integration_kwargs(f, bz::AbstractBZ; atol=nothing, kwargs...)
    atol = something(atol, zero(domain_type(bz)))/nsyms(bz) # rescaling by symmetries
    iterated_integration_kwargs(f, limits(bz); atol=atol, kwargs...)
end
iterated_integral_type(f, bz::AbstractBZ) = iterated_integral_type(f, limits(bz))
iterated_inference(f, bz::AbstractBZ) = iterated_inference(f, limits(bz))
alloc_segbufs(f, bz::AbstractBZ) = alloc_segbufs(f, limits(bz))
