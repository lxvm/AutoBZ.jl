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

"""
    TetrahedralLimits(a::SVector)
    TetrahedralLimits(a::CubicLimits)
    TetrahedralLimits(a, p)

A parametrization of the integration limits for a tetrahedron generated from the
automorphism group of the hypercube whose corners are `-a*p` and `a*p`. By
default, `p=0.5` which gives integration over the irreducible Brillouin zone of
the cube. If the entries of `a` vary then this implies the hypercube is
rectangular.
"""
struct TetrahedralLimits{d,T<:AbstractFloat} <: IntegrationLimits{d}
    a::SVector{d,T}
    p::T
end
TetrahedralLimits(a::SVector{d,T}) where {d,T<:AbstractFloat} = TetrahedralLimits(a, one(T)/2)
TetrahedralLimits(a::NTuple{d,T}) where {d,T} = TetrahedralLimits(SVector{d,T}(a))
TetrahedralLimits(c::CubicLimits) = TetrahedralLimits(c.u-c.l)
(t::TetrahedralLimits)(x::Number) = TetrahedralLimits(pop(t.a), x/last(t.a))
Base.eltype(::Type{TetrahedralLimits{d,T}}) where {d,T} = T

box(t::TetrahedralLimits{d,T}) where {d,T} = StaticArrays.sacollect(SVector{d,Tuple{T,T}}, (zero(T), a) for a in t.a)

function limits(t::TetrahedralLimits{d}, dim) where d
    dim == d || throw(ArgumentError("limit evaluation not supported for dim=$(dim)"))
    return (zero(t.p), t.p*last(t.a))
end
nsyms(t::TetrahedralLimits) = n_cube_automorphisms(ndims(t))
symmetries(t::TetrahedralLimits) = cube_automorphisms(Val{ndims(t)}())

"""
    cube_automorphisms(d::Integer)

return a generator of the symmetries of the cube in `d` dimensions, optionally
including the identity.
"""
cube_automorphisms(n::Val{d}) where {d} = (S*P for S in sign_flip_matrices(n), P in permutation_matrices(n))
n_cube_automorphisms(d) = n_sign_flips(d) * n_permutations(d)

sign_flip_tuples(n::Val{d}) where {d} = Iterators.product(ntuple(_ -> (1,-1), n)...)
sign_flip_matrices(n::Val{d}) where {d} = (Diagonal(SVector{d,Int}(A)) for A in sign_flip_tuples(n))
n_sign_flips(d::Integer) = 2^d

permutation_matrices(t::Val{n}) where {n} = (StaticArrays.sacollect(SMatrix{n,n,Int,n^2}, ifelse(j == p[i], 1, 0) for i in 1:n, j in 1:n) for p in permutations(ntuple(identity, t)))
n_permutations(n::Integer) = factorial(n)
#= less performant code (at least when n=3)
permutation_matrices(t::Val{n}) where {n} = (SparseArrays.sparse(Base.OneTo(n), p, ones(n), n, n) for p in permutations(ntuple(identity, t)))
permutation_tuples(C::NTuple{N,T}) where {N,T} = @inbounds((C[i], p...)::NTuple{N,T} for i in eachindex(C) for p in permutation_tuples(C[[j for j in eachindex(C) if j != i]]))
permutation_tuples(C::NTuple{1}) = C;
=#