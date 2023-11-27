#=
    DEFINITIONS FOR SELF ENERGY INTERPOLANTS
=#

"""
    AbstractSelfEnergy

An abstract type whose instances implement the following interface:
- instances are callable and return a square matrix as a function of frequency
- instances have methods `lb` and `ub` that return the lower and upper bounds of
  of the frequency domain for which the instance can be evaluated
"""
abstract type AbstractSelfEnergy end

"""
    lb(::AbstractSelfEnergy)

Return the greatest lower bound of the domain of the self energy evaluator
"""
function lb end


"""
    ub(::AbstractSelfEnergy)

Return the least upper bound of the domain of the self energy evaluator
"""
function ub end

#=
    DEFINITONS FOR WANNIER INTERPOLANTS
=#

"""
    AbstractWannierInterp{N,T,iip} <: AbstractFourierSeries{N,T,iip}

Abstract supertype for all Wannier-interpolated quantities in `AutoBZ`
"""
abstract type AbstractWannierInterp{N,T,iip} <: AbstractFourierSeries{N,T,iip} end

# by just evaluating the interpolant we simplify type inference
function fourier_type(f::AbstractWannierInterp, ::Type{T}) where {T}
    return typeof(f(fill(oneunit(eltype(T)), SVector{ndims(f),eltype(T)})))
end

#=
    DEFINITIONS FOR INTERPOLATED TENSOR OPERATORS WITH BAND/ORBITAL INDICES
=#

"""
    AbstractGauge

Abstract supertype of gauges (or bases) for orbital/band indices
"""
abstract type AbstractGauge end

"""
    Wannier <: AbstractGauge

Singleton type representing the Wannier gauge
"""
struct Wannier <: AbstractGauge end

"""
    Hamiltonian <: AbstractGauge

Singleton type representing the Hamiltonian gauge
"""
struct Hamiltonian <: AbstractGauge end

"""
    to_gauge(::AbstractGauge, h) where gauge

Transform the Hamiltonian according to the following values of `gauge`
- [`Wannier`](@ref): keeps `h, vs` in the original, orbital basis
- [`Hamiltonian`](@ref): diagonalizes `h` and rotates `h` into the energy, band basis
"""
to_gauge(::Wannier, H) = H
to_gauge(::Wannier, (h,U)::Eigen) = U * Diagonal(h) * U'

function to_gauge(::Hamiltonian, H::AbstractMatrix)
    # this test will fail if we do contour deformation. It would be better to check the
    # types of the inputs are real instead, but we don't have access to the FourierValue

    # It would be better to fix this ambiguity by requiring the series evaluator return a
    # Hermitian-wrapped array
    isapproxhermitian(H, atol=real(oneunit(eltype(H)))/10^6) || throw(ArgumentError("found non-Hermitian Hamiltonian"))
    _eigen(Hermitian(H)) # need to wrap with Hermitian for type stability
end


"""
    AbstractGaugeInterp{G,N,T,iip} <: AbstractWannierInterp{N,T,iip}

An abstract subtype of `AbstractFourierSeries` representing in-place
Fourier series evaluators for Wannier-interpolated quantities with a choice of
basis, or gauge, `G`, which is typically [`Hamiltonian`](@ref) or [`Wannier`](@ref).
For details, see [`to_gauge`](@ref).
"""
abstract type AbstractGaugeInterp{G,N,T,iip} <: AbstractWannierInterp{N,T,iip} end

gauge(::AbstractGaugeInterp{G}) where G = G

"""
    GaugeDefault(::Type{T})::AbstractCoordinate where T

[`AbstractGaugeInterp`](@ref)s should define this trait to declare the
gauge that they assume their data is in.
"""
GaugeDefault(::T) where {T<:AbstractGaugeInterp} = GaugeDefault(T)


show_details(w::AbstractGaugeInterp) =
    " in $(gauge(w)) gauge"

to_gauge(gi::AbstractGaugeInterp, w) = to_gauge(gauge(gi), w)

"""
    AbstractHamiltonianInterp{G,N,T,iip} <: AbstractGaugeInterp{G,N,T,iip}

Abstract type representing Hamiltonians, which are matrix-valued Hermitian Fourier series.
They should also have period 1, but produce derivatives with wavenumber 1.
"""
abstract type AbstractHamiltonianInterp{G,N,T,iip} <: AbstractGaugeInterp{G,N,T,iip} end

"""
    parentseries(::AbstractHamiltonianInterp)::FourierSeries

Return the Fourier series that the Hamiltonian wraps
"""
parentseries(::AbstractHamiltonianInterp)

show_dims(h::AbstractHamiltonianInterp) = show_dims(parentseries(h))

function shift!(f::FourierSeries, λ_::Number)
    λ = convert(eltype(eltype(f.c)), λ_)
    idx = map((i,o) -> i-o-1, first(CartesianIndices(f.c)).I, f.o)
    f.c[idx...] -= λ*I
    return f
end

"""
    shift!(h::AbstractHamiltonianInterp, λ::Number)

Modifies and returns `h` such that it returns `h - λ*I`. Will throw a
`BoundsError` if this operation cannot be done on the existing data.
"""
function shift!(h::AbstractHamiltonianInterp, λ::Number)
    shift!(parentseries(h), λ)
    return h
end

#=
    DEFINITIONS FOR INTERPOLATED TENSOR OPERATORS WITH COORDINATE INDICES
=#

"""
    AbstractCoordinate

Abstract supertype of bases for coordinate/spatial indices
"""
abstract type AbstractCoordinate end

"""
    Lattice <: AbstractCoordinate

Singleton type representing lattice coordinates.
"""
struct Lattice <: AbstractCoordinate end

"""
    Cartesian <: AbstractCoordinate

Singleton type representing Cartesian coordinates.
"""
struct Cartesian <: AbstractCoordinate end

"""
    LatticeRep()

Symmetry representation of objects that transform under the group action in the
same way as the lattice.
"""
struct LatticeRep <: AbstractSymRep end

"""
    CartesianRep()

Symmetry representation of objects that transform under the group action in the
same way as the lattice and in Cartesian coordinates.
"""
struct CartesianRep <: AbstractSymRep end

function symmetrize_(::LatticeRep, bz::SymmetricBZ, x::AbstractVector)
    r = zero(x)
    for S in bz.syms
        r += S * x
    end
    r
end
function symmetrize_(::LatticeRep, bz::SymmetricBZ, x::AbstractMatrix)
    r = zero(x)
    for S in bz.syms
        r += S * x * S'
    end
    r
end
function symmetrize_(::CartesianRep, bz::SymmetricBZ, x::AbstractVector)
    B = bz.A'; invB = _inv(B)
    invB * symmetrize_(LatticeRep(), bz, B * x)
end
function symmetrize_(::CartesianRep, bz::SymmetricBZ, x::AbstractMatrix)
    B = bz.A'; invB = _inv(B)
    invB * symmetrize_(LatticeRep(), bz, B * x * transpose(B)) * transpose(invB)
end

coord_to_rep(::Lattice) = LatticeRep()
coord_to_rep(::Cartesian) = CartesianRep()

"""
    to_coord(B::AbstractCoordinate, D::AbstractCoordinate, A, vs)

If `B` and `D` are the same type return `vs`, however and if they differ return `A*vs`.
"""
to_coord(::T, ::T, A, vs) where {T<:AbstractCoordinate} = vs
to_coord(::AbstractCoordinate, ::AbstractCoordinate, A, vs) = to_coord_(A, vs)
to_coord_(A, vs::AbstractVector) = A * vs
to_coord_(A, vs::AbstractMatrix) = A * vs * transpose(A) # would be nice if this kept vs Symmetric

"""
    AbstractCoordInterp{B,G,N,T,iip} <:AbstractGaugeInterp{G,N,T,iip}

An abstract subtype of `AbstractGaugeInterp` also containing information about
the coordinate basis `B`, which is either [`Lattice`](@ref) or
[`Cartesian`](@ref). For details see [`to_coord`](@ref) and
[`CoordDefault`](@ref).
"""
abstract type AbstractCoordInterp{B,G,N,T,iip} <:AbstractGaugeInterp{G,N,T,iip} end

"""
    CoordDefault(::Type{T})::AbstractCoordinate where T

[`AbstractCoordInterp`](@ref)s should define this trait to declare the
coordinate basis where they assume their data is in.
"""
CoordDefault(::T) where {T<:AbstractCoordInterp} = CoordDefault(T)

"""
    coord(::AbstractCoordInterp{B})::AbstractCoordinate where B = B

Return the [`AbstractCoordinate`](@ref) basis in which an
[`AbstractCoordInterp`](@ref) will be evaluated.
"""
coord(::AbstractCoordInterp{B}) where B = B

coord_to_rep(ci::AbstractCoordInterp) = coord_to_rep(coord(ci))
to_coord(ci::AbstractCoordInterp, A, x) =
    to_coord(coord(ci), CoordDefault(ci), A, x)

show_details(v::AbstractCoordInterp) =
    " in $(gauge(v)) gauge and $(coord(v)) coordinates"

#=
    DEFINITIONS FOR INTERPOLATED VELOCITY OPERATORS
=#

"""
    AbstractVelocityComponent

Abstract supertype representing velocity components.
"""
abstract type AbstractVelocityComponent end

"""
    Whole <: AbstractVelocityComponent

Singleton type representing whole velocities
"""
struct Whole <: AbstractVelocityComponent end

"""
    Inter <: AbstractVelocityComponent

Singleton type representing interband velocities
"""
struct Inter <: AbstractVelocityComponent end

"""
    Intra <: AbstractVelocityComponent

Singleton type representing intraband velocities
"""
struct Intra <: AbstractVelocityComponent end

to_vcomp_gauge_mass(::Whole, ::Wannier, H, vs::NTuple, ms::NTuple) = (H, vs, ms)
function to_vcomp_gauge_mass(vcomp::C, g::Wannier, H, vs::NTuple, ms::NTuple) where C
    E, vhs = to_vcomp_gauge(vcomp, Hamiltonian(), H, vs)
    return to_gauge(g, E, vhs)..., ms
end

function to_vcomp_gauge_mass(vcomp::C, ::Hamiltonian, H::AbstractMatrix, vws::NTuple{N}, mws::NTuple{M}) where {C,N,M}
    E, vhs, mhs = to_gauge(Hamiltonian(), H, vws, mws)
    return (E, to_vcomp(vcomp, vhs), mhs)
end

"""
    to_vcomp_gauge(::Val{C}, ::Val{G}, h, vs...) where {C,G}

Take the velocity components of `vs` in any gauge according to the value of `C`
- [`Whole`](@ref): return the whole velocity (sum of interband and intraband components)
- [`Intra`](@ref): return the intraband velocity (diagonal in Hamiltonian gauge)
- [`Inter`](@ref): return the interband velocity (off-diagonal terms in Hamiltonian gauge)

Transform the velocities into a gauge according to the following values of `G`
- [`Wannier`](@ref): keeps `H, vs` in the original, orbital basis
- [`Hamiltonian`](@ref): diagonalizes `H` and rotates `H, vs` into the energy, band basis
"""
to_vcomp_gauge

to_vcomp_gauge(::Whole, ::Wannier, H, vs::NTuple) = (H, vs)
function to_vcomp_gauge(vcomp::C, g::Wannier, H, vs::NTuple) where C
    E, vhs = to_vcomp_gauge(vcomp, Hamiltonian(), H, vs)
    to_gauge(g, E, vhs)
end

function to_vcomp_gauge(vcomp::C, ::Hamiltonian, H::AbstractMatrix, vws::NTuple{N}) where {C,N}
    E, vhs = to_gauge(Hamiltonian(), H, vws)
    return (E, to_vcomp(vcomp, vhs))
end

function to_gauge(g::Wannier, H::Eigen, vhs::NTuple{N}) where N
    U = H.vectors
    return (to_gauge(g, H), ntuple(n -> U * vhs[n] * U', Val(N)))
end
function to_gauge(g::Hamiltonian, H::AbstractMatrix, vws::NTuple{N}) where N
    E = to_gauge(g, H)
    U = E.vectors
    return (E, ntuple(n -> U' * vws[n] * U, Val(N)))
end
function to_gauge(g::Hamiltonian, H::AbstractMatrix, vws::NTuple{N}, mws::NTuple{M}) where {N,M}
    E = to_gauge(g, H)
    U = E.vectors
    return (E, ntuple(n -> U' * vws[n] * U, Val(N)), ntuple(n -> U' * mws[n] * U, Val(M)))
end

to_vcomp(::Whole, vhs::NTuple{N}) where {N} = vhs
to_vcomp(::Inter, vhs::NTuple{N}) where {N} =
    ntuple(n -> vhs[n] - Diagonal(vhs[n]), Val(N))
to_vcomp(::Intra, vhs::NTuple{N}) where {N} =
    ntuple(n -> Diagonal(vhs[n]), Val(N))



"""
    AbstractVelocityInterp{C,B,G,N,T,iip} <:AbstractCoordInterp{B,G,N,T,iip}

An abstract subtype of `AbstractCoordInterp` also containing information the
velocity component, `C`, which is typically `Val(:whole)`, `Val(:inter)`, or
`Val(:intra)`. For details see [`to_vcomp_gauge`](@ref).
Since the velocity depends on the Hamiltonian, subtypes should also evaluate the
Hamiltonian.
"""
abstract type AbstractVelocityInterp{C,B,G,N,T,iip} <:AbstractCoordInterp{B,G,N,T,iip} end

vcomp(::AbstractVelocityInterp{C}) where C = C

"""
    VcompDefault(::Type{T})::AbstractVelocityComponent where T

[`AbstractVelocityInterp`](@ref)s should define this trait to declare the
velocity component that they assume their data is in.
"""
VcompDefault(::T) where {T<:AbstractVelocityInterp} = VcompDefault(T)


show_details(v::AbstractVelocityInterp) =
    " in $(gauge(v)) gauge and $(coord(v)) coordinates with $(vcomp(v)) velocities"

to_vcomp_gauge_mass(vi::AbstractVelocityInterp, h, vs::NTuple, ms::NTuple) =
    to_vcomp_gauge_mass(vcomp(vi), gauge(vi), h, vs, ms)

to_vcomp_gauge(vi::AbstractVelocityInterp, h, vs::NTuple) =
    to_vcomp_gauge(vcomp(vi), gauge(vi), h, vs)
function to_vcomp_gauge(vi::AbstractVelocityInterp, h, vs::SVector)
    rh, rvs = to_vcomp_gauge(vcomp(vi), gauge(vi), h, vs.data)
    return rh, SVector(rvs)
end

"""
    parentseries(::AbstractVelocityInterp)::DerivativeSeries

Return the Hamiltonian object used for Wannier interpolation
"""
parentseries(::AbstractVelocityInterp)

parentseries(d::DerivativeSeries{1}) = d.f
parentseries(d::DerivativeSeries) = parentseries(d.f)

function shift!(d::DerivativeSeries, λ)
    shift!(parentseries(d), λ)
    return d
end

"""
    shift!(::AbstractVelocityInterp, λ)

Offset the zero-point energy in a Hamiltonian system by a constant
"""
function shift!(v::AbstractVelocityInterp, λ)
    shift!(parentseries(v), λ)
    return v
end
