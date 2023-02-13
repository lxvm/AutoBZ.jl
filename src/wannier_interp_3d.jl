export AbstractWannierInterp3D, hamiltonian, gauge, to_gauge, shift!
export Hamiltonian3D

"""
    AbstractWannierInterp{gauge,N} <: AbstractInplaceFourierSeries{N}

An abstract subtype of `AbstractInplaceFourierSeries` representing in-place
Fourier series evaluators for Wannier-interpolated quantities with a choice of
basis, or `gauge`.
"""
abstract type AbstractWannierInterp{gauge,N,T} <: AbstractInplaceFourierSeries{N,T} end

"""
    hamiltonian(::AbstractWannierInterp)

Return the Hamiltonian object used for Wannier interpolation
"""
function hamiltonian end

gauge(::AbstractWannierInterp{G}) where G = G


"""
    to_gauge(::Val{gauge}, H) where gauge

Transform the Hamiltonian according to the following values of `gauge`
- `:Wannier`: keeps `H, vs` in the original, orbital basis
- `:Hamiltonian`: diagonalizes `H` and rotates `H` into the energy, band basis
"""
to_gauge(::Val{:Wannier}, H::AbstractMatrix) = H
function to_gauge(::Val{:Wannier}, E::Eigen)
    H, U = E
    U * Diagonal(H) * U'
end
function to_gauge(::Val{:Hamiltonian}, H::AbstractMatrix)
    ishermitian(H) || throw(ArgumentError("found non-Hermitian Hamiltonian"))
    eigen(Hermitian(H)) # need to wrap with Hermitian for type stability
end

"""
    shift!(::AbstractWannierInterp, λ)


"""
function shift!(w::AbstractWannierInterp, λ)
    shift!(hamiltonian(w), λ)
    return HV
end

# TODO define a mechanism for converting gauge 

"""
    Hamiltonian3D(coeffs::Array{T,3}; period=(1.0, 1.0, 1.0), gauge=:Wannier)

This type is an `AbstractFourierSeries{3}` designed for in-place evaluation of
`FourierSeries`, and unlike `FourierSeries` is specialized for 3D Fourier series
and does not allocate a new array every time `contract` is called on it. This
type stores the intermediate arrays used in a calculation and assumes that the
size of `coeffs` on each axis is odd because it treats the zeroth harmonic as
the center of the array (i.e. `(size(coeffs) .÷ 2) .+ 1`).
"""
struct Hamiltonian3D{gauge,T,TH} <: AbstractWannierInterp{gauge,3,T}
    period::NTuple{3,Float64}
    h3::Array{T,3}
    h2::Array{T,2}
    h1::Array{T,1}
    h0::Array{TH,0}
end

function Hamiltonian3D(h3::Array{T,3}; period=(1.0,1.0,1.0), gauge=:Wannier) where T
    @assert all(map(isodd, size(h3)))
    h2 = Array{T,2}(undef, size(h3,1), size(h3, 2))
    h1 = Array{T,1}(undef, size(h3,1))
    TH = Base.promote_op(to_gauge, Val{gauge}, T)
    h0 = Array{TH,0}(undef)
    Hamiltonian3D{Val(gauge),T,TH}(period, h3, h2, h1, h0)
end

#=
AutoBZ.period(H::Hamiltonian3D) = H.period
AutoBZ .value(H::Hamiltonian3D) = only(H.h0)


@generated AutoBZ.contract!(H::Hamiltonian3D, x::Number, ::Type{Val{d}}) where d =
    :(fourier_kernel!(H.h$(d-1), H.h$d, x, inv(H.period[$d])); return H)
AutoBZ.contract!(H::Hamiltonian3D, x::Number, ::Type{Val{1}}) =
    (H.h0[] = H(x); return H)
    (H::Hamiltonian3D{gauge})(x::Number) where gauge =
    to_gauge(gauge, fourier_kernel(H.h1, x, inv(H.period[1])))
    =#
    
"""
    shift!(H::Hamiltonian3D, λ::Number)

Modifies and returns `H` such that it returns `H - λ*I`.
"""
function shift!(H::Hamiltonian3D{T}, λ_::Number) where T
    λ = convert(eltype(T), λ_)
    i = div.(size(H.h3), 2) .+ 1
    H.coeffs[i...] -= λ*I
    return H
end
