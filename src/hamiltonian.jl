export AbstractWannierInterp, hamiltonian, gauge, to_gauge, shift!
export Hamiltonian

"""
    hamiltonian(::AbstractWannierInterp)

Return the Hamiltonian object used for Wannier interpolation
"""
function hamiltonian end

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
    Hamiltonian(coeffs::Array{T,3}; period=(1.0, 1.0, 1.0), gauge=:Wannier)

This type is an `AbstractFourierSeries{3}` designed for in-place evaluation of
`FourierSeries`, and unlike `FourierSeries` is specialized for 3D Fourier series
and does not allocate a new array every time `contract` is called on it. This
type stores the intermediate arrays used in a calculation and assumes that the
size of `coeffs` on each axis is odd because it treats the zeroth harmonic as
the center of the array (i.e. `(size(coeffs) .÷ 2) .+ 1`).
"""
struct Hamiltonian{G,N,T,F} <: AbstractWannierInterp{G,N,T}
    f::F
    Hamiltonian{G}(f::F) where {G,F<:InplaceFourierSeries} =
        new{G,ndims(f),eltype(f),F}(f)
end

Hamiltonian(h; gauge=:Wannier, kwargs...) =
    Hamiltonian{Val(gauge)}(InplaceFourierSeries(h; kwargs...))
    
period(h::Hamiltonian) = period(h.f)

contract!(h::Hamiltonian{G}, x::Number, ::Val{d}) where {G,d} =
    Hamiltonian{G}(contract!(h.f, x, Val(d)))

evaluate(h::Hamiltonian{G,1}, x::NTuple{1}) where G =
    to_gauge(G, evaluate(h.f, x))

"""
    shift!(H::Hamiltonian, λ::Number)

Modifies and returns `H` such that it returns `H - λ*I`.
"""
function shift!(H::Hamiltonian, λ_::Number)
    λ = convert(eltype(eltype(H)), λ_)
    for i in CartesianIndices(H.f.c)
        all(iszero, i.I .+ H.f.o) || continue
        H.f.c[i] -= λ*I
        break
    end
    return H
end
