"""
    Hamiltonian(f::InplaceFourierSeries; gauge=:Wannier)

A wrapper for `InplaceFourierSeries` with an additional gauge that allows for
convenient diagonalization of the result. For details see [`to_gauge`](@ref).
"""
struct Hamiltonian{G,N,T,F} <: AbstractWannierInterp{G,N,T}
    f::F
    Hamiltonian{G}(f::F) where {G,F<:InplaceFourierSeries} =
        new{G,ndims(f),eltype(f),F}(f)
end

# recursively wrap inner Fourier series with Hamiltonian
Hamiltonian(f::InplaceFourierSeries; gauge=:Wannier) =
    Hamiltonian{Val(gauge)}(f)


hamiltonian(h::Hamiltonian) = h

"""
    shift!(h::Hamiltonian, λ::Number)

Modifies and returns `h` such that it returns `h - λ*I`. Will throw a
`BoundsError` if this operation cannot be done on the existing data.
"""
function shift!(h::Hamiltonian, λ_::Number)
    λ = convert(eltype(eltype(h)), λ_)
    c = coefficients(h)
    idx = first(CartesianIndices(c)).I .- offset(h.f) .- 1
    h.f.c[idx...] -= λ*I
    return h
end

period(h::Hamiltonian) = period(h.f)

contract(h::Hamiltonian{G}, x::Number, ::Val{d}) where {G,d} =
    Hamiltonian{G}(contract(h.f, x, Val(d)))

evaluate(h::Hamiltonian{G,1}, x::NTuple{1}) where G =
    to_gauge(G, evaluate(h.f, x))

"""
    to_gauge(::Val{gauge}, h) where gauge

Transform the Hamiltonian according to the following values of `gauge`
- `:Wannier`: keeps `h, vs` in the original, orbital basis
- `:Hamiltonian`: diagonalizes `h` and rotates `h` into the energy, band basis
"""
to_gauge(::Val{:Wannier}, H::AbstractMatrix) = H
to_gauge(::Val{:Wannier}, (h,U)::Eigen) = U * Diagonal(h) * U'

function to_gauge(::Val{:Hamiltonian}, H::AbstractMatrix)
    ishermitian(H) || throw(ArgumentError("found non-Hermitian Hamiltonian"))
    eigen(Hermitian(H)) # need to wrap with Hermitian for type stability
end

coefficients(h::Hamiltonian) = coefficients(h.f)
