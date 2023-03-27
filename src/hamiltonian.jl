"""
    HamiltonianInterp(f::InplaceFourierSeries; gauge=:Wannier)

A wrapper for `InplaceFourierSeries` with an additional gauge that allows for
convenient diagonalization of the result. For details see [`to_gauge`](@ref).
"""
struct HamiltonianInterp{G,N,T,F} <: AbstractGaugeInterp{G,N,T}
    f::F
    HamiltonianInterp{G}(f::F) where {G,F<:Union{FourierSeries,InplaceFourierSeries}} =
        new{G,ndims(f),eltype(f),F}(f)
end

# recursively wrap inner Fourier series with Hamiltonian
HamiltonianInterp(f::InplaceFourierSeries; gauge=GaugeDefault(HamiltonianInterp)) =
    HamiltonianInterp{gauge}(f)

GaugeDefault(::Type{<:HamiltonianInterp}) = Wannier()


"""
    shift!(h::HamiltonianInterp, λ::Number)

Modifies and returns `h` such that it returns `h - λ*I`. Will throw a
`BoundsError` if this operation cannot be done on the existing data.
"""
function shift!(h::HamiltonianInterp, λ_::Number)
    λ = convert(eltype(eltype(h)), λ_)
    c = coefficients(h)
    idx = first(CartesianIndices(c)).I .- offset(h.f) .- 1
    h.f.c[idx...] -= λ*I
    return h
end

period(h::HamiltonianInterp) = period(h.f)

contract(h::HamiltonianInterp, x::Number, ::Val{d}) where d =
    HamiltonianInterp{gauge(h)}(contract(h.f, x, Val(d)))

evaluate(h::HamiltonianInterp, x::NTuple{1}) =
    to_gauge(h, evaluate(h.f, x))

coefficients(h::HamiltonianInterp) = coefficients(h.f)
