export Integrand, SpectralFunction, DOSIntegrand

"""
Realizations of this type should implement an `eltype`, be callable with
`AbstractVector` (for evaluation of the FourierSeries at that point) and
`AbstractMatrix` (where the matrix is the pre-evaluated FourierSeries), have a
`contract` method.
"""
abstract type Integrand{N} end

"""
SpectralFunction(ϵ,ω,η,μ)
A function that calculates the imaginary part of the Green's function
"""
struct SpectralFunction{N,T<:FourierSeries{N}} <: Integrand{N}
ϵ::T
ω::Float64
η::Float64
μ::Float64
end

Base.eltype(::Type{<:SpectralFunction{N,T}}) where {N,T} = Base.promote_op(imag, eltype(T))
(f::SpectralFunction)(k::AbstractVector) = imag(hinv(complex(f.ω + f.μ, f.η)*I - f.ϵ(k)))/(-pi)
(f::SpectralFunction)(ϵ::AbstractMatrix) = imag(hinv(complex(f.ω + f.μ, f.η)*I - ϵ))/(-pi)
contract(f::SpectralFunction, x) = SpectralFunction(contract(f.ϵ, x), f.ω, f.η, f.μ)

"""
DOSIntegrand(::SpectralFunction)
A function whose integral gives the density of states.
"""
struct DOSIntegrand{N,T} <: Integrand{N}
A::SpectralFunction{N,T}
end

DOSIntegrand(ϵ, ω, η, μ) = DOSIntegrand(SpectralFunction(ϵ, ω, η, μ))
Base.eltype(::Type{<:DOSIntegrand{N,T}}) where {N,T} = eltype(eltype(SpectralFunction{N,T}))
(f::DOSIntegrand)(k::AbstractVector) = htr(f.A(k))
(f::DOSIntegrand)(A::AbstractMatrix) = htr(f.A(A))
contract(f::DOSIntegrand, x) = DOSIntegrand(contract(f.A, x))

# Method patches for compatibility with HermitianFourierSeries
function (f::SpectralFunction{N,T})(k::AbstractVector) where {N,T<:HermitianFourierSeries}
    z = complex(f.ω + f.μ, f.η)
    A = SVector{6,ComplexF64}(z,zero(z),zero(z),z,zero(z),z) - f.ϵ(k)
    imag(hinv(A))/(-pi)
end
function (f::SpectralFunction{N,T})(ϵ::SVector{6}) where {N,T<:HermitianFourierSeries}
    z = complex(f.ω + f.μ, f.η)
    @inbounds A = SVector{6}(z-ϵ[1],-ϵ[2],-ϵ[3],z-ϵ[4],-ϵ[5],z-ϵ[6])
    imag(hinv(A))/(-pi)
end
(f::DOSIntegrand)(A::SVector{6}) = htr(f.A(A))