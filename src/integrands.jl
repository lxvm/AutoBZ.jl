export Integrand, SpectralFunction, DOSIntegrand, DipoleMatrix, OpticalConductivityIntegrand

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
(f::SpectralFunction)(ϵ::AbstractMatrix) = _spectral_function(ϵ, f.ω, f.η, f.μ)
(f::SpectralFunction)(k::AbstractVector) = f(f.ϵ(k))
contract(f::SpectralFunction, x) = SpectralFunction(contract(f.ϵ, x), f.ω, f.η, f.μ)

_spectral_function(ϵ::AbstractMatrix, ω, η, μ) = imag(hinv(complex(ω+μ, η)*I - ϵ))/(-pi)

"""
DOSIntegrand(::SpectralFunction)
A function whose integral gives the density of states.
"""
struct DOSIntegrand{N,T} <: Integrand{N}
    A::SpectralFunction{N,T}
end

DOSIntegrand(ϵ, ω, η, μ) = DOSIntegrand(SpectralFunction(ϵ, ω, η, μ))
Base.eltype(::Type{<:DOSIntegrand{N,T}}) where {N,T} = eltype(eltype(SpectralFunction{N,T}))
(f::DOSIntegrand)(A::AbstractMatrix) = htr(f.A(A))
(f::DOSIntegrand)(k::AbstractVector) = f(f.A(k))
contract(f::DOSIntegrand, x) = DOSIntegrand(contract(f.A, x))

struct DipoleMatrix1{N,T<:FourierSeries{N},α} <: Integrand{N}
    ϵ::T
    function DipoleMatrix1(ϵ::T, α::Int) where {N,T<:FourierSeries{N}}
        (1 <= α <= N) || throw("cannot differentiate axis α=$α, choose in 1:$N")
        new{N,T,α}(ϵ)
    end
end

DipoleMatrix = DipoleMatrix1
Base.eltype(::Type{<:DipoleMatrix{N,T,α}}) where {N,T,α} = eltype(T)

fermi(ω, β) = fermi(promote(ω, β)...)
fermi(ω::T, β::T) where {T<:Real} = inv(one(T) + exp(β*ω))

function fermi_window(ω, Ω, β)
    if Ω == zero(Ω)
        return -fermi(ω, β)^2 * β * exp(β*ω)
    else # TODO: implement in numerically stable fashion to prevent cancellation
        return (fermi(ω+Ω, β) - fermi(ω, β)) / Ω
    end
end

struct OpticalConductivityIntegrand1{N,T<:FourierSeries{N}} <: Integrand{N}
    ϵ::T
    Ω::Float64
    β::Float64
    η::Float64
    μ::Float64
end

OpticalConductivityIntegrand = OpticalConductivityIntegrand1

Base.eltype(::Type{<:OpticalConductivityIntegrand{N,T}}) where {N,T} = eltype(eltype(T))
function (f::OpticalConductivityIntegrand)(ϵ, ν₁, ν₂, ν₃)
    Aω = _spectral_function(ϵ, ω, f.η, f.μ)
    AΩ = _spectral_function(ϵ, ω+f.Ω, f.η, f.μ)
    ν₁Aω = ν₁*Aω
    ν₂Aω = ν₂*Aω
    ν₃Aω = ν₃*Aω
    ν₁AΩ = ν₁*AΩ
    ν₂AΩ = ν₂*AΩ
    ν₃AΩ = ν₃*AΩ
    fermi_window(ω, f.Ω, f.β) * SMatrix{3,3,ComplexF64,9}(tr.((ν₁Aω*ν₁AΩ, ν₂Aω*ν₁AΩ, ν₃Aω*ν₁AΩ, ν₁Aω*ν₂AΩ, ν₂Aω*ν₂AΩ, ν₃Aω*ν₂AΩ, ν₁Aω*ν₃AΩ, ν₂Aω*ν₃AΩ, ν₃Aω*ν₃AΩ)))
end
function (f::OpticalConductivityIntegrand)(kω::SVector{4})
    k = pop(kω)
    ω = last(kω)
    ϵ = f.ϵ(k)
    C = f.ϵ.coeffs
    ϕ = (2π*im) .* k ./ f.ϵ.period
    ν₁ = sum(CartesianIndices(C), init=zero(ϵ)) do i
        @inbounds i[1] * C[i] * exp(dot(ϕ, convert(SVector, i)))
    end
    ν₂ = sum(CartesianIndices(C), init=zero(ϵ)) do i
        @inbounds i[2] * C[i] * exp(dot(ϕ, convert(SVector, i)))
    end
    ν₃ = sum(CartesianIndices(C), init=zero(ϵ)) do i
        @inbounds i[3] * C[i] * exp(dot(ϕ, convert(SVector, i)))
    end
    f(ϵ, ν₁, ν₂, ν₃)
end
(f::OpticalConductivityIntegrand)(ω::SVector{1}) = f(first(ω))
function (f::OpticalConductivityIntegrand{1})(ω::Number)
    fermi_window(ω, f.Ω, f.β) * tr()
end
function contract(f::OpticalConductivityIntegrand, k::Number)

end