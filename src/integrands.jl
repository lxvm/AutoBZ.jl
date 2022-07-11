export Integrand, SpectralFunction, DOSIntegrand, OpticalConductivityIntegrand

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

_spectral_function(ϵ::AbstractMatrix, ω, η, μ) = imag(inv(complex(ω+μ, η)*I - ϵ))/(-pi)

"""
    DOSIntegrand(::SpectralFunction)
A function whose integral gives the density of states.
"""
struct DOSIntegrand{N,T} <: Integrand{N}
    A::SpectralFunction{N,T}
end

DOSIntegrand(ϵ, ω, η, μ) = DOSIntegrand(SpectralFunction(ϵ, ω, η, μ))
Base.eltype(::Type{<:DOSIntegrand{N,T}}) where {N,T} = eltype(eltype(SpectralFunction{N,T}))
(f::DOSIntegrand)(A::AbstractMatrix) = tr(f.A(A))
(f::DOSIntegrand)(k::AbstractVector) = f(f.A(k))
contract(f::DOSIntegrand, x) = DOSIntegrand(contract(f.A, x))

fermi(ω, β, μ) = _fermi(β*(ω-μ))
function _fermi(x)
    y = exp(x)
    inv(one(y) + y)
end
function fermi′(ω, β, μ)
    x = β*(ω-μ)
    dx = β
    -_fermi(x)^2 * dx * exp(x)
end

function fermi_window(ω, Ω, β, μ)
    if Ω == zero(Ω)
        return fermi′(ω, β, μ)
    else # TODO: implement in numerically stable fashion to prevent cancellation
        return (fermi(ω+Ω, β, μ) - fermi(ω, β, μ)) / Ω
    end
end

"""
    OpticalConductivityIntegrand(ϵ, ν₁, ν₂, ν₃, Ω, β, η, μ)
A function whose integral over the BZ and the frequency axis gives the optical conductivity
"""
struct OpticalConductivityIntegrand{N,T<:FourierSeries{N},T1,T2,T3,d}  <: Integrand{d}
    ϵ::T
    ν₁::T1
    ν₂::T2
    ν₃::T3
    Ω::Float64
    β::Float64
    η::Float64
    μ::Float64
    function OpticalConductivityIntegrand(ϵ::T, ν₁::T1, ν₂::T2, ν₃::T3, Ω, β, η, μ) where {N,T<:FourierSeries{N},T1<:FourierSeriesDerivative{N},T2<:FourierSeriesDerivative{N},T3<:FourierSeriesDerivative{N}}
        new{N,T,T1,T2,T3,N+1}(ϵ, ν₁, ν₂, ν₃, Ω, β, η, μ)
    end
end

function OpticalConductivityIntegrand(ϵ, Ω, β, η, μ)
    ν₁ = FourierSeriesDerivative{Tuple{1,0,0}}(ϵ)
    ν₂ = FourierSeriesDerivative{Tuple{0,1,0}}(ϵ)
    ν₃ = FourierSeriesDerivative{Tuple{0,0,1}}(ϵ)
    OpticalConductivityIntegrand(ϵ, ν₁, ν₂, ν₃, Ω, β, η, μ)
end
Base.eltype(::Type{<:OpticalConductivityIntegrand{N,T}}) where {N,T} = SMatrix{3,3,ComplexF64,9}
function (f::OpticalConductivityIntegrand)(ω, ϵ, ν₁, ν₂, ν₃)
    Aω = _spectral_function(ϵ, ω, f.η, f.μ)
    AΩ = _spectral_function(ϵ, ω+f.Ω, f.η, f.μ)
    ν₁Aω = ν₁*Aω
    ν₂Aω = ν₂*Aω
    ν₃Aω = ν₃*Aω
    ν₁AΩ = ν₁*AΩ
    ν₂AΩ = ν₂*AΩ
    ν₃AΩ = ν₃*AΩ
    fermi_window(ω, f.Ω, f.β, f.μ) * SMatrix{3,3,ComplexF64,9}(tr.((ν₁Aω*ν₁AΩ, ν₂Aω*ν₁AΩ, ν₃Aω*ν₁AΩ, ν₁Aω*ν₂AΩ, ν₂Aω*ν₂AΩ, ν₃Aω*ν₂AΩ, ν₁Aω*ν₃AΩ, ν₂Aω*ν₃AΩ, ν₃Aω*ν₃AΩ)))
end
function (f::OpticalConductivityIntegrand)(kω::SVector{4})
    k = pop(kω)
    ω = last(kω)
    f(ω, f.ϵ(k), f.ν₁(k), f.ν₂(k), f.ν₃(k))
end
(f::OpticalConductivityIntegrand)(ω::SVector{1}) = f(first(ω))
(f::OpticalConductivityIntegrand{0})(ω::Number) = f(ω, first(f.ϵ.coeffs), first(f.ν₁.ϵ.coeffs), first(f.ν₂.ϵ.coeffs), first(f.ν₃.ϵ.coeffs))

contract(f::OpticalConductivityIntegrand, k::Number) = OpticalConductivityIntegrand(contract(f.ϵ, k), contract(f.ν₁, k), contract(f.ν₂, k), contract(f.ν₃, k), f.Ω, f.β, f.η, f.μ)