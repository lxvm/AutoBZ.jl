export GreensFunction, SpectralFunction, DOSIntegrand, GammaIntegrand, OpticalConductivityIntegrand, FixedKGridGammaIntegrand, FixedKGridOpticalConductivity, EquispaceOpticalConductivity

greens_function(H, ω, Σ, μ) = greens_function(H, ω+μ, Σ)
greens_function(H, ω, Σ) = greens_function(H, ω*I-Σ(ω))
greens_function(H, ω, Σ::AbstractMatrix) = greens_function(H, ω*I-Σ)
greens_function(H, ω, η::Real) = greens_function(H, complex(ω, η)*I)
greens_function(H::AbstractMatrix, M) = inv(M-H)

"""
    GreensFunction(H,ω,Σ,μ)
    GreensFunction(H,ω,Σ)
    GreensFunction(H,ω,η::Real)
    GreensFunction(H,M)

A struct that calculates the lattice Green's function from a Hamiltonian.
```math
G(k;H,ω,η,μ) = {((\\omega + \\mu + i \eta) I - H(k))}^{-1}
```
"""
struct GreensFunction{TH<:FourierSeries,TM}
    H::TH
    M::TM
end

greens_function(H::FourierSeries, M) = GreensFunction(H, M)
greens_function(H::FourierSeries, M::UniformScaling) = GreensFunction(H, M.λ*one(eltype(H)))

GreensFunction(args...) = greens_function(args...)

Base.eltype(::Type{<:GreensFunction{TH}}) where {TH} = Base.promote_op(inv, eltype(TH))

(g::GreensFunction)(k::Union{AbstractVector,Number}) = g(g.H(k))
(g::GreensFunction)(H_k::AbstractMatrix) = greens_function(H_k, g.M)

contract(g::GreensFunction, x::Number) = GreensFunction(contract(g.H, x), g.M)


spectral_function(args...) = spectral_function(greens_function(args...))
spectral_function(G::AbstractMatrix) = imag(G)/(-pi)

"""
    SpectralFunction(::GreensFunction)

A struct that calculates the imaginary part of the Green's function.
```math
A(k;H,ω,η,μ) = {\\pi}^{-1} \\Im[G(k;H,ω,η,μ)]
```
"""
struct SpectralFunction{TG<:GreensFunction}
    G::TG
end

spectral_function(G::GreensFunction) = SpectralFunction(G)

SpectralFunction(args...) = spectral_function(args...)

Base.eltype(::Type{<:SpectralFunction{TG}}) where {TG} = Base.promote_op(imag, eltype(TG))

(A::SpectralFunction)(k::Union{AbstractVector,Number}) = A(A.G(k))
(A::SpectralFunction)(G_k::AbstractMatrix) = spectral_function(G_k)

contract(A::SpectralFunction, x::Number) = SpectralFunction(contract(A.G, x))


dos_integrand(args...) = dos_integrand(spectral_function(args...))
dos_integrand(A::AbstractMatrix) = tr(A)

"""
    DOSIntegrand(::SpectralFunction)

A struct whose integral gives the density of states.
```math
D(k;H,ω,η,μ) = \\Tr[A(k;H,ω,η,μ)]
```
"""
struct DOSIntegrand{TA<:SpectralFunction}
    A::TA
end

dos_integrand(A::SpectralFunction) = DOSIntegrand(A)

DOSIntegrand(args...) = dos_integrand(args...)

Base.eltype(::Type{<:DOSIntegrand{TA}}) where {TA} = eltype(eltype(TA))

(D::DOSIntegrand)(k::Union{AbstractVector,Number}) = D(D.A(k))
(D::DOSIntegrand)(A_k::AbstractMatrix) = dos_integrand(A_k)

contract(D::DOSIntegrand, x::Number) = DOSIntegrand(contract(D.A, x))



function gamma_integrand(H, ν₁, ν₂, ν₃, ω, Ω, η, μ)
    Gω = greens_function(H, ω, η, μ)
    GΩ = greens_function(H, ω+Ω, η, μ)
    Aω = spectral_function(Gω)
    AΩ = spectral_function(GΩ)
    ν₁Aω = ν₁*Aω
    ν₂Aω = ν₂*Aω
    ν₃Aω = ν₃*Aω
    ν₁AΩ = ν₁*AΩ
    ν₂AΩ = ν₂*AΩ
    ν₃AΩ = ν₃*AΩ
    SMatrix{3,3,ComplexF64,9}(tr.((ν₁Aω*ν₁AΩ, ν₂Aω*ν₁AΩ, ν₃Aω*ν₁AΩ, ν₁Aω*ν₂AΩ, ν₂Aω*ν₂AΩ, ν₃Aω*ν₂AΩ, ν₁Aω*ν₃AΩ, ν₂Aω*ν₃AΩ, ν₃Aω*ν₃AΩ)))
end

"""
GammaIntegrand(H, ν₁, ν₂, ν₃, ω, Ω, β, η, μ)

A function whose integral over the BZ gives the transport distribution.
```math
\\Gamma_{\\alpha\\beta}(k) = \\Tr[\\nu^\\alpha(k) A(k,\\omega) \\nu^\\beta(k) A(k, \\omega+\\Omega)]
```
"""
struct GammaIntegrand{TH<:FourierSeries,T1,T2,T3}
    H::TH
    ν₁::T1
    ν₂::T2
    ν₃::T3
    ω::Float64
    Ω::Float64
    η::Float64
    μ::Float64
end

function GammaIntegrand(H, ω, Ω, η, μ)
    ν₁ = FourierSeriesDerivative(H, SVector(1,0,0))
    ν₂ = FourierSeriesDerivative(H, SVector(0,1,0))
    ν₃ = FourierSeriesDerivative(H, SVector(0,0,1))
    GammaIntegrand(H, ν₁, ν₂, ν₃, ω, Ω, η, μ)
end
Base.eltype(::Type{<:GammaIntegrand}) = SMatrix{3,3,ComplexF64,9}
(f::GammaIntegrand)(H_k, ν₁_k, ν₂_k, ν₃_k) = gamma_integrand(H_k, ν₁_k, ν₂_k, ν₃_k, f.ω, f.Ω, f.η, f.μ)
(f::GammaIntegrand)(k::Union{AbstractVector,Number}) = f(f.H(k), f.ν₁(k), f.ν₂(k), f.ν₃(k))
contract(f::GammaIntegrand, k::Number) = GammaIntegrand(contract(f.H, k), contract(f.ν₁, k), contract(f.ν₂, k), contract(f.ν₃, k), f.ω, f.Ω, f.η, f.μ)


fermi(ω, β, μ) = fermi(ω-μ, β)
fermi(ω, β) = fermi(β*ω)
function fermi(x)
    y = exp(x)
    inv(one(y) + y)
end

fermi′(ω, β, μ) = fermi′(ω-μ, β)
fermi′(ω, β) = β*fermi′(β*ω)
function fermi′(x)
    y = cosh(x)
    -0.5inv(one(y)+y)
end

"Evaluates a unitless window function determined by the Fermi distribution"
fermi_window(ω, Ω, β, μ) = fermi_window(ω-μ, Ω, β)
fermi_window(ω, Ω, β) = fermi_window(β*ω, β*Ω)
function fermi_window(x, y)
    if y == zero(y)
        return -fermi′(x)
    else
        # equivalent to (fermi(x) - fermi(x+y)) / y
        # TODO: prevent overflow of cosh(y/2) for y/2>710
        return tanh(y/2) / (y*(cosh(x+y/2)/cosh(y/2)+1))
    end
end

optical_conductivity(H, ν₁, ν₂, ν₃, ω, Ω, β, η, μ) = β * fermi_window(ω, Ω, β) * gamma_integrand(H, ν₁, ν₂, ν₃, ω, Ω, η, μ)

"""
    OpticalConductivityIntegrand(H, ν₁, ν₂, ν₃, Ω, β, η, μ)

A function whose integral over the BZ and the frequency axis gives the optical conductivity
"""
struct OpticalConductivityIntegrand{TH<:FourierSeries,T1,T2,T3}
    H::TH
    ν₁::T1
    ν₂::T2
    ν₃::T3
    Ω::Float64
    β::Float64
    η::Float64
    μ::Float64
end

function OpticalConductivityIntegrand(H, Ω, β, η, μ)
    ν₁ = FourierSeriesDerivative(H, SVector(1,0,0))
    ν₂ = FourierSeriesDerivative(H, SVector(0,1,0))
    ν₃ = FourierSeriesDerivative(H, SVector(0,0,1))
    OpticalConductivityIntegrand(H, ν₁, ν₂, ν₃, Ω, β, η, μ)
end
Base.eltype(::Type{<:OpticalConductivityIntegrand}) = SMatrix{3,3,ComplexF64,9}
(f::OpticalConductivityIntegrand)(H_k, ν₁_k, ν₂_k, ν₃_k, ω) = optical_conductivity(H_k, ν₁_k, ν₂_k, ν₃_k, ω, f.Ω, f.β, f.η, f.μ)
function (f::OpticalConductivityIntegrand)(kω::SVector)
    k = pop(kω)
    ω = last(kω)
    f(f.H(k), f.ν₁(k), f.ν₂(k), f.ν₃(k), ω)
end
(f::OpticalConductivityIntegrand)(ω::SVector{1}) = f(only(ω))
(f::OpticalConductivityIntegrand)(ω::Number) = f(only(f.H.coeffs), only(f.ν₁.f.coeffs), only(f.ν₂.f.coeffs), only(f.ν₃.f.coeffs), ω)

contract(f::OpticalConductivityIntegrand, k::Number) = OpticalConductivityIntegrand(contract(f.H, k), contract(f.ν₁, k), contract(f.ν₂, k), contract(f.ν₃, k), f.Ω, f.β, f.η, f.μ)


"""
In this struct, H, ν₁, ν₂, ν₃ are these values precomputed on an equispace grid.
dvol is the volume of the discretized grid cell.
"""
struct FixedKGridGammaIntegrand{TH,T1,T2,T3}
    η::Float64
    μ::Float64
    H::TH
    ν₁::T1
    ν₂::T2
    ν₃::T3
    dvol::Float64
end

Base.eltype(::Type{<:FixedKGridGammaIntegrand}) = SMatrix{3,3,ComplexF64,9}
(f::FixedKGridGammaIntegrand)(ω::Number, Ω::Number) = f.dvol * sum(CartesianIndices(f.H); init=zero(eltype(f))) do i
    gamma_integrand(f.H[i], f.ν₁[i], f.ν₂[i], f.ν₃[i], ω, Ω, f.η, f.μ)
end

"""
In this struct, H, ν₁, ν₂, ν₃ are these values precomputed on an equispace grid.
dvol is the volume of the discretized grid cell.
"""
struct FixedKGridOpticalConductivity{TH,T1,T2,T3}
    Ω::Float64
    β::Float64
    η::Float64
    μ::Float64
    H::TH
    ν₁::T1
    ν₂::T2
    ν₃::T3
    dvol::Float64
end

Base.eltype(::Type{<:FixedKGridOpticalConductivity}) = SMatrix{3,3,ComplexF64,9}
(f::FixedKGridOpticalConductivity)(ω::SVector{1}) = f(only(ω))
(f::FixedKGridOpticalConductivity)(ω::Number) = f.dvol * fermi_window(ω, f.Ω, f.β) * sum(CartesianIndices(f.H); init=zero(eltype(f))) do i
    gamma_integrand(f.H[i], f.ν₁[i], f.ν₂[i], f.ν₃[i], ω, f.Ω, f.η, f.μ)
end


mutable struct EquispaceOpticalConductivity{TH,T1,T2,T3,VH,V1,V2,V3}
    σ::OpticalConductivityIntegrand{TH,T1,T2,T3}
    Ω::Float64
    β::Float64
    η::Float64
    μ::Float64
    p1::Int
    p2::Int
    H1::Vector{VH}
    ν₁1::Vector{V1}
    ν₂1::Vector{V2}
    ν₃1::Vector{V3}
    H2::Vector{VH}
    ν₁2::Vector{V1}
    ν₂2::Vector{V2}
    ν₃2::Vector{V3}
    rtol::Float64
    atol::Float64
end

Base.eltype(::Type{<:EquispaceOpticalConductivity}) = SMatrix{3,3,ComplexF64,9}

(f::EquispaceOpticalConductivity)(ω::SVector{1}) = f(only(ω))
function (f::EquispaceOpticalConductivity)(ω::Number)
    int1s[i], f.p1, f.r1, wsym1, int2s[i], p2, r2, wsym2 = resolve_integrand(f(H, ω, η, μ), H, p1, r1, wsym1, p2, r2, wsym2, s, η; atol=f.atol, rtol=f.rtol)
end