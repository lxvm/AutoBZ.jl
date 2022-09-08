export GreensFunction, SpectralFunction, DOSIntegrand, GammaIntegrand, OCIntegrand, FixedKGridGammaIntegrand, FixedKGridOpticalConductivity, EquispaceOpticalConductivity

greens_function(H, ω, Σ, μ) = greens_function(H, (ω+μ)*I-Σ(ω))
greens_function(H, ω, Σ::AbstractMatrix, μ) = greens_function(H, (ω+μ)*I-Σ)
greens_function(H, ω, Σ) = greens_function(H, ω*I-Σ(ω))
greens_function(H, ω, Σ::AbstractMatrix) = greens_function(H, ω*I-Σ)
greens_function(H::AbstractMatrix, M) = inv(M-H)

"""
    GreensFunction(H,ω,Σ,μ)
    GreensFunction(H,ω,Σ)
    GreensFunction(H,M)

A struct that calculates the lattice Green's function from a Hamiltonian.
```math
G(k;H,ω,η,μ) = {((\\omega + \\mu) I - H(k) - \\Sigma(\\omega))}^{-1}
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


gamma_integrand(H, ν₁, ν₂, ν₃, Σ, ω, Ω, μ) = gamma_integrand(H, ν₁, ν₂, ν₃, (ω+μ)*I-Σ(ω), (ω+Ω+μ)*I-Σ(ω+Ω))
gamma_integrand(H, ν₁, ν₂, ν₃, Σ::AbstractMatrix, ω, Ω, μ) = gamma_integrand(H, ν₁, ν₂, ν₃, (ω+μ)*I-Σ, (ω+Ω+μ)*I-Σ)
function gamma_integrand(H, ν₁, ν₂, ν₃, Mω, MΩ)
    Gω = greens_function(H, Mω)
    GΩ = greens_function(H, MΩ)
    Aω = spectral_function(Gω)
    AΩ = spectral_function(GΩ)
    gamma_integrand_(ν₁, ν₂, ν₃, Aω, AΩ)
end
function gamma_integrand_(ν₁, ν₂, ν₃, Aω, AΩ)
    ν₁Aω = ν₁*Aω
    ν₂Aω = ν₂*Aω
    ν₃Aω = ν₃*Aω
    ν₁AΩ = ν₁*AΩ
    ν₂AΩ = ν₂*AΩ
    ν₃AΩ = ν₃*AΩ
    SMatrix{3,3,ComplexF64,9}(tr.((ν₁Aω*ν₁AΩ, ν₂Aω*ν₁AΩ, ν₃Aω*ν₁AΩ, ν₁Aω*ν₂AΩ, ν₂Aω*ν₂AΩ, ν₃Aω*ν₂AΩ, ν₁Aω*ν₃AΩ, ν₂Aω*ν₃AΩ, ν₃Aω*ν₃AΩ)))
end

"""
    GammaIntegrand(H, Σ, ω, Ω, μ)
    GammaIntegrand(H, ν₁, ν₂, ν₃, Mω, MΩ)

A function whose integral over the BZ gives the transport distribution.
```math
\\Gamma_{\\alpha\\beta}(k) = \\Tr[\\nu^\\alpha(k) A(k,\\omega) \\nu^\\beta(k) A(k, \\omega+\\Omega)]
```
"""
struct GammaIntegrand{TH,T1,T2,T3,M1,M2}
    H::TH
    ν₁::T1
    ν₂::T2
    ν₃::T3
    Mω::M1
    MΩ::M2
end

function GammaIntegrand(H, Σ, ω, Ω, μ)
    ν₁ = FourierSeriesDerivative(H, SVector(1,0,0))
    ν₂ = FourierSeriesDerivative(H, SVector(0,1,0))
    ν₃ = FourierSeriesDerivative(H, SVector(0,0,1))
    Mω = (ω+μ)*I-Σ(ω)
    MΩ = (ω+Ω+μ)*I-Σ(ω+Ω)
    GammaIntegrand(H, ν₁, ν₂, ν₃, Mω, MΩ)
end
Base.eltype(::Type{<:GammaIntegrand}) = SMatrix{3,3,ComplexF64,9}
(g::GammaIntegrand)(H_k, ν₁_k, ν₂_k, ν₃_k) = gamma_integrand(H_k, ν₁_k, ν₂_k, ν₃_k, g.Mω, g.MΩ)
(g::GammaIntegrand)(k::Union{AbstractVector,Number}) = g(g.H(k), g.ν₁(k), g.ν₂(k), g.ν₃(k))
contract(g::GammaIntegrand, k::Number) = GammaIntegrand(contract(g.H, k), contract(g.ν₁, k), contract(g.ν₂, k), contract(g.ν₃, k), g.Mω, g.MΩ)


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
fermi_window(x, y) = ifelse(y == zero(y), -fermi′(x), fermi_window_(x, y))

fermi_window_(x, y) = fermi_window_(promote(float(x), float(y))...)
function fermi_window_(x::T, y::T) where {T<:AbstractFloat}
    half_y = y*T(0.5)
    (tanh(half_y)/y)/(one(T)+cosh_ratio(x+half_y, half_y))
end

cosh_ratio(x, y) = cosh(x)/cosh(y)
function cosh_ratio(x::T, y::T) where {T<:Union{Float32,Float64}}
    abs_x = abs(x)
    abs_y = abs(y)
    arg_large = Base.Math.H_LARGE_X(T)
    arg_small = EXP_P1_SMALL_X(T)
    if max(abs_x, abs_y) < arg_large
        cosh(x)/cosh(y)
    elseif arg_large <= abs_x && -2*abs_y > arg_small
        exp(abs_x-abs_y)/(one(T)+exp(-2*abs_y))
    elseif arg_large <= abs_y && -2*abs_x > arg_small
        exp(abs_x-abs_y)*(one(T)+exp(-2*abs_x))
    else
        exp(abs_x-abs_y)
    end
end

# log(eps(T))
EXP_P1_SMALL_X(::Type{Float64}) = -36.04365338911715
EXP_P1_SMALL_X(::Type{Float32}) = -15.942385f0

oc_integrand(H, ν₁, ν₂, ν₃, Σ, ω, Ω, β, μ) = β * fermi_window(ω, Ω, β) * gamma_integrand(H, ν₁, ν₂, ν₃, Σ, ω, Ω, μ)

"""
    OCIntegrand(H, ν₁, ν₂, ν₃, Ω, β, η, μ)

A function whose integral over the BZ and the frequency axis gives the optical conductivity
"""
struct OCIntegrand{TH,T1,T2,T3,TS}
    H::TH
    ν₁::T1
    ν₂::T2
    ν₃::T3
    Σ::TS
    Ω::Float64
    β::Float64
    μ::Float64
end

function OCIntegrand(H, Σ, Ω, β, μ)
    ν₁ = FourierSeriesDerivative(H, SVector(1,0,0))
    ν₂ = FourierSeriesDerivative(H, SVector(0,1,0))
    ν₃ = FourierSeriesDerivative(H, SVector(0,0,1))
    OCIntegrand(H, ν₁, ν₂, ν₃, Σ, Ω, β, μ)
end
Base.eltype(::Type{<:OCIntegrand}) = SMatrix{3,3,ComplexF64,9}
(f::OCIntegrand)(H_k, ν₁_k, ν₂_k, ν₃_k, ω) = oc_integrand(H_k, ν₁_k, ν₂_k, ν₃_k, f.Σ, ω, f.Ω, f.β, f.μ)
function (f::OCIntegrand)(kω::SVector)
    k = pop(kω)
    ω = last(kω)
    f(f.H(k), f.ν₁(k), f.ν₂(k), f.ν₃(k), ω)
end
(f::OCIntegrand)(ω::SVector{1}) = f(only(ω))
(f::OCIntegrand)(ω::Number) = f(only(f.H.coeffs), only(f.ν₁.f.coeffs), only(f.ν₂.f.coeffs), only(f.ν₃.f.coeffs), ω)

contract(f::OCIntegrand, k::Number) = OCIntegrand(contract(f.H, k), contract(f.ν₁, k), contract(f.ν₂, k), contract(f.ν₃, k), f.Σ, f.Ω, f.β, f.μ)


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
    σ::OCIntegrand{TH,T1,T2,T3}
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