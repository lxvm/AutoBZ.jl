export GreensFunction, SpectralFunction, DOSIntegrand, OpticalConductivityIntegrand, FixedKGridOpticalConductivity, EquispaceOpticalConductivity

"""
    GreensFunction(ϵ,ω,η,μ)

A struct that calculates the lattice Green's function from a Hamiltonian.
"""
struct GreensFunction{N,T<:FourierSeries{N}}
    ϵ::T
    ω::Float64
    η::Float64
    μ::Float64
end

Base.eltype(::Type{<:GreensFunction{N,T}}) where {N,T} = Base.promote_op(inv, eltype(T))
(f::GreensFunction)(ϵ_k::AbstractMatrix) = _Greens_function(ϵ_k, f.ω, f.η, f.μ)
(f::GreensFunction)(k::Union{AbstractVector,Number}) = f(f.ϵ(k))
contract(f::GreensFunction, x::Number) = GreensFunction(contract(f.ϵ, x), f.ω, f.η, f.μ)

_Greens_function(ϵ::AbstractMatrix, ω, η, μ) = inv(complex(ω+μ, η)*I - ϵ)


"""
    SpectralFunction(::GreensFunction)
    SpectralFunction(ϵ,ω,η,μ)

A struct that calculates the imaginary part of the Green's function.
"""
struct SpectralFunction{N,T}
    G::GreensFunction{N,T}
end

SpectralFunction(ϵ, ω, η, μ) = SpectralFunction(GreensFunction(ϵ, ω, η, μ))
Base.eltype(::Type{<:SpectralFunction{N,T}}) where {N,T} = Base.promote_op(imag, eltype(GreensFunction{N,T}))
(f::SpectralFunction)(G_k::AbstractMatrix) = _spectral_function(G_k)
(f::SpectralFunction)(k::Union{AbstractVector,Number}) = f(f.G(k))
contract(f::SpectralFunction, x::Number) = SpectralFunction(contract(f.G, x))

_spectral_function(G::AbstractMatrix) = imag(G)/(-pi)

"""
    DOSIntegrand(::SpectralFunction)
    DOSIntegrand(ϵ,ω,η,μ)

A struct whose integral gives the density of states.
"""
struct DOSIntegrand{N,T}
    A::SpectralFunction{N,T}
end

DOSIntegrand(ϵ, ω, η, μ) = DOSIntegrand(SpectralFunction(ϵ, ω, η, μ))
Base.eltype(::Type{<:DOSIntegrand{N,T}}) where {N,T} = eltype(eltype(SpectralFunction{N,T}))
(f::DOSIntegrand)(A_k::AbstractMatrix) = _DOS_integrand(A_k)
(f::DOSIntegrand)(k::Union{AbstractVector,Number}) = f(f.A(k))
contract(f::DOSIntegrand, x::Number) = DOSIntegrand(contract(f.A, x))

_DOS_integrand(A) = tr(A)

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
struct OpticalConductivityIntegrand{N,T<:FourierSeries{N},T1,T2,T3}
    ϵ::T
    ν₁::T1
    ν₂::T2
    ν₃::T3
    Ω::Float64
    β::Float64
    η::Float64
    μ::Float64
end

function OpticalConductivityIntegrand(ϵ, Ω, β, η, μ)
    ν₁ = FourierSeriesDerivative(ϵ, SVector(1,0,0))
    ν₂ = FourierSeriesDerivative(ϵ, SVector(0,1,0))
    ν₃ = FourierSeriesDerivative(ϵ, SVector(0,0,1))
    OpticalConductivityIntegrand(ϵ, ν₁, ν₂, ν₃, Ω, β, η, μ)
end
Base.eltype(::Type{<:OpticalConductivityIntegrand}) = SMatrix{3,3,ComplexF64,9}
(f::OpticalConductivityIntegrand)(ϵ, ν₁, ν₂, ν₃, ω) = -fermi_window(ω, f.Ω, f.β, f.μ) * _optical_conductivity(ϵ, ν₁, ν₂, ν₃, ω, f.Ω, f.η, f.μ)
function (f::OpticalConductivityIntegrand)(kω::SVector{4})
    k = pop(kω)
    ω = last(kω)
    f(f.ϵ(k), f.ν₁(k), f.ν₂(k), f.ν₃(k), ω)
end
(f::OpticalConductivityIntegrand)(ω::SVector{1}) = f(first(ω))
(f::OpticalConductivityIntegrand{0})(ω::Number) = f(first(f.ϵ.coeffs), first(f.ν₁.ϵ.coeffs), first(f.ν₂.ϵ.coeffs), first(f.ν₃.ϵ.coeffs), ω)

contract(f::OpticalConductivityIntegrand, k::Number) = OpticalConductivityIntegrand(contract(f.ϵ, k), contract(f.ν₁, k), contract(f.ν₂, k), contract(f.ν₃, k), f.Ω, f.β, f.η, f.μ)

function _optical_conductivity(ϵ, ν₁, ν₂, ν₃, ω, Ω, η, μ)
    # zero(μ) since μ is already accounted for by the Fermi windows
    Gω = _Greens_function(ϵ, ω, η, zero(μ))
    GΩ = _Greens_function(ϵ, ω+Ω, η, zero(μ))
    Aω = _spectral_function(Gω)
    AΩ = _spectral_function(GΩ)
    ν₁Aω = ν₁*Aω
    ν₂Aω = ν₂*Aω
    ν₃Aω = ν₃*Aω
    ν₁AΩ = ν₁*AΩ
    ν₂AΩ = ν₂*AΩ
    ν₃AΩ = ν₃*AΩ
    SMatrix{3,3,ComplexF64,9}(tr.((ν₁Aω*ν₁AΩ, ν₂Aω*ν₁AΩ, ν₃Aω*ν₁AΩ, ν₁Aω*ν₂AΩ, ν₂Aω*ν₂AΩ, ν₃Aω*ν₂AΩ, ν₁Aω*ν₃AΩ, ν₂Aω*ν₃AΩ, ν₃Aω*ν₃AΩ)))
end

"""
In this struct, ϵ, ν₁, ν₂, ν₃ are these values precomputed on an equispace grid.
dvol is the volume of the discretized grid cell.
"""
struct FixedKGridOpticalConductivity{T,T1,T2,T3}
    Ω::Float64
    β::Float64
    η::Float64
    μ::Float64
    ϵ::T
    ν₁::T1
    ν₂::T2
    ν₃::T3
    dvol::Float64
end

Base.eltype(::Type{<:FixedKGridOpticalConductivity}) = SMatrix{3,3,ComplexF64,9}
(f::FixedKGridOpticalConductivity)(ω::SVector{1}) = f(first(ω))
(f::FixedKGridOpticalConductivity)(ω::Number) = -f.dvol * fermi_window(ω, f.Ω, f.β, f.μ) * sum(CartesianIndices(f.ϵ); init=zero(eltype(f))) do i
    _optical_conductivity(f.ϵ[i], f.ν₁[i], f.ν₂[i], f.ν₃[i], ω, f.Ω, f.η, f.μ)
end

mutable struct EquispaceOpticalConductivity{N,T,T1,T2,T3,V,V1,V2,V3}
    σ::OpticalConductivityIntegrand{N,T,T1,T2,T3}
    Ω::Float64
    β::Float64
    η::Float64
    μ::Float64
    p1::Int
    p2::Int
    ϵ1::Array{V,N}
    ν₁1::Array{V1,N}
    ν₂1::Array{V2,N}
    ν₃1::Array{V3,N}
    ϵ2::Array{V,N}
    ν₁2::Array{V1,N}
    ν₂2::Array{V2,N}
    ν₃2::Array{V3,N}
    rtol::Float64
    atol::Float64
end

Base.eltype(::Type{<:EquispaceOpticalConductivity}) = SMatrix{3,3,ComplexF64,9}

(f::EquispaceOpticalConductivity)(ω::SVector{1}) = f(first(ω))
function (f::EquispaceOpticalConductivity)(w::Number)
    int1s[i], f.p1, f.r1, wsym1, int2s[i], p2, r2, wsym2 = resolve_integrand(f(ϵ, ω, η, μ), ϵ, p1, r1, wsym1, p2, r2, wsym2, s, η; atol=f.atol, rtol=f.rtol)
end

function resolve_integrand(f, ϵ::FourierSeries{N}, p1, r1, wsym1, p2, r2, wsym2, s, η; atol=0, rtol=1e-3) where {N}
    int1 = evaluate_integrand(f, p1, r1, wsym1, N)
    int2 = evaluate_integrand(f, p2, r2, wsym2, N)
    err = norm(int1 - int2)
    while err > max(rtol*norm(int1), atol)
        p1 = p2
        r1 = r2
        wsym1 = wsym2
        int1 = int2

        p2 = refine_grid_heuristic(p1, s, η)
        r2, wsym2 = evaluate_series_ibz(ϵ, p2)
        int2 = evaluate_integrand_ibz(f, p2, r2, wsym2)
        err = norm(int1 - int2)
    end
    err, int1, p1, r1, wsym1, int2, p2, r2, wsym2
end