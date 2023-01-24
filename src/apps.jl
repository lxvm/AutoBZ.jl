"""
    gloc_integrand(H, Σ, ω)
    gloc_integrand(H, M)

Returns `inv(M-H)` where `M = ω*I-Σ(ω)`
"""
gloc_integrand(H::AbstractMatrix, M) = inv(M-H)

"""
    diaggloc_integrand(H, Σ, ω)
    diaggloc_integrand(H, M)

Returns `diag(inv(M-H))` where `M = ω*I-Σ(ω)`
"""
diaggloc_integrand(H::AbstractMatrix, M) = diag_inv(M-H)

"""
    dos_integrand(H, Σ, ω)
    dos_integrand(H, M)

Returns `imag(tr(inv(M-H)))/(-pi)` where `M = ω*I-Σ(ω)`
"""
dos_integrand(H::AbstractMatrix, M) = imag(tr_inv(M-H))/(-pi)

# Generic behavior for single Green's function integrands (methods and types)
for name in ("Gloc", "DiagGloc", "DOS")
    # create and export symbols
    f = Symbol(lowercase(name), "_integrand")
    T = Symbol(name, "Integrand")
    E = Symbol(name, "Integrator")
    @eval export $f, $T, $E

    # Define how self-energies are handled
    @eval $f(H, Σ::AbstractSelfEnergy, ω) = $f(H, ω*I-Σ(ω))
    @eval $f(H, Σ::AbstractMatrix, ω) = $f(H, ω*I-Σ)
    @eval $f(H::AbstractFourierSeries, M) = FourierIntegrand($f, H, M)

    # Define the type alias to have the same behavior as the function
    @eval const $T = FourierIntegrand{typeof($f)}
    @eval $T(args...) = $f(args...)

    # Define an integrator based on the function
    @eval const $E = FourierIntegrator{typeof($f)}

    # Define default equispace grid stepping based 
    @eval function AutoBZ.equispace_npt_update(npt, f::$T, atol, rtol)
        η = im_sigma_to_eta(-imag(f.p[1]))
        npt_update_eta(npt, η, period(f.s)[1])
    end

    # Define job scripts
end

# helper functions for equispace updates

im_sigma_to_eta(x::UniformScaling) = -x.λ
im_sigma_to_eta(x::AbstractMatrix) = minimum(abs, x)

"""
    npt_update_eta(npt, η, [n₀=6.0, Δn=2.3])

Implements the heuristics for incrementing kpts suggested in Appendix A
http://arxiv.org/abs/2211.12959. Choice of `n₀≈2π`, close to the period of a
canonical BZ, approximately places a point in every box of size `η`. Choice of
`Δn≈log(10)` should get an extra digit of accuracy from PTR upon refinement.
"""
npt_update_eta(npt, η, n₀=6.0, Δn=2.3) = npt + npt_update_eta_(η, npt == 0 ? n₀ : Δn)
npt_update_eta_(η, c) = max(50, round(Int, c/η))


# transport and conductivity integrands

export TransportIntegrand, KineticIntegrand, EquispaceKineticIntegrand, AutoEquispaceKineticIntegrand

spectral_function(H, M) = imag(inv(M-H))/(-pi)

transport_integrand(HV, Σ::AbstractSelfEnergy, ω₁, ω₂) = transport_integrand(HV, ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂))
transport_integrand(HV, Σ::AbstractMatrix, ω₁, ω₂) = transport_integrand(HV, ω₁*I-Σ, ω₂*I-Σ)
transport_integrand(HV::AbstractFourierSeries, Mω₁, Mω₂) = FourierIntegrand(transport_integrand, HV, Mω₁, Mω₂)
transport_integrand(HV::Tuple, Mω₁, Mω₂) = transport_integrand(HV..., Mω₁, Mω₂)
function transport_integrand(H, ν₁, ν₂, ν₃, Mω₁, Mω₂)
    # Aω₁ = spectral_function(H, Mω₁)
    # Aω₂ = spectral_function(H, Mω₂)
    # Probably missing a factor of (2*pi)^-3 to convert reciprocal space volume
    # to real space 1/V, V the volume of the unit cell
    # transport_integrand_(ν₁, ν₂, ν₃, Aω₁, Aω₂)
    transport_integrand_(ν₁, ν₂, ν₃, spectral_function(H, Mω₁), spectral_function(H, Mω₂))
end
function transport_integrand_(ν₁::V, ν₂::V, ν₃::V, Aω₁::A, Aω₂::A) where {V,A}
    T = Base.promote_op((v, a) -> tr_mul(v*a,v*a), V, A)
    ν₁Aω₁ = ν₁*Aω₁
    ν₂Aω₁ = ν₂*Aω₁
    ν₃Aω₁ = ν₃*Aω₁
    ν₁Aω₂ = ν₁*Aω₂
    ν₂Aω₂ = ν₂*Aω₂
    ν₃Aω₂ = ν₃*Aω₂
    SMatrix{3,3,T,9}((
        tr_mul(ν₁Aω₁, ν₁Aω₂), tr_mul(ν₂Aω₁, ν₁Aω₂), tr_mul(ν₃Aω₁, ν₁Aω₂),
        tr_mul(ν₁Aω₁, ν₂Aω₂), tr_mul(ν₂Aω₁, ν₂Aω₂), tr_mul(ν₃Aω₁, ν₂Aω₂),
        tr_mul(ν₁Aω₁, ν₃Aω₂), tr_mul(ν₂Aω₁, ν₃Aω₂), tr_mul(ν₃Aω₁, ν₃Aω₂),
    ))
end

const TransportIntegrand = FourierIntegrand{typeof(transport_integrand)}
TransportIntegrand(args...) = transport_integrand(args...)

const TransportIntegrator = FourierIntegrand{typeof(transport_integrand)}

function AutoBZ.equispace_npt_update(npt, Γ::TransportIntegrand, atol, rtol)
    ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
    ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
    npt_update_eta(npt, min(ηω₁, ηω₂), period(Γ.s)[1])
end


kinetic_integrand(HV, Σ, ω, Ω, β, n) = kinetic_integrand(transport_integrand(HV, Σ, ω, ω+Ω), ω, Ω, β, n)
kinetic_integrand(Γ, ω, Ω, β, n) = (ω*β)^n * β * fermi_window(β*ω, β*Ω) * Γ

"""
    KineticIntegrand(HV, Σ, β, n, [Ω=0])

A function whose integral over the BZ and the frequency axis gives the kinetic
coefficient. Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion. Use
this type only for adaptive integration and order the limits so that the
integral over the Brillouin zone is the outer integral and the frequency
integral is the inner integral. Based on
[TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html)
"""
struct KineticIntegrand{T<:Union{BandEnergyVelocity3D,BandEnergyBerryVelocity3D},TS<:AbstractSelfEnergy}
    HV::T
    Σ::TS
    β::Float64
    n::Int
    Ω::Float64
    function KineticIntegrand(HV::T, Σ::TS, β, n, Ω=0.0) where {T,TS}
        β == Inf && Ω == 0 && error("Ω=1/β=0 encountered. This case requires distributional frequency integrals, or can be computed with a TransportIntegrand")
        new{T,TS}(HV, Σ, β, n, Ω)
    end
end

Base.eltype(::Type{<:KineticIntegrand}) = SMatrix{3,3,ComplexF64,9}
# innermost integral is the frequency integral
(A::KineticIntegrand)(kω::SVector{4}) = kinetic_integrand(A.HV(pop(kω)), A.Σ, last(kω), A.Ω, A.β, A.n)
(A::KineticIntegrand)(ω::SVector{1}) = A(only(ω))
(A::KineticIntegrand)(ω::Number) = kinetic_integrand(value(A.HV), A.Σ, ω, A.Ω, A.β, A.n)

# dim - 1 to accomodate the innermost frequency integral
AutoBZ.iterated_pre_eval(A::KineticIntegrand, k, dim) = (contract!(A.HV, k, dim-1); return A)

TransportIntegrand(A::KineticIntegrand, ω::Float64) = TransportIntegrand(A.HV, A.Σ, ω, ω+A.Ω)

"""
    EquispaceKineticIntegrand(A::KineticIntegrand, l, npt, pre::Vector{Tuple{eltype(A.HV),Int}})
    EquispaceKineticIntegrand(A, l, npt)

This type represents an `KineticIntegrand`, `A` integrated adaptively in frequency
and with equispace integration over the Brillouin zone with a fixed number of
grid points `npt`. The argument `l` should be an `IntegrationLimits` for just
the Brillouin zone. This type should be called by an adaptive integration
routine whose limits of integration are only the frequency variable.
"""
struct EquispaceKineticIntegrand{T,TS,TL,THV}
    A::KineticIntegrand{T,TS}
    l::TL
    npt::Int
    pre::Vector{Tuple{THV,Int}}
end


function EquispaceKineticIntegrand(A::KineticIntegrand, l::IntegrationLimits, npt::Int)
    pre = equispace_pre_eval(TransportIntegrand(A, 0.0), l, npt)
    EquispaceKineticIntegrand(A, l, npt, pre)
end
(f::EquispaceKineticIntegrand)(ω::SVector{1}) = f(only(ω))
function (f::EquispaceKineticIntegrand)(ω::Number)
    Γ, = equispace_integration(TransportIntegrand(f.A, ω), f.l, f.npt; pre=f.pre)
    return kinetic_integrand(Γ, ω, f.A.Ω, f.A.β, f.A.n)
end

Base.eltype(::Type{<:EquispaceKineticIntegrand}) = SMatrix{3,3,ComplexF64,9}


"""
    AutoEquispaceKineticIntegrand(A, l, atol, rtol, npt1, pre1, npt2, pre2)
    AutoEquispaceKineticIntegrand(A, l, atol, rtol; npt1=0, pre1=Tuple{eltype(A.HV),Int}[], npt2=0,pre2=Tuple{eltype(σ.HV),Int}[])

This type represents an `KineticIntegrand`, `A` integrated adaptively in frequency
and with equispace integration over the Brillouin zone with a number of grid
points necessary to meet the maximum of the tolerances given by `atol` and
`rtol`. The argument `l` should be an `IntegrationLimits` for just the Brillouin
zone. This type should be called by an adaptive integration routine whose limits
of integration are only the frequency variable.

The keyword arguments, which are just passed to
[`automatic_equispace_integration`](@ref), are:
- `pre1`: a `Vector` containing tuples of the evaluated Hamiltonian + band
    velocities and integration weights
- `npt1`: an integer that should be equivalent to `length(pre1)`
- `pre2`: a `Vector` containing tuples of the evaluated Hamiltonian + band
    velocities and integration weights on a more refined grid than `pre1`
- `npt2`: an integer that should be equivalent to `length(pre)`
"""
mutable struct AutoEquispaceKineticIntegrand{T,TS,TL,THV}
    A::KineticIntegrand{T,TS}
    l::TL
    atol::Float64
    rtol::Float64
    npt1::Int
    pre1::Vector{Tuple{THV,Int}}
    npt2::Int
    pre2::Vector{Tuple{THV,Int}}
end


AutoEquispaceKineticIntegrand(A, l, atol, rtol; npt1=0, pre1=Tuple{eltype(A.HV),Int}[], npt2=0,pre2=Tuple{eltype(A.HV),Int}[]) =
    AutoEquispaceKineticIntegrand(A, l, atol, rtol, npt1, pre1, npt2, pre2)

Base.eltype(::Type{<:AutoEquispaceKineticIntegrand}) = SMatrix{3,3,ComplexF64,9}

(f::AutoEquispaceKineticIntegrand)(ω::SVector{1}) = f(only(ω))
function (f::AutoEquispaceKineticIntegrand)(ω::Number)
    Γ, _, other = automatic_equispace_integration(TransportIntegrand(f.A, ω), f.l; npt1=f.npt1, pre1=f.pre1, npt2=f.npt2, pre2=f.pre2, atol=f.atol, rtol=f.rtol)
    f.npt1 = other.npt1
    f.pre1 = other.pre1
    f.npt2 = other.npt2
    f.pre2 = other.pre2
    return kinetic_integrand(Γ, ω, f.A.Ω, f.A.β, f.A.n)
end