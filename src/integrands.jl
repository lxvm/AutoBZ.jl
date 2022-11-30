export WannierIntegrand

"""
    WannierIntegrand(f, s::AbstractFourierSeries, p)

A type generically representing an integrand `f` whose entire dependence on the
variables of integration is in a Fourier series `s`, and which may also accept
some input parameters `p`, which are preferrably contained in a tuple. The
caller must be aware that their function, `f`, will be called at many evaluation
points, `x`, in the following way: `f(s(x), p...)`. Therefore the caller is
expected to know the type of `s(x)` (hint: `eltype(s)`) and the layout of the
parameters in the tuple `p`.
"""
struct WannierIntegrand{TF,TS<:AbstractFourierSeries,TP}
    f::TF
    s::TS
    p::TP
end
(w::WannierIntegrand)(x) = evaluate_integrand(w, w.s(x))
evaluate_integrand(w::WannierIntegrand, s_x) = w.f(s_x, w.p...)

# pre-defined integrands
export DOSIntegrand, TransportIntegrand, KineticIntegrand, EquispaceKineticIntegrand, AutoEquispaceKineticIntegrand

dos_integrand(H, ω, Σ, μ) = dos_integrand(H, (ω+μ)*I-Σ(ω))
dos_integrand(H, ω, Σ::AbstractMatrix, μ) = dos_integrand(H, (ω+μ)*I-Σ)
dos_integrand(H, ω, Σ) = dos_integrand(H, ω*I-Σ(ω))
dos_integrand(H, ω, Σ::AbstractMatrix) = dos_integrand(H, ω*I-Σ)
dos_integrand(H::AbstractMatrix, M) = imag(tr_inv(M-H))/(-pi)

"""
    DOSIntegrand(H, ω, Σ, μ)
    DOSIntegrand(H, ω, Σ)
    DOSIntegrand(H, M)

A type whose integral gives the density of states.
```math
D(ω) = -{\\pi}^{-1} \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\Im[{((\\omega + \\mu) I - H(k) - \\Sigma(\\omega))}^{-1}]]
```
This type works with both adaptive and equispace integration routines.
"""
struct DOSIntegrand{TH<:AbstractFourierSeries,TM}
    H::TH
    M::TM
end

dos_integrand(H::AbstractFourierSeries, M) = DOSIntegrand(H, M)

DOSIntegrand(args...) = dos_integrand(args...)

Base.eltype(::Type{<:DOSIntegrand{TH}}) where {TH} = eltype(Base.promote_op(imag, Base.promote_op(inv, eltype(TH))))

(D::DOSIntegrand)(k) = evaluate_integrand(D, D.H(k))
evaluate_integrand(D::DOSIntegrand, H_k) = dos_integrand(H_k, D.M)

spectral_function(H, M) = imag(inv(M-H))/(-pi)

transport_integrand(H, ν₁, ν₂, ν₃, Σ, ω₁, ω₂, μ) = transport_integrand(H, ν₁, ν₂, ν₃, (ω₁+μ)*I-Σ(ω₁), (ω₂+μ)*I-Σ(ω₂))
transport_integrand(H, ν₁, ν₂, ν₃, Σ::AbstractMatrix, ω₁, ω₂, μ) = transport_integrand(H, ν₁, ν₂, ν₃, (ω₁+μ)*I-Σ, (ω₂+μ)*I-Σ)
function transport_integrand(H, ν₁, ν₂, ν₃, Mω₁, Mω₂)
    Aω₁ = spectral_function(H, Mω₁)
    Aω₂ = spectral_function(H, Mω₂)
    # Probably missing a factor of (2*pi)^-3 to convert reciprocal space volume
    # to real space 1/V, V the volume of the unit cell
    transport_integrand_(ν₁, ν₂, ν₃, Aω₁, Aω₂)
end
function transport_integrand_(ν₁, ν₂, ν₃, Aω₁, Aω₂)
    ν₁Aω₁ = ν₁*Aω₁
    ν₂Aω₁ = ν₂*Aω₁
    ν₃Aω₁ = ν₃*Aω₁
    ν₁Aω₂ = ν₁*Aω₂
    ν₂Aω₂ = ν₂*Aω₂
    ν₃Aω₂ = ν₃*Aω₂
    SMatrix{3,3,ComplexF64,9}((
        tr_mul(ν₁Aω₁, ν₁Aω₂), tr_mul(ν₂Aω₁, ν₁Aω₂), tr_mul(ν₃Aω₁, ν₁Aω₂),
        tr_mul(ν₁Aω₁, ν₂Aω₂), tr_mul(ν₂Aω₁, ν₂Aω₂), tr_mul(ν₃Aω₁, ν₂Aω₂),
        tr_mul(ν₁Aω₁, ν₃Aω₂), tr_mul(ν₂Aω₁, ν₃Aω₂), tr_mul(ν₃Aω₁, ν₃Aω₂),
    ))
end

"""
    TransportIntegrand(HV, Σ, ω₁, ω₂, μ)
    TransportIntegrand(HV, Mω₁, Mω₂)

A type whose integral over the BZ gives the transport distribution.
```math
\\Gamma_{\\alpha\\beta}(\\omega, \\Omega) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega) \\nu_\\beta(k) A(k, \\omega+\\Omega)]
```
This type works with both adaptive and equispace integration routines. Based on
https://triqs.github.io/dft_tools/latest/guide/transport.html#wien2k-optics-package
"""
struct TransportIntegrand{T<:Union{BandEnergyVelocity3D,BandEnergyBerryVelocity3D},M1,M2}
    HV::T
    Mω₁::M1
    Mω₂::M2
end

function TransportIntegrand(HV, Σ, ω₁, ω₂, μ)
    Mω₁ = (ω₁+μ)*I-Σ(ω₁)
    Mω₂ = (ω₂+μ)*I-Σ(ω₂)
    TransportIntegrand(HV, Mω₁, Mω₂)
end
Base.eltype(::Type{<:TransportIntegrand}) = SMatrix{3,3,ComplexF64,9}
(Γ::TransportIntegrand)(k) = evaluate_integrand(Γ, Γ.HV(k))
evaluate_integrand(Γ::TransportIntegrand, HV_k) = transport_integrand(HV_k..., Γ.Mω₁, Γ.Mω₂)

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

kinetic_integrand(H, ν₁, ν₂, ν₃, Σ, ω, Ω, β, μ, n) = kinetic_integrand(transport_integrand(H, ν₁, ν₂, ν₃, Σ, ω, ω+Ω, μ), ω, Ω, β, n)
kinetic_integrand(Γ, ω, Ω, β, n) = (ω*β)^n * β * fermi_window(ω, Ω, β) * Γ

"""
    KineticIntegrand(HV, Σ, β, μ, n, [Ω=0])

A function whose integral over the BZ and the frequency axis gives the kinetic
coefficient. Mathematically, this computes
```math
\\A{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Transport_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion. Use
this type only for adaptive integration and order the limits so that the
integral over the Brillouin zone is the outer integral and the frequency
integral is the inner integral. Based on
https://triqs.github.io/dft_tools/latest/guide/transport.html#wien2k-optics-package
"""
struct KineticIntegrand{T<:Union{BandEnergyVelocity3D,BandEnergyBerryVelocity3D},TS<:AbstractSelfEnergy}
    HV::T
    Σ::TS
    β::Float64
    μ::Float64
    n::Int
    Ω::Float64
    function KineticIntegrand(HV::T, Σ::TS, β, μ, n, Ω=0.0) where {T,TS}
        β == Inf && Ω == 0 && error("Ω=1/β=0 encountered. This case requires distributional frequency integrals, or can be computed with a TransportIntegrand")
        new{T,TS}(HV, Σ, β, μ, n, Ω)
    end
end

Base.eltype(::Type{<:KineticIntegrand}) = SMatrix{3,3,ComplexF64,9}
# innermost integral is the frequency integral
(A::KineticIntegrand)(ω::SVector{1}) = A(only(ω))
(A::KineticIntegrand)(ω::Number) = kinetic_integrand(value(A.HV)..., A.Σ, ω, A.Ω, A.β, A.μ, A.n)

TransportIntegrand(A::KineticIntegrand, ω::Float64) = TransportIntegrand(A.HV, A.Σ, ω, ω+A.Ω, A.μ)

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