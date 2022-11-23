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
export DOSIntegrand, GammaIntegrand, OCIntegrand, EquispaceOCIntegrand, AutoEquispaceOCIntegrand

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

gamma_integrand(H, ν₁, ν₂, ν₃, Σ, ω, Ω, μ) = gamma_integrand(H, ν₁, ν₂, ν₃, (ω+μ)*I-Σ(ω), (ω+Ω+μ)*I-Σ(ω+Ω))
gamma_integrand(H, ν₁, ν₂, ν₃, Σ::AbstractMatrix, ω, Ω, μ) = gamma_integrand(H, ν₁, ν₂, ν₃, (ω+μ)*I-Σ, (ω+Ω+μ)*I-Σ)
function gamma_integrand(H, ν₁, ν₂, ν₃, Mω, MΩ)
    Aω = spectral_function(H, Mω)
    AΩ = spectral_function(H, MΩ)
    gamma_integrand_(ν₁, ν₂, ν₃, Aω, AΩ)
end
function gamma_integrand_(ν₁, ν₂, ν₃, Aω, AΩ)
    ν₁Aω = ν₁*Aω
    ν₂Aω = ν₂*Aω
    ν₃Aω = ν₃*Aω
    ν₁AΩ = ν₁*AΩ
    ν₂AΩ = ν₂*AΩ
    ν₃AΩ = ν₃*AΩ
    SMatrix{3,3,ComplexF64,9}((
        tr_mul(ν₁Aω, ν₁AΩ), tr_mul(ν₂Aω, ν₁AΩ), tr_mul(ν₃Aω, ν₁AΩ),
        tr_mul(ν₁Aω, ν₂AΩ), tr_mul(ν₂Aω, ν₂AΩ), tr_mul(ν₃Aω, ν₂AΩ),
        tr_mul(ν₁Aω, ν₃AΩ), tr_mul(ν₂Aω, ν₃AΩ), tr_mul(ν₃Aω, ν₃AΩ),
    ))
end

"""
    GammaIntegrand(HV, Σ, ω, Ω, μ)
    GammaIntegrand(HV, Mω, MΩ)

A type whose integral over the BZ gives the transport distribution.
```math
\\Gamma_{\\alpha\\beta}(\\omega, \\Omega) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega) \\nu_\\beta(k) A(k, \\omega+\\Omega)]
```
This type works with both adaptive and equispace integration routines. The
keyword `kind` determines the band velocity component (not yet implemented).
"""
struct GammaIntegrand{T,M1,M2}
    HV::T
    Mω::M1
    MΩ::M2
end

function GammaIntegrand(HV, Σ, ω, Ω, μ)
    Mω = (ω+μ)*I-Σ(ω)
    MΩ = (ω+Ω+μ)*I-Σ(ω+Ω)
    GammaIntegrand(HV, Mω, MΩ)
end
Base.eltype(::Type{<:GammaIntegrand}) = SMatrix{3,3,ComplexF64,9}
(g::GammaIntegrand)(k) = evaluate_integrand(g, g.HV(k))
evaluate_integrand(g::GammaIntegrand, HV_k) = gamma_integrand(HV_k..., g.Mω, g.MΩ)

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
    OCIntegrand(HV, Σ, Ω, β, μ)

A function whose integral over the BZ and the frequency axis gives the optical
conductivity. Mathematically, this computes
```math
\\sigma_{\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion. Use
this type only for adaptive integration and order the limits so that the
integral over the Brillouin zone is the outer integral and the frequency
integral is the inner integral. The keyword `kind` determines the
band velocity component (not yet implemented).
"""
struct OCIntegrand{T,TS}
    HV::T
    Σ::TS
    Ω::Float64
    β::Float64
    μ::Float64
    function OCIntegrand(HV::T, Σ::TS, Ω, β, μ) where {T,TS}
        β == Inf && Ω == 0 && error("Ω=1/β=0 encountered. This case requires distributional frequency integrals, or can be computed with a GammaIntegrand")
        new{T,TS}(HV, Σ, Ω, β, μ)
    end
end

Base.eltype(::Type{<:OCIntegrand}) = SMatrix{3,3,ComplexF64,9}
(f::OCIntegrand)(ω::SVector{1}) = f(only(ω))
(f::OCIntegrand)(ω::Number) = oc_integrand(value(f.HV)..., f.Σ, ω, f.Ω, f.β, f.μ)

GammaIntegrand(σ::OCIntegrand, ω::Float64) = GammaIntegrand(σ.HV, σ.Σ, ω, σ.Ω, σ.μ)

"""
    EquispaceOCIntegrand(σ::OCIntegrand, l, npt, pre::Vector{Tuple{NTuple{4, SMatrix{3, 3, ComplexF64, 9}},Int}})
    EquispaceOCIntegrand(σ, l, npt)

This type represents an `OCIntegrand`, `σ` integrated adaptively in frequency
and with equispace integration over the Brillouin zone with a fixed number of
grid points `npt`. The argument `l` should be an `IntegrationLimits` for just
the Brillouin zone. This type should be called by an adaptive integration
routine whose limits of integration are only the frequency variable.
"""
struct EquispaceOCIntegrand{T,TS,TL,THV}
    σ::OCIntegrand{T,TS}
    l::TL
    npt::Int
    pre::Vector{Tuple{THV,Int}}
end
function EquispaceOCIntegrand(σ::OCIntegrand, l::IntegrationLimits, npt::Int)
    pre = equispace_pre_eval(GammaIntegrand(σ, 0.0), l, npt)
    EquispaceOCIntegrand(σ, l, npt, pre)
end
(f::EquispaceOCIntegrand)(ω::SVector{1}) = f(only(ω))
function (f::EquispaceOCIntegrand)(ω::Number)
    g = GammaIntegrand(f.σ, ω)
    int, _ = equispace_integration(g, f.l, f.npt; pre=f.pre)
    return f.σ.β * fermi_window(ω, f.σ.Ω, f.σ.β) * int
end

Base.eltype(::Type{<:EquispaceOCIntegrand}) = SMatrix{3,3,ComplexF64,9}

"""
    AutoEquispaceOCIntegrand(σ, l, atol, rtol, npt1, pre1, npt2, pre2)
    AutoEquispaceOCIntegrand(σ, l, atol, rtol; npt1=0, pre1=Tuple{eltype(σ.HV),Int}[], npt2=0,pre2=Tuple{eltype(σ.HV),Int}[])

This type represents an `OCIntegrand`, `σ` integrated adaptively in frequency
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
mutable struct AutoEquispaceOCIntegrand{T,TS,TL,TH}
    σ::OCIntegrand{T,TS}
    l::TL
    atol::Float64
    rtol::Float64
    npt1::Int
    pre1::Vector{Tuple{TH,Int}}
    npt2::Int
    pre2::Vector{Tuple{TH,Int}}
end
AutoEquispaceOCIntegrand(σ, l, atol, rtol; npt1=0, pre1=Tuple{eltype(σ.HV),Int}[], npt2=0,pre2=Tuple{eltype(σ.HV),Int}[]) = AutoEquispaceOCIntegrand(σ, l, atol, rtol, npt1, pre1, npt2, pre2)

Base.eltype(::Type{<:AutoEquispaceOCIntegrand}) = SMatrix{3,3,ComplexF64,9}

(f::AutoEquispaceOCIntegrand)(ω::SVector{1}) = f(only(ω))
function (f::AutoEquispaceOCIntegrand)(ω::Number)
    g = GammaIntegrand(f.σ, ω)
    int, err, other = automatic_equispace_integration(g, f.l; npt1=f.npt1, pre1=f.pre1, npt2=f.npt2, pre2=f.pre2, atol=f.atol, rtol=f.rtol)
    f.npt1 = other.npt1
    f.pre1 = other.pre1
    f.npt2 = other.npt2
    f.pre2 = other.pre2
    return f.σ.β * fermi_window(ω, f.σ.Ω, f.σ.β) * int
end