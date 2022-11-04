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
export GreensFunction, SpectralFunction, DOSIntegrand, GammaIntegrand, OCIntegrand, EquispaceOCIntegrand, AutoEquispaceOCIntegrand

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
G(\\omega) = \\int_{\\text{BZ}} dk {((\\omega + \\mu) I - H(k) - \\Sigma(\\omega))}^{-1}
```
This type works with both adaptive and equispace integration routines.
"""
struct GreensFunction{TH<:FourierSeries,TM}
    H::TH
    M::TM
end

greens_function(H::FourierSeries, M) = GreensFunction(H, M)

GreensFunction(args...) = greens_function(args...)

Base.eltype(::Type{<:GreensFunction{TH}}) where {TH} = Base.promote_op(inv, eltype(TH))

(g::GreensFunction)(k) = evaluate_integrand(g, g.H(k))
evaluate_integrand(g::GreensFunction, H_k) = greens_function(H_k, g.M)

spectral_function(args...) = spectral_function(greens_function(args...))
spectral_function(G::AbstractMatrix) = imag(G)/(-pi)

"""
    SpectralFunction(::GreensFunction)

A type that whose integral gives the imaginary part of the Green's function.
```math
A(ω) = {\\pi}^{-1} \\Im[G(ω)]
```
This type works with both adaptive and equispace integration routines.
"""
struct SpectralFunction{TG<:GreensFunction}
    G::TG
end

spectral_function(G::GreensFunction) = SpectralFunction(G)

SpectralFunction(args...) = spectral_function(args...)

Base.eltype(::Type{<:SpectralFunction{TG}}) where {TG} = Base.promote_op(imag, eltype(TG))

(A::SpectralFunction)(k) = spectral_function(A.G(k))
evaluate_integrand(A::SpectralFunction, H_k) = spectral_function(evaluate_integrand(A.G, H_k))

dos_integrand(args...) = dos_integrand(spectral_function(args...))
dos_integrand(A::AbstractMatrix) = tr(A)

"""
    DOSIntegrand(::SpectralFunction)

A type whose integral gives the density of states.
```math
D(ω) = \\operatorname{Tr}[A(ω)]
```
This type works with both adaptive and equispace integration routines.
"""
struct DOSIntegrand{TA<:SpectralFunction}
    A::TA
end

dos_integrand(A::SpectralFunction) = DOSIntegrand(A)

DOSIntegrand(args...) = dos_integrand(args...)

Base.eltype(::Type{<:DOSIntegrand{TA}}) where {TA} = eltype(eltype(TA))

(D::DOSIntegrand)(k) = dos_integrand(D.A(k))
evaluate_integrand(D::DOSIntegrand, H_k) = dos_integrand(evaluate_integrand(D.A, H_k))


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
    SMatrix{3,3,ComplexF64,9}(map(tr, (ν₁Aω*ν₁AΩ, ν₂Aω*ν₁AΩ, ν₃Aω*ν₁AΩ, ν₁Aω*ν₂AΩ, ν₂Aω*ν₂AΩ, ν₃Aω*ν₂AΩ, ν₁Aω*ν₃AΩ, ν₂Aω*ν₃AΩ, ν₃Aω*ν₃AΩ)))
end

"""
    GammaIntegrand(H::FourierSeries, Σ, ω, Ω, μ)
    GammaIntegrand(HV, Σ, ω, Ω, μ)
    GammaIntegrand(HV, Mω, MΩ)

A type whose integral over the BZ gives the transport distribution.
```math
\\Gamma_{\\alpha\\beta}(\\omega, \\Omega) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega) \\nu_\\beta(k) A(k, \\omega+\\Omega)]
```
This type works with both adaptive and equispace integration routines.
"""
struct GammaIntegrand{T,M1,M2}
    HV::T
    Mω::M1
    MΩ::M2
end

GammaIntegrand(H::FourierSeries, Σ, ω, Ω, μ) = GammaIntegrand(BandEnergyVelocity(H), Σ, ω, Ω, μ)
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
    OCIntegrand(H::FourierSeries, Σ, Ω, β, μ)
    OCIntegrand(HV, Σ, Ω, β, μ)

A function whose integral over the BZ and the frequency axis gives the optical
conductivity. Mathematically, this computes
```math
\\sigma_{\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion. Use
this type only for adaptive integration and order the limits so that the
integral over the Brillouin zone is the outer integral and the frequency
integral is the inner integral.
"""
struct OCIntegrand{T,TS}
    HV::T
    Σ::TS
    Ω::Float64
    β::Float64
    μ::Float64
end

OCIntegrand(H::FourierSeries, Σ, Ω::Float64, β::Float64, μ::Float64) = OCIntegrand(BandEnergyVelocity(H), Σ, Ω, β, μ)
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
struct EquispaceOCIntegrand{T,TS,TL}
    σ::OCIntegrand{T,TS}
    l::TL
    npt::Int
    pre::Vector{Tuple{NTuple{4, SMatrix{3, 3, ComplexF64, 9}},Int}}
end
function EquispaceOCIntegrand(σ::OCIntegrand, l::IntegrationLimits, npt::Int)
    pre = equispace_pre_eval(σ, l, npt)
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