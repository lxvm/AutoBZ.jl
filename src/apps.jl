"""
    eval_self_energy(M)
    eval_self_energy(Σ, ω)

Defines the default behavior for handling single Green's function self energy
arguments
"""
eval_self_energy(M) = M
eval_self_energy(Σ::AbstractSelfEnergy, ω) = ω*I-Σ(ω)
eval_self_energy(Σ::AbstractMatrix, ω) = ω*I-Σ


"""
    gloc_integrand(h, Σ, ω)
    gloc_integrand(h, M)

Returns `inv(M-h)` where `M = ω*I-Σ(ω)`
"""
gloc_integrand(h::AbstractMatrix, M) = inv(M-h)
gloc_integrand(h::AbstractMatrix, Σ, ω) = gloc_integrand(h, eval_self_energy(Σ, ω))

"""
    diaggloc_integrand(h, Σ, ω)
    diaggloc_integrand(h, M)

Returns `diag(inv(M-h))` where `M = ω*I-Σ(ω)`
"""
diaggloc_integrand(h::AbstractMatrix, M) = diag_inv(M-h)
diaggloc_integrand(h::AbstractMatrix, Σ, ω) = diaggloc_integrand(h, eval_self_energy(Σ, ω))

"""
    dos_integrand(h, Σ, ω)
    dos_integrand(h, M)

Returns `imag(tr(inv(M-h)))/(-pi)` where `M = ω*I-Σ(ω)`. It is unsafe to use
this in the inner integral for small eta due to the localized tails of the
integrand. The default, safe version also integrates the real part which is less
localized, at the expense of a slight slow-down due to complex arithmetic.
See [`AutoBZ.Jobs.safedos_integrand`](@ref).
"""
dos_integrand(h::AbstractMatrix, M) = imag(tr_inv(M-h))/(-pi)
dos_integrand(h::Eigen, M::UniformScaling) = dos_integrand(Diagonal(h.values), M)
dos_integrand(h::Eigen, M::AbstractMatrix) = error("not implemented")#imag(tr_inv(M-h))/(-pi)
dos_integrand(h::AbstractMatrix, Σ, ω) = dos_integrand(h, eval_self_energy(Σ, ω))

"""
    safedos_integrand(h, Σ, ω)
    safedos_integrand(h, M)

Returns `tr(inv(M-h))/(-pi)` where `M = ω*I-Σ(ω)`, without. This safe version
also integrates the real part of DOS which is less localized, at the expense of
a slight slow-down due to complex arithmetic. See
[`AutoBZ.Jobs.dos_integrand`](@ref).
"""
safedos_integrand(h::AbstractMatrix, M) = tr_inv(M-h)/(-pi)
safedos_integrand(h::AbstractMatrix, Σ, ω) = safedos_integrand(h, eval_self_energy(Σ, ω))

# Generic behavior for single Green's function integrands (methods and types)
for name in ("Gloc", "DiagGloc", "DOS", "SafeDOS")
    # create and export symbols
    f = Symbol(lowercase(name), "_integrand")
    T = Symbol(name, "Integrand")
    E = Symbol(name, "Integrator")

    # Define interface for converting to the FourierIntegrand
    @eval $f(h::AbstractFourierSeries, args...) =
        FourierIntegrand($f, h, eval_self_energy(args...))

    # Define the type alias to have the same behavior as the function
    """
    $(T)(h, Σ, ω)

    See `FourierIntegrand` for more details.
    """
    @eval const $T = FourierIntegrand{typeof($f)}
    @eval $T(h::AbstractFourierSeries, args...) = $f(h, args...)

    # Define an integrator based on the function
    """
    $(E)(routine, bz, h, Σ)

    Integrates `$(T)` over `bz` as a function of `ω`.
    See `AutoBZCore.FourierIntegrator` for more details.
    """
    @eval const $E = FourierIntegrator{typeof($f)}

    # Define default equispace grid stepping based 
    @eval function AutoSymPTR.npt_update(f::$T, npt::Integer)
        η = im_sigma_to_eta(-imag(f.p[1]))
        eta_npt_update(npt, η, period(f.s)[1])
    end
end

# helper functions for equispace updates

im_sigma_to_eta(x::UniformScaling) = -x.λ
im_sigma_to_eta(x::AbstractMatrix) = minimum(abs, x)

"""
    eta_npt_update(npt, η, [n₀=6.0, Δn=2.3])

Implements the heuristics for incrementing kpts suggested in Appendix A
http://arxiv.org/abs/2211.12959. Choice of `n₀≈2π`, close to the period of a
canonical BZ, approximately places a point in every box of size `η`. Choice of
`Δn≈log(10)` should get an extra digit of accuracy from PTR upon refinement.
"""
eta_npt_update(npt, η, n₀=6.0, Δn=2.3) = npt + eta_npt_update_(η, npt == 0 ? n₀ : Δn)
eta_npt_update_(η, c) = max(50, round(Int, c/η))

# transport and conductivity integrands

transport_function_integrand(hv::AbstractVelocity, β) =
    FourierIntegrand{ndims(hv)}(transport_function_integrand, hv, β)
transport_function_integrand(hv, β) =
    transport_function_integrand(hv[1], hv[2], hv[3], hv[4], β)
function transport_function_integrand(h::Diagonal, v1, v2, v3, β)
    f′ = Diagonal(β .* fermi′.(β .* diag(h)))
    # TODO: reuse some of the matrix products and choose an efficient order
    SMatrix{3,3,T,9}((
        tr(v1*f′*v1), tr(v2*f′*v1), tr(v3*f′*v1),
        tr(v1*f′*v2), tr(v2*f′*v2), tr(v3*f′*v2),
        tr(v1*f′*v3), tr(v2*f′*v3), tr(v3*f′*v3),
    ))
end

"""
    TransportFunctionIntegrand(hv, β)

Computes the following integral
```math
\\D_{\\alpha\\beta} = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
"""
const TransportFunctionIntegrand =
    FourierIntegrand{typeof(transport_function_integrand)}

const TransportFunctionIntegrator =
    FourierIntegrator{typeof(transport_function_integrand)}


spectral_function(h, M) = imag(inv(M-h))/(-pi)

transport_distribution_integrand(hv, Σ::AbstractSelfEnergy, ω₁, ω₂) =
    transport_distribution_integrand(hv, ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂))
transport_distribution_integrand(hv, Σ::AbstractMatrix, ω₁, ω₂) =
    transport_distribution_integrand(hv, ω₁*I-Σ, ω₂*I-Σ)
transport_distribution_integrand(hv::AbstractVelocity, Mω₁, Mω₂) =
    FourierIntegrand(transport_distribution_integrand, hv, Mω₁, Mω₂)
transport_distribution_integrand(hv::Tuple, Mω₁, Mω₂) =
transport_distribution_integrand(hv..., Mω₁, Mω₂) # Splatting bad for inference
# transport_distribution_integrand(hv[1], hv[2], hv[3], hv[4], Mω₁, Mω₂)
function transport_distribution_integrand(h, ν₁, ν₂, ν₃, Mω₁, Mω₂)
    # Aω₁ = spectral_function(h, Mω₁)
    # Aω₂ = spectral_function(h, Mω₂)
    # Probably missing a factor of (2*pi)^-3 to convert reciprocal space volume
    # to real space 1/V, V the volume of the unit cell
    # transport_distribution_integrand_(ν₁, ν₂, ν₃, Aω₁, Aω₂)
    transport_distribution_integrand_(ν₁, ν₂, ν₃, spectral_function(h, Mω₁), spectral_function(h, Mω₂))
end
function transport_distribution_integrand_(ν₁::V, ν₂::V, ν₃::V, Aω₁::A, Aω₂::A) where {V,A}
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

"""
    TransportDistributionIntegrand(hv, Σ, ω₁, ω₂)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See `FourierIntegrand` for more details.
"""
const TransportDistributionIntegrand =
FourierIntegrand{typeof(transport_distribution_integrand)}
TransportDistributionIntegrand(hv::AbstractVelocity, args...) =
    transport_distribution_integrand(hv, args...)

"""
    TransportDistributionIntegrator(bz, hv, Σ, [ω₁, ω₂])

Integrates `TransportDistributionIntegrand` over `bz` as a function of `ω₁,ω₂`.
See `FourierIntegrator` for more details.
"""
const TransportDistributionIntegrator =
    FourierIntegrator{typeof(transport_distribution_integrand)}

function AutoSymPTR.npt_update(Γ::TransportDistributionIntegrand, npt::Integer)
    ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
    ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
    eta_npt_update(npt, min(ηω₁, ηω₂), period(Γ.s)[1])
end



"""
    get_safe_fermi_window_limits(Ω, β, lb, ub)

Given a frequency, `Ω`, inverse temperature, `β`,  returns an interval `(l,u)`
with possibly truncated limits of integration for the frequency integral at each
`(Ω, β)` point that are determined by the [`fermi_window_limits`](@ref) routine
set to the default tolerances for the decay of the Fermi window function. The
arguments `lb` and `ub` are lower and upper limits on the frequency to which the
default result gets truncated if the default result would recommend a wider
interval. If there is any truncation, a warning is emitted to the user, but the
program will continue with the truncated limits.
"""
function get_safe_fermi_window_limits(Ω, β, lb, ub)
    l, u = fermi_window_limits(Ω, β)
    if l < lb
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from below"
        l = lb
    end
    if u > ub
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from above"
        u = ub
    end
    l, u
end

kinetic_coefficient_integrand(hv, Σ, ω, Ω, β, n) =
    kinetic_coefficient_integrand(transport_distribution_integrand(hv, Σ, ω, ω+Ω), ω, Ω, β, n)
kinetic_coefficient_integrand(Γ, ω, Ω, β, n) =
    (ω*β)^n * β * fermi_window(β*ω, β*Ω) * Γ

function kinetic_coefficient_frequency_integral(hv_k::Tuple, Σ, n, β, Ω; quad=quadgk, quad_kw=(;), l=get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ)))
    # deal with distributional integral case
    Ω == 0 && β == Inf && return 0^n * transport_distribution_integrand(hv_k, Σ, Ω, Ω)
    # normal case
    f = let hv_k=hv_k, Σ=Σ, Ω=Ω, β=β, n=n
        ω -> kinetic_coefficient_integrand(hv_k, Σ, ω, Ω, β, n)
    end
    first(quad(quad_args(quad, l, f)...; quad_kw...))
end

"""
    KineticCoefficientIntegrand(hv, Σ, n, β, Ω; quad=iterated_integration, quad_kw=(;))

A function whose integral over the BZ gives the kinetic
coefficient. Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See `FourierIntegrator` for more details.
"""
const KineticCoefficientIntegrand =
    FourierIntegrand{typeof(kinetic_coefficient_frequency_integral)}

"""
    KineticCoefficientIntegrator(bz, hv, Σ, [n, β, Ω])

Integrates `KineticCoefficientIntegrand` over `bz` as a function of `n, β, Ω`.
See `FourierIntegrator` for more details. The exact choice of
parameters is up to the caller.
"""
const KineticCoefficientIntegrator =
    FourierIntegrator{typeof(kinetic_coefficient_frequency_integral)}

"""
    OpticalConductivityIntegrator(bz, hv, Σ, [β, Ω]; kwargs...)

Identical to [`KineticCoefficientIntegrator`](@ref) with `n=0`.
"""
OpticalConductivityIntegrator(routine, bz, hv, Σ, p...; kwargs...) =
    KineticCoefficientIntegrator(routine, bz, hv, Σ, 0, p...; kwargs...)


electron_density_integrand(h_k, Σ, ω, β, μ) =
    fermi(β*ω)*dos_integrand(h_k, (ω+μ)*I-Σ(ω)) # shift only energy, not self energy

function electron_density_frequency_integral(h_k::AbstractMatrix, quad, args, kwargs, Σ, β, μ)
    f = let h_k=h_k, Σ=Σ, β=β, μ=μ
        ω -> electron_density_integrand(h_k, Σ, ω, β, μ)
    end
    first(quad(f, args...; kwargs...))
end

"""
    ElectronDensityIntegrand(h, Σ, β, μ; quad=iterated_integration)

A function whose integral over the BZ gives the electron density.
Mathematically, this computes
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
See `FourierIntegrator` for more details.
"""
const ElectronDensityIntegrand =
    FourierIntegrand{typeof(electron_density_frequency_integral)}

"""
    ElectronDensityIntegrator(bz, h, Σ, β)

Integrates `ElectronDensityIntegrand` over `bz` as a function of `μ`.
See `FourierIntegrator` for more details.
"""
const ElectronDensityIntegrator =
    FourierIntegrator{typeof(electron_density_frequency_integral)}

# define how to symmetrize matrix-valued integrands with coordinate indices

"""
    CoordinateMatrixIntegrand

A type union of integrands whose return value is a matrix with coordinate indices
"""
const CoordinateMatrixIntegrand = Union{TransportDistributionIntegrand,TransportFunctionIntegrand,KineticCoefficientIntegrand}

# TODO: incorporate rotations to Cartesian basis due to lattice vectors
symmetrize(::CoordinateMatrixIntegrand, ::FullBZ, x::AbstractMatrix) = x
function symmetrize(::CoordinateMatrixIntegrand, bz::SymmetricBZ, x::AbstractMatrix)
    r = zero(x)
    for S in bz.syms
        r += S * x * S'
    end
    r
end

# replacement for TetrahedralLimits

cubic_sym_ibz(A; kwargs...) = cubic_sym_ibz(A, AutoBZ.canonical_reciprocal_basis(A); kwargs...)
cubic_sym_ibz(fbz::FullBZ; kwargs...) = cubic_sym_ibz(fbz.A, fbz.B; kwargs...)
function cubic_sym_ibz(A::M, B::M; kwargs...) where {N,T,M<:SMatrix{N,N,T}}
    lims = TetrahedralLimits(ntuple(n -> 1/2, Val{N}()))
    syms = vec(collect(cube_automorphisms(Val{N}())))
    SymmetricBZ(A, B, lims, syms; kwargs...)
end

"""
    cube_automorphisms(::Val{d}) where d

return a generator of the symmetries of the cube in `d` dimensions including the
identity.
"""
cube_automorphisms(n::Val{d}) where {d} = (S*P for S in sign_flip_matrices(n), P in permutation_matrices(n))
n_cube_automorphisms(d) = n_sign_flips(d) * n_permutations(d)

sign_flip_tuples(n::Val{d}) where {d} = Iterators.product(ntuple(_ -> (1,-1), n)...)
sign_flip_matrices(n::Val{d}) where {d} = (Diagonal(SVector{d,Int}(A)) for A in sign_flip_tuples(n))
n_sign_flips(d::Integer) = 2^d

function permutation_matrices(t::Val{n}) where {n}
    permutations = permutation_tuples(ntuple(identity, t))
    (StaticArrays.sacollect(SMatrix{n,n,Int,n^2}, ifelse(j == p[i], 1, 0) for i in 1:n, j in 1:n) for p in permutations)
end
permutation_tuples(C::NTuple{N,T}) where {N,T} = @inbounds((C[i], p...)::NTuple{N,T} for i in eachindex(C) for p in permutation_tuples(C[[j for j in eachindex(C) if j != i]]))
permutation_tuples(C::NTuple{1}) = C
n_permutations(n::Integer) = factorial(n)
