"""
    eval_self_energy(M)
    eval_self_energy(Σ, ω)

Defines the default behavior for handling single Green's function self energy
arguments
"""
eval_self_energy(Σ::AbstractSelfEnergy, ω) = ω*I-Σ(ω)
eval_self_energy(Σ::AbstractMatrix, ω) = ω*I-Σ


"""
    gloc_integrand(H, Σ, ω)
    gloc_integrand(H, M)

Returns `inv(M-H)` where `M = ω*I-Σ(ω)`
"""
gloc_integrand(H::AbstractMatrix, M) = inv(M-H)
gloc_integrand(H::AbstractMatrix, Σ, ω) = gloc_integrand(H, eval_self_energy(Σ, ω))

"""
    diaggloc_integrand(H, Σ, ω)
    diaggloc_integrand(H, M)

Returns `diag(inv(M-H))` where `M = ω*I-Σ(ω)`
"""
diaggloc_integrand(H::AbstractMatrix, M) = diag_inv(M-H)
diaggloc_integrand(H::AbstractMatrix, Σ, ω) = diaggloc_integrand(H, eval_self_energy(Σ, ω))

"""
    dos_integrand(H, Σ, ω)
    dos_integrand(H, M)

Returns `imag(tr(inv(M-H)))/(-pi)` where `M = ω*I-Σ(ω)`. It is unsafe to use
this in the inner integral for small eta due to the localized tails of the
integrand. The default, safe version also integrates the real part which is less
localized, at the expense of a slight slow-down due to complex arithmetic.
See [`AutoBZ.Jobs.safedos_integrand`](@ref).
"""
dos_integrand(H::AbstractMatrix, M) = imag(tr_inv(M-H))/(-pi)
dos_integrand(H::AbstractMatrix, Σ, ω) = dos_integrand(H, eval_self_energy(Σ, ω))

"""
    safedos_integrand(H, Σ, ω)
    safedos_integrand(H, M)

Returns `tr(inv(M-H))/(-pi)` where `M = ω*I-Σ(ω)`, without. This safe version
also integrates the real part of DOS which is less localized, at the expense of
a slight slow-down due to complex arithmetic. See
[`AutoBZ.Jobs.dos_integrand`](@ref).
"""
safedos_integrand(H::AbstractMatrix, M) = tr_inv(M-H)/(-pi)
safedos_integrand(H::AbstractMatrix, Σ, ω) = safedos_integrand(H, eval_self_energy(Σ, ω))

# Generic behavior for single Green's function integrands (methods and types)
for name in ("Gloc", "DiagGloc", "DOS", "SafeDOS")
    # create and export symbols
    f = Symbol(lowercase(name), "_integrand")
    T = Symbol(name, "Integrand")
    E = Symbol(name, "Integrator")
    @eval export $f, $T, $E

    # Define interface for converting to the FourierIntegrand
    @eval $f(H::AbstractFourierSeries{N}, args...) where N =
        FourierIntegrand{N}($f, H, eval_self_energy(args...))

    # Define the type alias to have the same behavior as the function
    """
    $(T)(H, Σ, ω)

    See [`AutoBZ.FourierIntegrand`](@ref) for more details.
    """
    @eval const $T = FourierIntegrand{typeof($f)}
    @eval $T(args...) = $f(args...)

    # Define an integrator based on the function
    """
    $(E)(bz, H, Σ)

    Integrates `$(T)` over `bz` as a function of `ω`.
    See [`AutoBZ.FourierIntegrator`](@ref) for more details.
    """
    @eval const $E = FourierIntegrator{typeof($f)}

    # Define default equispace grid stepping based 
    @eval function AutoSymPTR.npt_update(f::$T, npt::Integer)
        η = im_sigma_to_eta(-imag(f.p[1]))
        eta_npt_update(npt, η, period(f.s)[1])
    end

    # TODO: Define job scripts
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

# integrands using IteratedFourierIntegrand (definition is slightly more involved)

export ExperimentalDOSIntegrand, ExperimentalDOSIntegrator

"""
    ExperimentalDOSIntegrand(H, Σ, ω)

Constructor+Type alias for the DOS integrand implemented with
IteratedFourierIntegrand. Since taking the imaginary part after all of the
integrals doesn't cost much more than taking it after the first, use
[`AutoBZ.Jobs.SafeDOSIntegrand`](@ref). This type's purpose is to test
general-purpose iterated integration, which is incompatible with symmetries.
See [`AutoBZ.IteratedFourierIntegrand`](@ref) for more details.
"""
const ExperimentalDOSIntegrand{N} =
    IteratedFourierIntegrand{Tuple{typeof(safedos_integrand),typeof(imag),Vararg{typeof(identity),N}}}
ExperimentalDOSIntegrand(H::AbstractFourierSeries{N}, args...) where N = ExperimentalDOSIntegrand{N}(H, args...)
function ExperimentalDOSIntegrand{N}(H::AbstractFourierSeries{N}, args...) where N
    fs = ntuple(n -> n==1 ? safedos_integrand : (n==2 ? imag : identity), N)
    IteratedFourierIntegrand{N}(fs, H, eval_self_energy(args...))
end

# For type stability in the 1D case where imag must be taken outside
const ExperimentalDOSIntegrand1D =
    IteratedFourierIntegrand{Tuple{typeof(safedos_integrand)}}
AutoBZ.iterated_integrand(::ExperimentalDOSIntegrand1D, int, ::Type{Val{0}}) = imag(int)

"""
    ExperimentalDOSIntegrator(bz, H, Σ)

Integrates `ExperimentalDOSIntegrand` over `bz` as a function of `ω`.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const ExperimentalDOSIntegrator{N} =
    FourierIntegrator{Tuple{typeof(safedos_integrand),typeof(imag),Vararg{typeof(identity),N}}}

ExperimentalDOSIntegrator(lims::AbstractLimits{d}, args...; kwargs...) where d = ExperimentalDOSIntegrator{d}(lims, args...; kwargs...)

# transport and conductivity integrands

export TransportFunctionIntegrand, TransportFunctionIntegrator

transport_function_integrand(HV::AbstractHamiltonianVelocity, β) =
    FourierIntegrand{3}(transport_function_integrand, HV, β)
transport_function_integrand(HV, β) =
    transport_function_integrand(HV[1], HV[2], HV[3], HV[4], β)
function transport_function_integrand(H::Diagonal, v1, v2, v3, β)
    f′ = Diagonal(β .* fermi′.(β .* diag(H)))
    # TODO: reuse some of the matrix products and choose an efficient order
    SMatrix{3,3,T,9}((
        tr(v1*f′*v1), tr(v2*f′*v1), tr(v3*f′*v1),
        tr(v1*f′*v2), tr(v2*f′*v2), tr(v3*f′*v2),
        tr(v1*f′*v3), tr(v2*f′*v3), tr(v3*f′*v3),
    ))
end

"""
    TransportFunctionIntegrand(HV, β)

Computes the following integral
```math
\\D_{\\alpha\\beta} = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
"""
const TransportFunctionIntegrand =
    FourierIntegrand{typeof(transport_function_integrand)}
TransportFunctionIntegrand(; kwargs...) =
    transport_function_integrand(args...; kwargs...)


const TransportFunctionIntegrator =
    FourierIntegrator{typeof(transport_function_integrand)}

export TransportDistributionIntegrand, TransportDistributionIntegrator

spectral_function(H, M) = imag(inv(M-H))/(-pi)

transport_distribution_integrand(HV, Σ::AbstractSelfEnergy, ω) =
    transport_distribution_integrand(HV, ω*I-Σ(ω))
transport_distribution_integrand(HV, Σ::AbstractSelfEnergy, ω₁, ω₂) =
    transport_distribution_integrand(HV, ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂))
transport_distribution_integrand(HV, Σ::AbstractMatrix, ω₁, ω₂) =
    transport_distribution_integrand(HV, ω₁*I-Σ, ω₂*I-Σ)
transport_distribution_integrand(HV::AbstractHamiltonianVelocity, Mω₁, Mω₂) =
    FourierIntegrand{3}(transport_distribution_integrand, HV, Mω₁, Mω₂)
transport_distribution_integrand(HV::Tuple, Mω₁, Mω₂=Mω₁) =
    transport_distribution_integrand(HV[1], HV[2], HV[3], HV[4], Mω₁, Mω₂)
    # transport_distribution_integrand(HV..., Mω₁, Mω₂) # Splatting bad for inference
function transport_distribution_integrand(H, ν₁, ν₂, ν₃, Mω₁, Mω₂)
    # Aω₁ = spectral_function(H, Mω₁)
    # Aω₂ = spectral_function(H, Mω₂)
    # Probably missing a factor of (2*pi)^-3 to convert reciprocal space volume
    # to real space 1/V, V the volume of the unit cell
    # transport_distribution_integrand_(ν₁, ν₂, ν₃, Aω₁, Aω₂)
    transport_distribution_integrand_(ν₁, ν₂, ν₃, spectral_function(H, Mω₁), spectral_function(H, Mω₂))
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
    TransportDistributionIntegrand(HV, Σ, ω₁, ω₂)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See [`AutoBZ.FourierIntegrand`](@ref) for more details.
"""
const TransportDistributionIntegrand =
    FourierIntegrand{typeof(transport_distribution_integrand)}

TransportDistributionIntegrand(args...) =
    transport_distribution_integrand(args...)

"""
    TransportDistributionIntegrator(bz, HV, Σ, ω₁)

Integrates `TransportDistributionIntegrand` over `bz` as a function of `ω₂`.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const TransportDistributionIntegrator =
    FourierIntegrand{typeof(transport_distribution_integrand)}

function AutoSymPTR.npt_update(Γ::TransportDistributionIntegrand, npt::Integer)
    ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
    ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
    eta_npt_update(npt, min(ηω₁, ηω₂), period(Γ.s)[1])
end


export KineticCoefficientIntegrand, KineticCoefficientIntegrator

kinetic_coefficient_integrand(HV, Σ, ω, Ω, β, n) =
    kinetic_coefficient_integrand(transport_distribution_integrand(HV, Σ, ω, ω+Ω), ω, Ω, β, n)
kinetic_coefficient_integrand(Γ, ω, Ω, β, n) =
    (ω*β)^n * β * fermi_window(β*ω, β*Ω) * Γ

function kinetic_coefficient_frequency_integral(HV::AbstractHamiltonianVelocity, n, Σ, β, Ω; quad=quadgk, quad_kw=(;), kwargs...)
    lims = get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ))
    f = let HV_k=HV(zero(SVector{3,Float64})), Σ=Σ, Ω=Ω, β=β, n=n
        ω -> kinetic_coefficient_integrand(HV_k, Σ, ω, Ω, β, n)
    end
    FourierIntegrand{3}(kinetic_coefficient_frequency_integral, HV,
    quad, quad_args(quad, f, lims),
    quad_kwargs(quad, f, lims; quad_kw...),
    n, Σ, β, Ω; kwargs...)
end
kinetic_coefficient_frequency_integral(HV::AbstractHamiltonianVelocity, quad, args, kwargs, n, Σ, β, Ω) =
    FourierIntegrand{3}(kinetic_coefficient_frequency_integral, HV, quad, args, kwargs, n, Σ, β, Ω)

function kinetic_coefficient_frequency_integral(HV_k::Tuple, quad, args, kwargs, n, Σ, β, Ω=0)
    # deal with distributional integral case
    Ω == 0 && β == Inf && return 0^n * transport_distribution_integrand(HV_k, Σ, Ω)
    # normal case
    f = let HV_k=HV_k, Σ=Σ, Ω=Ω, β=β, n=n
        ω -> kinetic_coefficient_integrand(HV_k, Σ, ω, Ω, β, n)
    end
    first(quad(f, args...; kwargs...))
end

"""
    KineticCoefficientIntegrand(HV, n, Σ, β, Ω; quad=iterated_integration, quad_kw=(;))

A function whose integral over the BZ gives the kinetic
coefficient. Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const KineticCoefficientIntegrand =
    FourierIntegrand{typeof(kinetic_coefficient_frequency_integral)}
KineticCoefficientIntegrand(args...; kwargs...) =
    kinetic_coefficient_frequency_integral(args...; kwargs...)


# TODO, decide if β can be made a parameter as well
function kinetic_coefficient_integrator(bz, HV::AbstractHamiltonianVelocity, n, Σ, β; Ω=1.0, ps=Ω, quad=quadgk, quad_kw=(;), kwargs...)
    lims = get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ))
    f = let HV_k=HV(zero(SVector{3,Float64})), Σ=Σ, Ω=Ω, β=β, n=n
        ω -> kinetic_coefficient_integrand(HV_k, Σ, ω, Ω, β, n)
    end
    FourierIntegrator(kinetic_coefficient_frequency_integral, bz, HV,
    quad, quad_args(quad, f, lims), quad_kwargs(quad, f, lims; quad_kw...),
    n, Σ, β; ps=ps, kwargs...)
end

"""
    KineticCoefficientIntegrator(bz, HV, n, Σ, β)

Integrates `KineticCoefficientIntegrand` over `bz` as a function of `Ω`.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const KineticCoefficientIntegrator =
    FourierIntegrator{typeof(kinetic_coefficient_frequency_integral)}
KineticCoefficientIntegrator(args...; kwargs...) =
    kinetic_coefficient_integrator(args...; kwargs...)

export ElectronDensityIntegrand, ElectronDensityIntegrator

electron_density_integrand(H_k, Σ, ω, β, μ) =
    fermi(β*ω)*dos_integrand(H_k, (ω+μ)*I-Σ(ω)) # shift only energy, not self energy

function electron_density_frequency_integral(H::AbstractFourierSeries{N}, Σ, β, μ; quad=quadgk, quad_kw=(;), kwargs...) where N
    # choose limits as wide as self energy allows
    lims = (lb(Σ), ub(Σ))
    f = let H_k=H(zero(SVector{N,eltype(lims)})), Σ=Σ, β=β, μ=μ
        ω -> electron_density_integrand(H_k, Σ, ω, β, μ)
    end
    FourierIntegrand{N}(electron_density_frequency_integral, H,
    quad, quad_args(quad, f, lims), quad_kwargs(quad, f, lims; quad_kw...),
    Σ, β, μ; kwargs...)
end
electron_density_frequency_integral(H::AbstractFourierSeries{N}, quad, args, kwargs, Σ, β, μ) where N =
    FourierIntegrand{N}(electron_density_frequency_integral, H, quad, args, kwargs, Σ, β, μ)

function electron_density_frequency_integral(H_k::AbstractMatrix, quad, args, kwargs, Σ, β, μ)
    f = let H_k=H_k, Σ=Σ, β=β, μ=μ
        ω -> electron_density_integrand(H_k, Σ, ω, β, μ)
    end
    first(quad(f, args...; kwargs...))
end

"""
    ElectronDensityIntegrand(HV, Σ, β, μ; quad=iterated_integration)

A function whose integral over the BZ gives the electron density.
Mathematically, this computes
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const ElectronDensityIntegrand =
    FourierIntegrand{typeof(electron_density_frequency_integral)}
ElectronDensityIntegrand(args...; kwargs...) =
    electron_density_frequency_integral(args...; kwargs...)

function electron_density_integrator(bz, H::AbstractFourierSeries{N}, Σ, β; μ=0.0, ps=μ, quad=quadgk, quad_kw=(;), kwargs...) where N
    lims = (lb(Σ), ub(Σ))
    f = let H_k=H(zero(SVector{N,coefficient_type(bz)})), Σ=Σ, β=β, μ=μ
        ω -> electron_density_integrand(H_k, Σ, ω, β, μ)
    end
    FourierIntegrator(electron_density_frequency_integral, bz, H,
    quad, quad_args(quad, f, lims), quad_kwargs(quad, f, lims; quad_kw...),
    Σ, β; ps=ps, kwargs...)
end

"""
    ElectronDensityIntegrator(bz, HV, Σ, β)

Integrates `ElectronDensityIntegrand` over `bz` as a function of `μ`.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const ElectronDensityIntegrator =
    FourierIntegrator{typeof(electron_density_frequency_integral)}
ElectronDensityIntegrator(args...; kwargs...) =
    electron_density_integrator(args...; kwargs...)

# define how to symmetrize matrix-valued integrands with coordinate indices

"""
    CoordinateMatrixIntegrand

A type union of integrands whose return value is a matrix with coordinate indices
"""
const CoordinateMatrixIntegrand = Union{TransportDistributionIntegrand,TransportFunctionIntegrand,KineticCoefficientIntegrand}

# TODO: incorporate rotations to Cartesian basis due to lattice vectors
function AutoBZCore.symmetrize(::CoordinateMatrixIntegrand, l::SymmetricBZ, x::AbstractMatrix)
    r = zero(x)
    for S in symmetries(bz)
        r += S * x * S'
    end
    r
end

# replacement for TetrahedralLimits

cubic_sym_ibz(A; kwargs...) = cubic_sym_ibz(A, AutoBZ.canonical_reciprocal_basis(A); kwargs...)
cubic_sym_ibz(fbz::FullBZ; kwargs...) = cubic_sym_ibz(fbz.A, fbz.B; kwargs...)
function cubic_sym_ibz(A::M, B::M; kwargs...) where {N,T,M<:SMatrix{N,N,T}}
    nrmb = ntuple(n -> norm(B[:,n])/2, Val{N}())
    lims = TetrahedralLimits(nrmb)
    syms = vec(collect(cube_automorphisms(Val{N}())))
    SymmetricBZ(A, B, lims, syms; kwargs...)
end

"""
    cube_automorphisms(d::Integer)

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
