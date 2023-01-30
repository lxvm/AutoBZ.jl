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
    gloc_integrand(H, Σ, ω)
    gloc_integrand(H, M)

Returns `inv(M-H)` where `M = ω*I-Σ(ω)`
"""
gloc_integrand(H::AbstractMatrix, args...) = inv(eval_self_energy(args...)-H)

"""
    diaggloc_integrand(H, Σ, ω)
    diaggloc_integrand(H, M)

Returns `diag(inv(M-H))` where `M = ω*I-Σ(ω)`
"""
diaggloc_integrand(H::AbstractMatrix, args...) = diag_inv(eval_self_energy(args...)-H)

"""
    dos_integrand(H, Σ, ω)
    dos_integrand(H, M)

Returns `imag(tr(inv(M-H)))/(-pi)` where `M = ω*I-Σ(ω)`. It is unsafe to use
this in the inner integral for small eta due to the localized tails of the
integrand. The default, safe version also integrates the real part which is less
localized, at the expense of a slight slow-down due to complex arithmetic.
See [`AutoBZ.Jobs.safedos_integrand`](@ref).
"""
dos_integrand(H::AbstractMatrix, args...) = imag(tr_inv(eval_self_energy(args...)-H))/(-pi)

"""
    safedos_integrand(H, Σ, ω)
    safedos_integrand(H, M)

Returns `tr(inv(M-H))/(-pi)` where `M = ω*I-Σ(ω)`, without. This safe version
also integrates the real part of DOS which is less localized, at the expense of
a slight slow-down due to complex arithmetic. See
[`AutoBZ.Jobs.dos_integrand`](@ref).
"""
safedos_integrand(H::AbstractMatrix, args...) = tr_inv(eval_self_energy(args...)-H)/(-pi)

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
    @eval function AutoBZ.equispace_npt_update(f::$T, npt)
        η = im_sigma_to_eta(-imag(f.p[1]))
        npt_update_eta(npt, η, period(f.s)[1])
    end

    # TODO: Define job scripts
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
const ExperimentalDOSIntegrand{N} = IteratedFourierIntegrand{Tuple{typeof(safedos_integrand),typeof(imag),Vararg{typeof(identity),N}}}
ExperimentalDOSIntegrand(H::AbstractFourierSeries{N}, args...) where N = ExperimentalDOSIntegrand{N}(H, args...)
function ExperimentalDOSIntegrand{N}(H::AbstractFourierSeries{N}, args...) where N
    fs = ntuple(n -> n==1 ? safedos_integrand : (n==2 ? imag : identity), N)
    IteratedFourierIntegrand{N}(fs, H, eval_self_energy(args...))
end

# For type stability in the 1D case where imag must be taken outside
const ExperimentalDOSIntegrand1D = IteratedFourierIntegrand{Tuple{typeof(safedos_integrand)}}
AutoBZ.iterated_integrand(::ExperimentalDOSIntegrand1D, int, ::Type{Val{0}}) = imag(int)

"""
    ExperimentalDOSIntegrator(bz, H, Σ)

Integrates `ExperimentalDOSIntegrand` over `bz` as a function of `ω`.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const ExperimentalDOSIntegrator{N} = FourierIntegrator{Tuple{typeof(safedos_integrand),typeof(imag),Vararg{typeof(identity),N}}}

ExperimentalDOSIntegrator(lims::AbstractLimits{d}, args...; kwargs...) where d = ExperimentalDOSIntegrator{d}(lims, args...; kwargs...)

# transport and conductivity integrands

export TransportIntegrand, TransportIntegrator

spectral_function(H, M) = imag(inv(M-H))/(-pi)

transport_integrand(HV, Σ::AbstractSelfEnergy, ω) = transport_integrand(HV, ω*I-Σ(ω))
transport_integrand(HV, Σ::AbstractSelfEnergy, ω₁, ω₂) = transport_integrand(HV, ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂))
transport_integrand(HV, Σ::AbstractMatrix, ω₁, ω₂) = transport_integrand(HV, ω₁*I-Σ, ω₂*I-Σ)
transport_integrand(HV::AbstractBandVelocity3D, Mω₁, Mω₂) = FourierIntegrand{3}(transport_integrand, HV, Mω₁, Mω₂)
transport_integrand(HV::Tuple, Mω₁, Mω₂=Mω₁) = transport_integrand(HV..., Mω₁, Mω₂)
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

"""
    TransportIntegrand(HV, Σ, ω₁, ω₂)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See [`AutoBZ.FourierIntegrand`](@ref) for more details.
"""
const TransportIntegrand = FourierIntegrand{typeof(transport_integrand)}
TransportIntegrand(args...) = transport_integrand(args...)

"""
    TransportIntegrator(bz, HV, Σ, ω₁)

Integrates `TransportIntegrand` over `bz` as a function of `ω₂`.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const TransportIntegrator = FourierIntegrand{typeof(transport_integrand)}

function AutoBZ.equispace_npt_update(Γ::TransportIntegrand, npt)
    ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
    ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
    npt_update_eta(npt, min(ηω₁, ηω₂), period(Γ.s)[1])
end


export KineticIntegrand, KineticIntegrator

kinetic_integrand(HV, Σ, ω, Ω, β, n) = kinetic_integrand(transport_integrand(HV, Σ, ω, ω+Ω), ω, Ω, β, n)
kinetic_integrand(Γ, ω, Ω, β, n) = (ω*β)^n * β * fermi_window(β*ω, β*Ω) * Γ

function kinetic_frequency_integral(HV::AbstractBandVelocity3D, n, Σ, β, Ω; quad=iterated_integration, quad_kw=(;), kwargs...)
    lims = get_safe_freq_limits(Ω, β, lb(Σ), ub(Σ))
    f = let HV_k=HV(zero(SVector{3,Float64})), Σ=Σ, Ω=Ω, β=β, n=n
        ω -> kinetic_integrand(HV_k, Σ, ω, Ω, β, n)
    end
    FourierIntegrand{3}(kinetic_frequency_integral, HV,
    quad, quad_args(quad, f, lims),
    default_kwargs(quad, f, lims; quad_kw...),
    n, Σ, β, Ω; kwargs...)
end
kinetic_frequency_integral(HV::AbstractBandVelocity3D, quad, args, kwargs, n, Σ, β, Ω) =
    FourierIntegrand(kinetic_frequency_integral, HV, quad, args, kwargs, n, Σ, β, Ω)

function kinetic_frequency_integral(HV_k::Tuple, quad, args, kwargs, n, Σ, β, Ω=0)
    # deal with distributional integral case
    Ω == 0 && β == Inf && return 0^n * transport_integrand(HV_k, Σ, Ω)
    # normal case
    f = let HV_k=HV_k, Σ=Σ, Ω=Ω, β=β, n=n
        ω -> kinetic_integrand(HV_k, Σ, ω, Ω, β, n)
    end
    first(quad(f, args...; kwargs...))
end

"""
    KineticIntegrand(HV, n, Σ, β, Ω; quad=iterated_integration, quad_kw=(;))

A function whose integral over the BZ gives the kinetic
coefficient. Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const KineticIntegrand = FourierIntegrand{typeof(kinetic_frequency_integral)}
KineticIntegrand(args...; kwargs...) = kinetic_frequency_integral(args...; kwargs...)


# TODO, decide if β can be made a parameter as well
function kinetic_integrator(bz, HV::AbstractBandVelocity3D, n, Σ, β; Ω=1.0, ps=Ω, quad=iterated_integration, quad_kw=(;), kwargs...)
    lims = get_safe_freq_limits(Ω, β, lb(Σ), ub(Σ))
    f = let HV_k=HV(zero(SVector{3,Float64})), Σ=Σ, Ω=Ω, β=β, n=n
        ω -> kinetic_integrand(HV_k, Σ, ω, Ω, β, n)
    end
    FourierIntegrator(kinetic_frequency_integral, bz, HV,
    quad, quad_args(quad, f, lims), quad_kwargs(quad, f, lims; quad_kw...),
    n, Σ, β; ps=ps, kwargs...)
end

"""
    KineticIntegrator(bz, HV, n, Σ, β)

Integrates `KineticIntegrand` over `bz` as a function of `Ω`.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const KineticIntegrator = FourierIntegrator{typeof(kinetic_frequency_integral)}
KineticIntegrator(args...; kwargs...) = kinetic_integrator(args...; kwargs...)

export ElectronCountIntegrand, ElectronCountIntegrator

electron_count_integrand(H_k, Σ, ω, β, μ) = fermi(β*ω)*dos_integrand(H_k, Σ, ω+μ)

function electron_count_frequency_integral(H::AbstractFourierSeries{N}, Σ, β, μ; quad=iterated_integration, quad_kw=(;), lims=nothing, kwargs...) where N
    # TODO: see if there is a way to safely truncate limits
    f = let H_k=H(zero(SVector{3,Float64})), Σ=Σ, β=β, μ=μ
        ω -> electron_count_integrand(H_k, Σ, ω, β, μ)
    end
    FourierIntegrand{N}(electron_count_frequency_integral, H,
    quad, quad_args(quad, f, lims), quad_kwargs(quad, f, lims; quad_kw...),
    Σ, β, μ; kwargs...)
end
electron_count_frequency_integral(H::AbstractFourierSeries{N}, quad, args, kwargs, Σ, β, μ) where N =
    FourierIntegrand{N}(electron_count_frequency_integral, H, quad, args, kwargs, Σ, β, μ)

function electron_count_frequency_integral(H_k::AbstractMatrix, quad, args, kwargs, Σ, β, μ)
    f = let H_k=H_k, Σ=Σ, β=β, μ=μ
        ω -> electron_count_integrand(H_k, Σ, ω, β, μ)
    end
    first(quad(f, args...; kwargs...))
end

"""
    ElectronCountIntegrand(HV, Σ, β, μ; quad=iterated_integration)

A function whose integral over the BZ gives the electron count.
Mathematically, this computes
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const ElectronCountIntegrand = FourierIntegrand{typeof(electron_count_frequency_integral)}
ElectronCountIntegrand(args...; kwargs...) = electron_count_frequency_integral(args...; kwargs...)

function electron_count_integrator(bz, H::AbstractFourierSeries{N}, Σ, β; μ=0.0, ps=μ, quad=iterated_integration, lims=nothing, quad_kw=(;), kwargs...) where N
    # TODO: see if there is a way to safely truncate limits
    f = let H_k=H(zero(SVector{N,Float64})), Σ=Σ, β=β, μ=μ
        ω -> electron_count_integrand(H_k, Σ, ω, β, μ)
    end
    FourierIntegrator(electron_count_frequency_integral, bz, H,
    quad, quad_args(quad, f, lims), quad_kwargs(quad, f, lims; quad_kw...),
    Σ, β; ps=ps, kwargs...)
end

"""
    ElectronCountIntegrator(bz, HV, Σ, β)

Integrates `ElectronCountIntegrand` over `bz` as a function of `μ`.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const ElectronCountIntegrator = FourierIntegrator{typeof(electron_count_frequency_integral)}
ElectronCountIntegrator(args...; kwargs...) = electron_count_integrator(args...; kwargs...)

# replacement for TetrahedralLimits
cubic_sym_ibz(A; kwargs...) = cubic_sym_ibz(A, AutoBZ.canonical_reciprocal_basis(A); kwargs...)
cubic_sym_ibz(fbz::FullBZ; kwargs...) = cubic_sym_ibz(fbz.A, fbz.B; kwargs...)
function cubic_sym_ibz(A::M, B::M; kwargs...) where {N,T,M<:SMatrix{N,N,T}}
    F = float(T); AT = SVector{N,F}
    vert = unit_tetrahedron_vertices(AT)
    nrmb = ntuple(n -> norm(B[:,n])/2, Val{N}())
    hull = vrep(map(v -> nrmb .* v, vert), Line{F,AT}[], Ray{F,AT}[])
    lims = PolyhedralLimits(hull)
    syms = vec(collect(cube_automorphisms(Val{N}())))
    IrreducibleBZ(A, B, lims, syms; kwargs...)
end

function unit_tetrahedron_vertices(::Type{AT}) where {N,T,AT<:SVector{N,T}}
    vertices = Vector{AT}(undef, N+1)
    for n in 1:N+1
        vertices[n] = ntuple(i -> i < n ? T(1) : T(0), Val{N}())
    end
    vertices
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
