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

Returns `imag(tr(inv(M-H)))/(-pi)` where `M = ω*I-Σ(ω)`. It is unsafe to use
this in the inner integral for small eta due to the localized tails of the
integrand. The default, safe version also integrates the real part which is less
localized, at the expense of a slight slow-down due to complex arithmetic.
See [`AutoBZ.Jobs.safedos_integrand`](@ref).
"""
dos_integrand(H::AbstractMatrix, M) = imag(tr_inv(M-H))/(-pi)

"""
    safedos_integrand(H, Σ, ω)
    safedos_integrand(H, M)

Returns `tr(inv(M-H))/(-pi)` where `M = ω*I-Σ(ω)`, without. This safe version
also integrates the real part of DOS which is less localized, at the expense of
a slight slow-down due to complex arithmetic. See
[`AutoBZ.Jobs.dos_integrand`](@ref).
"""
safedos_integrand(H::AbstractMatrix, M) = tr_inv(M-H)/(-pi)


"""
    self_energy_args(M)
    self_energy_args(Σ, ω)

define the default behavior for handling single Green's function self energy arguments
"""
self_energy_args(M) = M
self_energy_args(Σ::AbstractSelfEnergy, ω) = ω*I-Σ(ω)
self_energy_args(Σ::AbstractMatrix, ω) = ω*I-Σ

# Generic behavior for single Green's function integrands (methods and types)
for name in ("Gloc", "DiagGloc", "DOS", "SafeDOS")
    # create and export symbols
    f = Symbol(lowercase(name), "_integrand")
    T = Symbol(name, "Integrand")
    E = Symbol(name, "Integrator")
    @eval export $f, $T, $E

    # Define interface for converting to the FourierIntegrand
    @eval $f(H::AbstractFourierSeries, args...) = FourierIntegrand($f, H, self_energy_args(args...))

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
function ExperimentalDOSIntegrand{N}(H::AbstractFourierSeries, args...) where N
    fs = ntuple(n -> n==1 ? safedos_integrand : (n==2 ? imag : identity), N)
    IteratedFourierIntegrand(fs, H, self_energy_args(args...))
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

export TransportIntegrand, TransportIntegrator, KineticIntegrand, KineticIntegrator

spectral_function(H, M) = imag(inv(M-H))/(-pi)

transport_integrand(HV, Σ::AbstractSelfEnergy, ω) = transport_integrand(HV, ω*I-Σ(ω))
transport_integrand(HV, Σ::AbstractSelfEnergy, ω₁, ω₂) = transport_integrand(HV, ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂))
transport_integrand(HV, Σ::AbstractMatrix, ω₁, ω₂) = transport_integrand(HV, ω₁*I-Σ, ω₂*I-Σ)
transport_integrand(HV::AbstractBandVelocity3D, Mω₁, Mω₂) = FourierIntegrand(transport_integrand, HV, Mω₁, Mω₂)
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


kinetic_integrand(HV, Σ, ω, Ω, β, n) = kinetic_integrand(transport_integrand(HV, Σ, ω, ω+Ω), ω, Ω, β, n)
kinetic_integrand(Γ, ω, Ω, β, n) = (ω*β)^n * β * fermi_window(β*ω, β*Ω) * Γ

function kinetic_frequency_integrator(HV::AbstractBandVelocity3D, n, Σ, β, Ω; routine=iterated_integration, kwargs...)
    lims = get_safe_freq_limits(Ω, β, lb(Σ), ub(Σ))
    f = let HV_k=HV(zero(SVector{3,Float64})), Σ=Σ, Ω=Ω, β=β, n=n
        ω -> kinetic_integrand(HV_k, Σ, ω, Ω, β, n)
    end
    FourierIntegrand(kinetic_frequency_integrator, HV, routine, lims, AutoBZ.default_kwargs(routine, f, lims; kwargs...), n, Σ, β, Ω)
end
function kinetic_frequency_integrator(HV_k::Tuple, routine, lims, kwargs, n, Σ, β, Ω=0)
    # deal with distributional integral case
    Ω == 0 && β == Inf && return 0^n * transport_integrand(HV_k, Σ, Ω)
    # normal case
    f = let HV_k=HV_k, Σ=Σ, Ω=Ω, β=β, n=n
        ω -> kinetic_integrand(HV_k, Σ, ω, Ω, β, n)
    end
    first(routine(f, lims; kwargs...))
end

"""
    KineticIntegrand(HV, n, Σ, β, [Ω=0]; routine=quadgk)

A function whose integral over the BZ gives the kinetic
coefficient. Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion. Use
this type only for adaptive integration and order the limits so that the
integral over the Brillouin zone is the outer integral and the frequency
integral is the inner integral. Based on
[TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const KineticIntegrand = FourierIntegrand{typeof(kinetic_frequency_integrator)}
KineticIntegrand(args...; kwargs...) = kinetic_frequency_integrator(args...; kwargs...)


"""
    KineticIntegrator(bz, HV, n, Σ, [β])

Integrates `KineticIntegrand` over `bz` as a function of `Ω`, or `β, Ω`.
See [`AutoBZ.FourierIntegrator`](@ref) for more details.
"""
const KineticIntegrator = FourierIntegrator{typeof(kinetic_frequency_integrator)}

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
