propagator_denominator(h::AbstractMatrix, M) = M-h
propagator_denominator(h::Eigen, M::UniformScaling) =
    propagator_denominator(Diagonal(h.values), M)
propagator_denominator((h, U)::Eigen, M::AbstractMatrix) =
    propagator_denominator(Diagonal(h), U' * M * U) # rotate to Hamiltonian gauge

"""
    gloc_integrand(h, M)

Returns `inv(M-h)` where `M = ω*I-Σ(ω)`
"""
gloc_integrand(h, M) = inv(propagator_denominator(h, M))

"""
    diaggloc_integrand(h, M)

Returns `diag(inv(M-h))` where `M = ω*I-Σ(ω)`
"""
diaggloc_integrand(h, M) = diag_inv(propagator_denominator(h, M))

"""
    trgloc_integrand(h, M)

Returns `tr(inv(M-h))` where `M = ω*I-Σ(ω)`
"""
trgloc_integrand(h, M) = tr_inv(propagator_denominator(h, M))

"""
    dos_integrand(h, M)

Returns `imag(tr(inv(M-h)))/(-pi)` where `M = ω*I-Σ(ω)`. It is unsafe to use
this in the inner integral for small eta due to the localized tails of the
integrand. The default, safe version also integrates the real part which is less
localized, at the expense of a slight slow-down due to complex arithmetic.
See [`AutoBZ.Jobs.safedos_integrand`](@ref).
"""
dos_integrand(h, M) = imag(tr_inv(propagator_denominator(h, M)))/(-pi)


# Generic behavior for single Green's function integrands (methods and types)
for name in ("Gloc", "DiagGloc", "TrGloc", "DOS")
    # create and export symbols
    f = Symbol(lowercase(name), "_integrand")
    T = Symbol(name, "Integrand")

    # Define the type alias to have the same behavior as the function
    """
    $(T)(h, Σ, ω)

    See `FourierIntegrand` for more details.
    """
    @eval $T(h::Hamiltonian, p...) = FourierIntegrand($f, h, p)
    
    # pre-evaluate the self energy when constructing the integrand
    @eval FourierIntegrand(f::typeof($f), h::Hamiltonian, (Σ, ω)::Tuple{AbstractSelfEnergy,Real}) =
        FourierIntegrand(f, h, (ω*I-Σ(ω),))

    # Define default equispace grid stepping based 
    @eval function npt_update(f::FourierIntegrand{typeof($f)}, npt::Integer)
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
TransportFunctionIntegrand(hv::AbstractVelocity, p...) =
    FourierIntegrand(transport_function_integrand, hv, p)

SymRep(::FourierIntegrand{typeof(transport_function_integrand)}) = LatticeRep()


spectral_function(h, M) = imag(gloc_integrand(h, M))/(-pi)

transport_distribution_integrand(hv, Σ::AbstractSelfEnergy, ω₁, ω₂) =
    transport_distribution_integrand(hv, ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂))
transport_distribution_integrand(hv, Σ::AbstractMatrix, ω₁, ω₂) =
    transport_distribution_integrand(hv, ω₁*I-Σ, ω₂*I-Σ)
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
TransportDistributionIntegrand(hv::AbstractVelocity, p...) =
    FourierIntegrand(transport_distribution_integrand, hv, p)
    
# pre-evaluate self energies when constructing integrand
FourierIntegrand(f::typeof(transport_distribution_integrand), hv::AbstractVelocity, (Σ, ω₁, ω₂)::Tuple{AbstractSelfEnergy,Real,Real}) = 
    FourierIntegrand(f, hv, (ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂)))

const TransportDistributionIntegrandType = FourierIntegrand{typeof(transport_distribution_integrand)}

SymRep(::TransportDistributionIntegrandType) = LatticeRep()

function npt_update(Γ::TransportDistributionIntegrandType, npt::Integer)
    ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
    ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
    eta_npt_update(npt, min(ηω₁, ηω₂), period(Γ.s)[1])
end



transport_fermi_integrand(ω, Γ, n, β, Ω=0.0) =
    (ω*β)^n * β * fermi_window(β*ω, β*Ω) * Γ((ω, ω+Ω))

kinetic_coefficient_integrand(ω, Σ, hv_k, n, β, Ω) =
    (ω*β)^n * β * fermi_window(β*ω, β*Ω) * transport_distribution_integrand(hv_k, Σ, ω, ω+Ω)

function kinetic_coefficient_frequency_integral(hv_k, frequency_solver, n, β, Ω=0.0)
    frequency_solver((hv_k, n, β, Ω))
end

"""
    KineticCoefficientIntegrand([bz=FullBZ,] alg::AbstractAutoBZAlgorithm, hv::AbstracVelocity, Σ, n, β, [Ω=0])
    KineticCoefficientIntegrand([lb=lb(Σ), ub=ub(Σ),] alg, hv::AbstracVelocity, Σ, n, β, [Ω=0])

A function whose integral over the BZ gives the kinetic
coefficient. Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
The argument `alg` determines what the order of integration is. Given a BZ
algorithm, the inner integral is the BZ integral. Otherwise it is the frequency
integral.
"""
function KineticCoefficientIntegrand(bz, alg::AbstractAutoBZAlgorithm, hv::AbstractVelocity, Σ, n, p...; kwargs...)
    # put the frequency integral outside if the provided algorithm is for the BZ
    transport_integrand = TransportDistributionIntegrand(hv, Σ)
    transport_solver = IntegralSolver(transport_integrand, bz, alg; kwargs...)
    Integrand(transport_fermi_integrand, transport_solver, n, p...)
end
KineticCoefficientIntegrand(alg::AbstractAutoBZAlgorithm, hv::AbstractVelocity, p...; kwargs...) =
    KineticCoefficientIntegrand(FullBZ(2pi*I(ndims(hv))), alg, hv, p...; kwargs...)

function KineticCoefficientIntegrand(lb, ub, alg, hv::AbstractVelocity, Σ, n, p...; kwargs...)
    # put the frequency integral inside otherwise
    frequency_integrand = Integrand(kinetic_coefficient_integrand, Σ)
    frequency_solver = IntegralSolver(frequency_integrand, lb, ub, alg; kwargs...)
    FourierIntegrand(kinetic_coefficient_frequency_integral, hv, frequency_solver, n, p...)
end
KineticCoefficientIntegrand(alg, hv::AbstractVelocity, Σ, p...; kwargs...) =
    KineticCoefficientIntegrand(lb(Σ), ub(Σ), alg, hv, Σ, p...; kwargs...)

SymRep(::FourierIntegrand{typeof(kinetic_coefficient_frequency_integral)}) = LatticeRep()


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
    if u+Ω > ub
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from above"
        u = ub-Ω
    end
    l, u
end

# provide safe limits of integration for frequency integrals
function construct_problem(s::IntegralSolver{iip,<:Integrand{typeof(transport_fermi_integrand)}}, p) where iip
    Ω, β = if (lp = length(p)) == 1
        if (lsp = length(s.f.p)) == 2
            zero(only(p)), only(p) # p is β
        elseif lsp == 3
            only(p), s.f.p[3] # p is Ω
        end
    elseif lp == 2
        p[2], ps[1]
    end
    a, b = get_safe_fermi_window_limits(Ω, β, s.lb, s.ub)
    IntegralProblem{iip}(s.f, a, b, p; s.kwargs...)
end
function construct_problem(s::IntegralSolver{iip,<:Integrand{typeof(kinetic_coefficient_integrand)}}, p::Tuple) where iip
    Ω, β = if length(p) == 3
        zero(p[3]), p[3]
    elseif length(p) == 4
        p[4], p[3]
    end
    a, b = get_safe_fermi_window_limits(Ω, β, s.lb, s.ub)
    IntegralProblem{iip}(s.f, a, b, p; s.kwargs...)
end

OpticalConductivityIntegrand(alg, hv::AbstractVelocity, Σ, p...; kwargs...) =
    KineticCoefficientIntegrand(alg, hv, Σ, 0, p...; kwargs...)
OpticalConductivityIntegrand(bz, alg, hv::AbstractVelocity, Σ, p...; kwargs...) =
    KineticCoefficientIntegrand(bz, alg, hv, Σ, 0, p...; kwargs...)
OpticalConductivityIntegrand(lb, ub, alg, hv::AbstractVelocity, Σ, p...; kwargs...) =
    KineticCoefficientIntegrand(lb, ub, alg, hv, Σ, 0, p...; kwargs...)


electron_density_integrand(ω, (h_k, Σ, β, μ)) =
    fermi(β*ω)*dos_integrand(h_k, (ω+μ)*I-Σ(ω)) # shift only energy, not self energy

function electron_density_frequency_integral(h_k::AbstractMatrix, Σ, β, μ, lb, ub, alg=QuadGKJL())
    prob = IntegralProblem(electron_density_integrand, lb, ub, (h_k, Σ, β, μ))
    only(solve(prob, alg))
end

"""
    ElectronDensityIntegrand

A function whose integral over the BZ gives the electron density.
Mathematically, this computes
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
See `FourierIntegrand` for more details.
"""
ElectronDensityIntegrand(h::Hamiltonian, p...) =
    FourierIntegrand(electron_density_frequency_integral, h, p)

