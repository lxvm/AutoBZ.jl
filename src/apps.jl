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
    @eval $T(h::HamiltonianInterp, args...; kwargs...) = FourierIntegrand($f, h, args...; kwargs...)
    
    # pre-evaluate the self energy when constructing the integrand
    @eval function FourierIntegrand(f::typeof($f), h::HamiltonianInterp, p::MixedParameters{<:Tuple{AbstractSelfEnergy,Real}})
        Σ, ω = p.args
        FourierIntegrand(f, h, MixedParameters((ω*I-Σ(ω),), p.kwargs))
    end
    @eval function FourierIntegrand(f::typeof($f), h::HamiltonianInterp, p::MixedParameters{<:Tuple{AbstractSelfEnergy,Real,Real}})
        Σ, ω, μ = p.args
        FourierIntegrand(f, h, MixedParameters(((ω+μ)*I-Σ(ω),), p.kwargs))
    end

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

function transport_function_integrand((h, vs)::Tuple{Eigen,SVector{N,T}}; β) where {N,T}
    f′ = Diagonal(β .* fermi′.(β .* h.values))
    f′vs = map(v -> f′*v, vs)
    data = ntuple(Val(N^2)) do n
        d, r = divrem(n-1, N)
        tr_mul(vs[d+1], f′vs[r+1])
    end
    SMatrix{N,N,eltype(T),N^2}(data)
end

"""
    TransportFunctionIntegrand(hv; β)

Computes the following integral
```math
\\D_{\\alpha\\beta} = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
"""
function TransportFunctionIntegrand(hv::AbstractVelocityInterp; kwargs...)
    @assert gauge(hv) isa Hamiltonian
    # TODO change to the Hamiltonian gauge automatically
    FourierIntegrand(transport_function_integrand, hv; kwargs...)
end

const TransportFunctionIntegrandType = FourierIntegrand{typeof(transport_function_integrand)}

SymRep(::TransportFunctionIntegrandType) = LatticeRep()


spectral_function(h, M) = imag(gloc_integrand(h, M))/(-pi)

transport_distribution_integrand(hv, Σ::AbstractSelfEnergy, ω₁, ω₂) =
    transport_distribution_integrand(hv, ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂))
transport_distribution_integrand(hv, Σ::AbstractMatrix, ω₁, ω₂) =
    transport_distribution_integrand(hv, ω₁*I-Σ, ω₂*I-Σ)
function transport_distribution_integrand((h, vs), Mω₁, Mω₂)
    transport_distribution_integrand_(vs, spectral_function(h, Mω₁), spectral_function(h, Mω₂))
end
function transport_distribution_integrand_(vs::SVector{N,V}, Aω₁::A, Aω₂::A) where {N,V,A}
    T = Base.promote_op((v, a) -> tr_mul(v*a,v*a), V, A)
    vsAω₁ = map(v -> v * Aω₁, vs)
    vsAω₂ = map(v -> v * Aω₂, vs)
    data = ntuple(Val(N^2)) do n
        d, r = divrem(n-1, N)
        tr_mul(vsAω₁[d+1], vsAω₂[r+1])
    end
    SMatrix{N,N,T,N^2}(data)
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
TransportDistributionIntegrand(hv::AbstractVelocityInterp, p...) =
    FourierIntegrand(transport_distribution_integrand, hv, p...)
    
# pre-evaluate self energies when constructing integrand
function FourierIntegrand(f::typeof(transport_distribution_integrand), hv::AbstractVelocityInterp, p::MixedParameters{<:Tuple{AbstractSelfEnergy,Real,Real}})
    Σ, ω₁, ω₂ = p.args
    FourierIntegrand(f, hv, MixedParameters((ω₁*I-Σ(ω₁), ω₂*I-Σ(ω₂)), p.kwargs))
end

const TransportDistributionIntegrandType = FourierIntegrand{typeof(transport_distribution_integrand)}

SymRep(::TransportDistributionIntegrandType) = LatticeRep()

function npt_update(Γ::TransportDistributionIntegrandType, npt::Integer)
    ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
    ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
    eta_npt_update(npt, min(ηω₁, ηω₂), period(Γ.s)[1])
end



transport_fermi_integrand(ω, Γ, n, β, Ω=0.0) =
    (ω*β)^n * β * fermi_window(β*ω, β*Ω) * Γ(ω, ω+Ω)

kinetic_coefficient_integrand(ω, Σ, hv_k, n, β, Ω) =
    (ω*β)^n * β * fermi_window(β*ω, β*Ω) * transport_distribution_integrand(hv_k, Σ, ω, ω+Ω)

function kinetic_coefficient_frequency_integral(hv_k, frequency_solver, n, β, Ω=0.0)
    frequency_solver(hv_k, n, β, Ω)
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
function KineticCoefficientIntegrand(bz, alg::AbstractAutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, n, p...; kwargs...)
    # put the frequency integral outside if the provided algorithm is for the BZ
    transport_integrand = TransportDistributionIntegrand(hv, Σ)
    transport_solver = IntegralSolver(transport_integrand, bz, alg; kwargs...)
    transport_solver(0.0, 10.0) # precompile the solver
    Integrand(transport_fermi_integrand, transport_solver, n, p...)
end
KineticCoefficientIntegrand(alg::AbstractAutoBZAlgorithm, hv::AbstractVelocityInterp, p...; kwargs...) =
    KineticCoefficientIntegrand(FullBZ(2pi*I(ndims(hv))), alg, hv, p...; kwargs...)

function KineticCoefficientIntegrand(lb, ub, alg, hv::AbstractVelocityInterp, Σ, n, p...; kwargs...)
    # put the frequency integral inside otherwise
    frequency_integrand = Integrand(kinetic_coefficient_integrand, Σ)
    frequency_solver = IntegralSolver(frequency_integrand, lb, ub, alg; do_inf_transformation=Val(false), kwargs...)
    frequency_solver(hv(fill(0.0, ndims(hv))), 0, 1.0, 10.0) # precompile the solver
    FourierIntegrand(kinetic_coefficient_frequency_integral, hv, frequency_solver, n, p...)
end
KineticCoefficientIntegrand(alg, hv::AbstractVelocityInterp, Σ, p...; kwargs...) =
    KineticCoefficientIntegrand(lb(Σ), ub(Σ), alg, hv, Σ, p...; kwargs...)

const KineticCoefficientIntegrandType = FourierIntegrand{typeof(kinetic_coefficient_frequency_integral)}

SymRep(::KineticCoefficientIntegrandType) = LatticeRep()


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
function construct_problem(s::IntegralSolver{iip,<:Integrand{typeof(transport_fermi_integrand)}}, p::MixedParameters) where iip
    # inverse temperature is the third parameter
    β = if (lsp = length(s.f.p.args)) >= 3
        s.f.p[3]
    else
        p[3-lsp]
    end
    # excitation frequency is the fourth parameter
    Ω = if lsp >= 4
        s.f.p[4]
    else
        p[4-lsp]
    end
    a, b = get_safe_fermi_window_limits(Ω, β, s.lb, s.ub)
    IntegralProblem{iip}(s.f, a, b, p; s.kwargs...)
end
function construct_problem(s::IntegralSolver{iip,<:Integrand{typeof(kinetic_coefficient_integrand)}}, p::MixedParameters) where iip
    Ω, β = if length(p.args) == 3
        zero(p[3]), p[3]
    elseif length(p.args) >= 4
        p[4], p[3]
    end
    a, b = get_safe_fermi_window_limits(Ω, β, s.lb, s.ub)
    IntegralProblem{iip}(s.f, a, b, p; s.kwargs...)
end

OpticalConductivityIntegrand(alg, hv::AbstractVelocityInterp, Σ, p...; kwargs...) =
    KineticCoefficientIntegrand(alg, hv, Σ, 0, p...; kwargs...)
OpticalConductivityIntegrand(bz, alg, hv::AbstractVelocityInterp, Σ, p...; kwargs...) =
    KineticCoefficientIntegrand(bz, alg, hv, Σ, 0, p...; kwargs...)
OpticalConductivityIntegrand(lb, ub, alg, hv::AbstractVelocityInterp, Σ, p...; kwargs...) =
    KineticCoefficientIntegrand(lb, ub, alg, hv, Σ, 0, p...; kwargs...)


dos_fermi_integrand(ω, dos, β, μ=0.0) =
    fermi(β*ω)*dos(ω, μ)

electron_density_integrand(ω, Σ, h_k, β, μ) =
    fermi(β*ω)*dos_integrand(h_k, (ω+μ)*I-Σ(ω)) # shift only energy, not self energy

electron_density_frequency_integral(h_k::AbstractMatrix, frequency_solver, β, μ=0.0) =
    frequency_solver(h_k, β, μ)

"""
    ElectronDensityIntegrand([bz=FullBZ], alg::AbstractAutoBZAlgorithm, h::HamiltonianInterp, Σ, β, [μ=0])
    ElectronDensityIntegrand([lb=lb(Σ), ub=ub(Σ),] alg, h::HamiltonianInterp, Σ, β, [μ=0])

A function whose integral over the BZ gives the electron density.
Mathematically, this computes
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
The argument `alg` determines what the order of integration is. Given a BZ
algorithm, the inner integral is the BZ integral. Otherwise it is the frequency
integral.
"""
function ElectronDensityIntegrand(bz, alg::AbstractAutoBZAlgorithm, h::HamiltonianInterp, Σ, p...; kwargs...)
    dos_int = DOSIntegrand(h, Σ)
    dos_solver = IntegralSolver(dos_int, bz, alg; kwargs...)
    dos_solver(0.0, 0.0) # precompile the solver
    Integrand(dos_fermi_integrand, dos_solver, p...)
end
ElectronDensityIntegrand(alg::AbstractAutoBZAlgorithm, h::HamiltonianInterp, p...; kwargs...) =
    ElectronDensityIntegrand(FullBZ(2pi*I(ndims(h))), alg, h, p...; kwargs...)

function ElectronDensityIntegrand(lb, ub, alg, h::HamiltonianInterp, Σ, p...; kwargs...)
    frequency_integrand = Integrand(electron_density_integrand, Σ)
    frequency_solver = IntegralSolver(frequency_integrand, lb, ub, alg; kwargs...)
    frequency_solver(h(fill(0.0, ndims(h))), 1.0, 0.0) # precompile the solver
    FourierIntegrand(electron_density_frequency_integral, h, frequency_solver, p...)
end
ElectronDensityIntegrand(alg, h::HamiltonianInterp, Σ, p...; kwargs...) =
    ElectronDensityIntegrand(lb(Σ), ub(Σ), alg, h, Σ, p...; kwargs...)

