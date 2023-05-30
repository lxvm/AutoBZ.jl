module AuxIntegrandsExt
using AuxQuad
using AutoBZ


struct AuxiliaryValue{T}
    val::T
    aux::T
end

LinearAlgebra.norm(a::AuxiliaryValue) = AuxiliaryValue(norm(a.val), norm(a.aux))
Base.size(a::AuxiliaryValue) = size(a.val)
Base.eltype(::Type{AuxiliaryValue{T}}) where T = T
Base.:*(a::AuxiliaryValue, b::AuxiliaryValue) = AuxiliaryValue(a.val*b.val, a.aux*b.aux)
Base.:*(a::AuxiliaryValue, b) = AuxiliaryValue(a.val*b, a.aux*b)
Base.:*(a, b::AuxiliaryValue) = AuxiliaryValue(a*b.val, a*b.aux)
Base.:+(a::AuxiliaryValue, b::AuxiliaryValue) = AuxiliaryValue(a.val+b.val, a.aux+b.aux)
Base.:-(a::AuxiliaryValue, b::AuxiliaryValue) = AuxiliaryValue(a.val-b.val, a.aux-b.aux)
Base.:/(a::AuxiliaryValue, b) = AuxiliaryValue(a.val/b, a.aux/b)
Base.zero(a::AuxiliaryValue) = AuxiliaryValue(zero(a.val), zero(a.aux))
Base.isinf(a::AuxiliaryValue) = isinf(a.val) || isinf(a.aux)
Base.isnan(a::AuxiliaryValue) = isnan(a.val) || isnan(a.aux)

# sort the heap by value
# Base.isless(a::AuxiliaryValue, b::AuxiliaryValue) = isless(a.val, b.val)
# sort the heap by auxiliary value
Base.isless(a::AuxiliaryValue, b::AuxiliaryValue) = isless(a.aux, b.aux)
# Base.isless(a::AuxiliaryValue, b) = isless(a.aux, b)
# Base.isless(a, b::AuxiliaryValue) = isless(a, b.aux)

# strict error comparisons (De Morgan's Laws)
Base.:>(a::AuxiliaryValue, b::AuxiliaryValue) = >(a.val, b) || >(a.aux, b)
Base.:>(a::AuxiliaryValue, b) = >(a.val, b) || >(a.aux, b)
Base.:>(a, b::AuxiliaryValue) = >(a, b.val) || >(a, b.aux)

Base.:<(a::AuxiliaryValue, b::AuxiliaryValue) = <(a.val, b) && <(a.aux, b)
Base.:<(a::AuxiliaryValue, b) = <(a.val, b) && <(a.aux, b)
Base.:<(a, b::AuxiliaryValue) = <(a, b.val) && <(a, b.aux)

Base.isequal(a::AuxiliaryValue, b::AuxiliaryValue) = isequal(a.val, b.val) && isequal(a.aux, b.aux)
Base.isequal(a::AuxiliaryValue, b) = isequal(a.val, b) && isequal(a.aux, b)
Base.isequal(a, b::AuxiliaryValue) = isequal(a, b.val) && isequal(a, b.aux)
Base.max(a::AuxiliaryValue, b::AuxiliaryValue) = AuxiliaryValue(max(a.val, b.val), max(a.aux, b.aux))
Base.max(a::AuxiliaryValue, b) = AuxiliaryValue(max(a.val, b), max(a, b.aux))
Base.max(a, b::AuxiliaryValue) = AuxiliaryValue(max(a, b.val), max(a, b.aux))


function aux_transport_distribution_integrand_(vs::SVector{N,V}, Gω₁::G, Gω₂::G) where {N,V,G}
    vsGω₁ = map(v -> v * Gω₁, vs)
    vsGω₂ = map(v -> v * Gω₂, vs)
    Aω₁ = spectral_function(Gω₁)
    Aω₂ = spectral_function(Gω₂)
    vsAω₁ = map(v -> v * Aω₁, vs)
    vsAω₂ = map(v -> v * Aω₂, vs)
    return AuxiliaryValue(tr_kron(vsAω₁, vsAω₂), tr_kron(vsGω₁, vsGω₂))
end

aux_transport_distribution_integrand((h, vs), Mω₁, Mω₂) =
    aux_transport_distribution_integrand_(vs, gloc_integrand(h, Mω₁), gloc_integrand(h, Mω₂))

"""
    TransportDistributionIntegrand(hv, Σ, ω₁, ω₂, μ)
    TransportDistributionIntegrand(hv, Σ, ω₁, ω₂; μ)
    TransportDistributionIntegrand(hv, Σ; ω₁, ω₂, μ)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See `FourierIntegrand` for more details.
"""
AuxTransportDistributionIntegrand(hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    FourierIntegrand(aux_transport_distribution_integrand, hv, Σ, args...; kwargs...)

function FourierIntegrand(f::typeof(aux_transport_distribution_integrand), hv::AbstractVelocityInterp, p::EvalM2Type)
    FourierIntegrand(f, hv, evalM2(p))
end

const AuxTransportDistributionIntegrandType = FourierIntegrand{typeof(aux_transport_distribution_integrand)}

SymRep(Γ::AuxTransportDistributionIntegrandType) = coord_to_rep(Γ.s)

function npt_update(Γ::AuxTransportDistributionIntegrandType, npt::Integer)
    ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
    ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
    eta_npt_update(npt, min(ηω₁, ηω₂), maximum(period(Γ.s)))
end



aux_kinetic_coefficient_integrand(ω, Σ, hv_k, n::Real, β::Real, Ω::Real, μ::Real) =
    (ω*β)^n * fermi_window(β, ω, Ω) * aux_transport_distribution_integrand(hv_k, evalM2(Σ, ω, ω+Ω, μ)...)

function AuxKineticCoefficientIntegrand(bz, alg::AbstractAutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral outside if the provided algorithm is for the BZ
    transport_integrand = AuxTransportDistributionIntegrand(hv, Σ)
    transport_solver = IntegralSolver(transport_integrand, bz, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    transport_solver(max(-10.0, lb(Σ)), min(10.0, ub(Σ)), 0) # precompile the solver
    Integrand(transport_fermi_integrand, transport_solver, args...; kwargs...)
end
AuxKineticCoefficientIntegrand(alg::AbstractAutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(FullBZ(2pi*I(ndims(hv))), alg, hv, Σ, args...; kwargs...)

function AuxKineticCoefficientIntegrand(lb_, ub_, alg, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral inside otherwise
    frequency_integrand = Integrand(aux_kinetic_coefficient_integrand, Σ)
    frequency_solver = IntegralSolver(frequency_integrand, lb_, ub_, alg; do_inf_transformation=Val(false), abstol=abstol, reltol=reltol, maxiters=maxiters)
    frequency_solver(hv(fill(0.0, ndims(hv))), 0, 1e100, 0.0, 0) # precompile the solver
    FourierIntegrand(kinetic_coefficient_frequency_integral, hv, frequency_solver, args...; kwargs...)
end
AuxKineticCoefficientIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(lb(Σ), ub(Σ), alg, hv, Σ, args...; kwargs...)

"""
    AuxOpticalConductivityIntegrand

Returns a `AuxKineticCoefficientIntegrand` with `n=0`. See
[`AuxKineticCoefficientIntegrand`](@ref) for further details
"""
AuxOpticalConductivityIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(alg, hv, Σ, 0, args...; kwargs...)
AuxOpticalConductivityIntegrand(bz, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(bz, alg, hv, Σ, 0, args...; kwargs...)
AuxOpticalConductivityIntegrand(lb, ub, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(lb, ub, alg, hv, Σ, 0, args...; kwargs...)

function canonize_kc_params(solver_::IntegralSolver{iip,<:Integrand{typeof(aux_kinetic_coefficient_integrand)}}, n_, β_, Ω_, μ_; n=n_, β=β_, Ω=Ω_, μ=μ_) where iip
    iszero(Ω) && isinf(β) && throw(ArgumentError("Ω=0, T=0 not yet implemented. As a workaround, evaluate the KCIntegrand at ω=0"))
    a, b = get_safe_fermi_window_limits(Ω, β, solver_.lb, solver_.ub)
    solver = IntegralSolver(solver_.f, a, b, solver_.alg, sensealg = solver_.sensealg,
            do_inf_transformation = solver_.do_inf_transformation, kwargs = solver_.kwargs,
            abstol = solver_.abstol, reltol = solver_.reltol, maxiters = solver_.maxiters)
    (solver, n, β, Ω, μ)
end


end