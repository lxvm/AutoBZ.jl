function aux_transport_distribution_integrand(k, v, (auxfun, Σ, (Mω₁, Mω₂, isdistinct)))
    h, vs = v
    if isdistinct
        Gω₁ = gloc_integrand(propagator_denominator(h, Mω₁))
        Gω₂ = gloc_integrand(propagator_denominator(h, Mω₂))
        Aω₁ = spectral_function(Gω₁)
        Aω₂ = spectral_function(Gω₂)
        return AutoBZCore.IteratedIntegration.AuxValue(transport_distribution_integrand_(vs, Aω₁, Aω₂), auxfun(vs, Gω₁, Gω₂))
    else
        Gω = gloc_integrand(propagator_denominator(h, Mω₁))
        Aω = spectral_function(Gω)
        return AutoBZCore.IteratedIntegration.AuxValue(transport_distribution_integrand_(vs, Aω), auxfun(vs, Gω, Gω))
    end
end


"""
    AuxTransportDistributionSolver([auxfun], Σ, hv::AbstractVelocityInterp, bz, bzalg; ω₁, ω₂, μ=0, kws...)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_auxtd!(solver; ω₁, ω₂, μ)` to update the parameters.
"""
function AuxTransportDistributionSolver(auxfun, Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, bzalg; ω₁, ω₂, μ=zero(ω₁), kwargs...)
    p = (auxfun, Σ, evalM2(; Σ, ω₁, ω₂, μ))
    k = SVector(period(hv))
    hvk = hv(k)
    proto = aux_transport_distribution_integrand(k, hvk, p)
    f = FourierIntegralFunction(aux_transport_distribution_integrand, hv, proto)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kwargs...)
    return init(prob, bzalg)
end
AuxTransportDistributionSolver(Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, bzalg; kws...) = AuxTransportDistributionSolver(_trG_auxfun, Σ, hv, bz, bzalg; _trG_kws(; kws...)...)
_trG_auxfun(vs, Gω₁, Gω₂) = tr(Gω₁) + tr(Gω₂)
function _trG_kws(; kws...)
    (!haskey(kws, :abstol) || !(kws[:abstol] isa AutoBZCore.IteratedIntegration.AuxValue)) && @warn "pick a sensible default auxiliary tolerance"
    return (; kws...)
end

function update_auxtd!(solver; ω₁, ω₂, μ=zero(ω₁))
    Σ = solver.p[2]
    solver.p = (solver.p[1], Σ, evalM2(; Σ, ω₁, ω₂, μ))
    return
end
