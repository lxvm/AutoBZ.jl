"""
    AuxKineticCoefficientSolver([auxfun], hv, bz, bzalg, Σ, falg; n, β, Ω, μ=0, kws...)
    AuxKineticCoefficientSolver([auxfun], Σ, falg, hv, bz, bzalg; n, β, Ω, μ=0, kws...)

A solver for kinetic coefficients using an auxiliary integrand.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
The default `auxfun` is the sum of the Green's functions.
Use `AutoBZ.update_auxkc!(solver; β, Ω, μ, n)` to change parameters.
"""
function AuxKineticCoefficientSolver(auxfun, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, falg; β, Ω, n, μ=zero(inv(oneunit(β))), bandwidth=one(μ), kws...)
    k = SVector(period(hv))
    hvk = hv(k)
    proto = (zero(Ω)*β)^n * fermi_window(β, zero(Ω), Ω) * aux_transport_distribution_integrand(k, hvk, (auxfun, Σ, evalM2(; Σ, ω₁=zero(Ω), ω₂=Ω, μ)))
    f = IntegralFunction(proto) do ω, (Σ, hv, (; β, μ, Ω, n))
        Γ = aux_transport_distribution_integrand(k, hv, (auxfun, Σ, evalM2(; Σ, ω₁=ω, ω₂=ω+Ω, μ)))
        return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
    end
    V = abs(det(bz.B))
    inner_kws = _rescale_abstol(inv(V*nsyms(bz)); kws...)
    fprob = IntegralProblem(f, get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ)), (Σ, hvk, (; β, μ, Ω, n)); inner_kws...)
    up = (solver, k, hv, p) -> begin
    #     if iszero(Ω) && isinf(β)
    #         # we pass in β=4 since fermi_window(4,0,0)=1, the weight of the delta
    #         # function, and also this prevents (0*β)^n from giving NaN when n!=0
    #         return Ω * f.f(Ω, MixedParameters(; Σ, n, β=4*oneunit(β), Ω, μ, hv_k))
    #     end
        Σ = solver.p[1]
        solver.p = (Σ, hv, p)
        solver.dom = get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ))
        return
    end
    post = (sol, k, h, p) -> sol.value
    f = CommonSolveFourierIntegralFunction(fprob, falg, up, post, hv, proto*Ω)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, (; β, μ, Ω, n); kws...)
    return init(prob, bzalg)
end
function AuxKineticCoefficientSolver(auxfun, Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm; β, Ω, n, μ=zero(Ω), kws...)
    inner_kws = _rescale_abstol(inv(max(Ω, inv(β))); kws...)
    auxtd_solver = AuxTransportDistributionSolver(auxfun, Σ, hv, bz, bzalg; ω₁=zero(Ω), ω₂=Ω, μ, inner_kws...)
    p = (; β, μ, Ω, n)
    proto = (zero(Ω)*β)^n * fermi_window(β, zero(Ω), Ω) * auxtd_solver.f.prototype
    f = IntegralFunction(proto) do ω, (; β, μ, Ω, n)
        update_auxtd!(auxtd_solver; ω₁=ω, ω₂=ω+Ω, μ)
        Γ = solve!(auxtd_solver).value
        return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
    end
    prob = IntegralProblem(f, get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ)), p; kws...)
    return init(prob, falg)
end

AuxKineticCoefficientSolver(hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, falg; kws...) = AuxKineticCoefficientSolver(_trG_auxfun, hv, bz, bzalg, Σ, falg; _trG_kws(; kws...)...)
AuxKineticCoefficientSolver(Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm; kws...) = AuxKineticCoefficientSolver(_trG_auxfun, Σ, falg, hv, bz, bzalg; _trG_kws(; kws...)...)
update_auxkc!(args...; kws...) = update_kc!(args...; kws...)
function update_auxkc!(solver::AutoBZCore.IntegralSolver; β, Ω, n, μ=zero(Ω))
    Σ = solver.f.f.auxtd_solver.p[2]
    solver.p = (; β, μ, Ω, n)
    solver.dom = get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ))
    return
end

"""
    AuxOpticalConductivitySolver([auxfun], hv, bz, bzalg, Σ, falg; β, Ω, μ=0, kws...)
    AuxOpticalConductivitySolver([auxfun], Σ, falg, hv, bz, bzalg; β, Ω, μ=0, kws...)

A solver for the optical conductivity. For details see [`AuxKineticCoefficientSolver`](@ref)
and note that by default the parameter `n=0`. Use `AutoBZ.update_auxoc!(solver; β, Ω, μ)` to
change parameters.
"""
AuxOpticalConductivitySolver(args...; kws...) = AuxKineticCoefficientSolver(args...; kws..., n=0)
update_auxoc!(solver; kws...) = update_auxkc!(solver; kws..., n=0)
