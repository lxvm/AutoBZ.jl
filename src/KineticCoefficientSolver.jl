"""
    KineticCoefficientSolver(hv, bz, bzalg, Σ, falg; n, β, Ω, μ=0, kws...)
    KineticCoefficientSolver(Σ, falg, hv, bz, bzalg; n, β, Ω, μ=0, kws...)

A solver for kinetic coefficients.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
Use `AutoBZ.update_kc!(solver; β, Ω, μ, n)` to change parameters.

Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
"""
function KineticCoefficientSolver(Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm; β, Ω, n, μ=zero(Ω), kws...)
    inner_kws = _rescale_abstol(inv(max(Ω, inv(β))); kws...)
    td_solver = TransportDistributionSolver(Σ, hv, bz, bzalg; ω₁=zero(Ω), ω₂=Ω, μ, inner_kws...)
    p = (; β, μ, Ω, n)
    proto = (zero(Ω)*β)^n * fermi_window(β, zero(Ω), Ω) * td_solver.f.prototype
    f = IntegralFunction(proto) do ω, (; β, μ, Ω, n)
        update_td!(td_solver; ω₁=ω, ω₂=ω+Ω, μ)
        Γ = solve!(td_solver).value
        return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
    end
    prob = IntegralProblem(f, get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ)), p; kws...)
    return init(prob, falg)
end

function update_kc!(solver::AutoBZCore.IntegralSolver; β, Ω, n, μ=zero(Ω))
    Σ = solver.f.f.td_solver.p[1]
    solver.p = (; β, μ, Ω, n)
    solver.dom = get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ))
    return
end

function KineticCoefficientSolver(hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, falg; β, Ω, n, μ=zero(inv(oneunit(β))), bandwidth=one(μ), kws...)
    k = SVector(period(hv))
    hvk = hv(k)
    proto = (zero(Ω)*β)^n * fermi_window(β, zero(Ω), Ω) * transport_distribution_integrand(k, hvk, (Σ, evalM2(; Σ, ω₁=zero(Ω), ω₂=Ω, μ)))
    f = IntegralFunction(proto) do ω, (Σ, hv, (; β, μ, Ω, n))
        Γ = transport_distribution_integrand(k, hv, (Σ, evalM2(; Σ, ω₁=ω, ω₂=ω+Ω, μ)))
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

function update_kc!(solver::AutoBZCore.AutoBZCache; β, Ω, n, μ=zero(Ω))
    solver.p = (; β, μ, Ω, n)
    return
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
function get_safe_fermi_window_limits(Ω, β, lb, ub; kwargs...)
    l, u = fermi_window_limits(Ω, β; kwargs...)
    if l < lb
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from below"
        l = oftype(l, lb)
    end
    if u+Ω > ub
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from above"
        u = oftype(u, ub-Ω)
    end
    l, u
end

"""
    OpticalConductivitySolver(hv, bz, bzalg, Σ, falg; β, Ω, μ=0, kws...)
    OpticalConductivitySolver(Σ, falg, hv, bz, bzalg; β, Ω, μ=0, kws...)

A solver for the optical conductivity. For details see [`KineticCoefficientSolver`](@ref)
and note that by default the parameter `n=0`. Use `AutoBZ.update_oc!(solver; β, Ω, μ)` to
change parameters.
"""
OpticalConductivitySolver(args...; kws...) = KineticCoefficientSolver(args...; kws..., n=0)
update_oc!(solver; kws...) = update_kc!(solver; kws..., n=0)
