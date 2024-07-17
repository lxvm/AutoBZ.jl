"""
    ElectronDensitySolver(h, bz, bzalg, Σ, fdom, falg; β, μ=0, kws...)
    ElectronDensitySolver(Σ, fdom, falg, h, bz, bzalg; β, μ=0, kws...)

A solver for the electron density.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
Use `AutoBZ.update_density!(solver; β, μ=0)`

A function whose integral over the BZ gives the electron density.
Mathematically, this computes
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
To get the density/number of electrons, multiply the result of this integral by `n_sp/det(bz.B)`
"""
function ElectronDensitySolver(Σ::AbstractSelfEnergy, fdom, falg, h::AbstractHamiltonianInterp, bz, bzalg; β, μ=zero(inv(oneunit(β))), bandwidth=one(μ), kws...)
    # TODO better estimate the bandwidth
    V = abs(det(bz.B))
    inner_kws = _rescale_abstol(inv(bandwidth); kws...)
    dos_solver = DOSSolver(Σ, h, bz, bzalg; ω=(fdom[1]+fdom[2])/2, μ, inner_kws...)
    p = (; β, μ)
    proto = dos_solver.f.prototype * V * fermi(β, (fdom[1]+fdom[2])/2)
    f = IntegralFunction(proto) do ω, (; β, μ)
        update_gloc!(dos_solver; ω, μ)
        dos = solve!(dos_solver).value
        return dos*fermi(β, ω)
    end
    prob = IntegralProblem(f, get_safe_fermi_function_limits(β, lb(Σ), ub(Σ)), p; kws...)
    return init(prob, falg)
end

function _rescale_abstol(s; kws...)
    haskey(NamedTuple(kws), :abstol) || return (; kws...)
    return (; kws..., abstol=NamedTuple(kws).abstol*s)
end

function update_density!(solver::AutoBZCore.IntegralSolver; β, μ=zero(inv(oneunit(β))))
    solver.p = (; β, μ)
    solver.dom = get_safe_fermi_function_limits(β, solver.dom...)
    # TODO rescale inner tolerance
    return
end

function ElectronDensitySolver(h::AbstractHamiltonianInterp, bz, bzalg, Σ::AbstractSelfEnergy, fdom, falg; β, μ=zero(inv(oneunit(β))), bandwidth=one(μ), kws...)
    V = abs(det(bz.B))
    k = period(h)
    hk = h(k)
    proto = dos_integrand(propagator_denominator(hk, _evalM(Σ(zero(μ)), zero(μ), μ)))*fermi(β, (fdom[1]+fdom[2])/2)
    f = IntegralFunction(proto) do ω, (Σ, h, (; β, μ))
        return fermi(β, ω)*dos_integrand(propagator_denominator(h, _evalM(Σ(ω), ω, μ)))
    end
    inner_kws = _rescale_abstol(inv(V*nsyms(bz)); kws...)
    fprob = IntegralProblem(f, fdom, (Σ, hk, (; β, μ)); inner_kws...)
    up = (solver, k, h, p) -> begin
        solver.p = (solver.p[1], h, p[2])
        # updating the domain in update! uses internals but also emits fewer warnings than this
        # solver.dom = get_safe_fermi_function_limits(β, fdom...)
        return
    end
    post = (sol, k, h, p) -> sol.value
    f = CommonSolveFourierIntegralFunction(fprob, falg, up, post, h, proto*μ)
    prob = AutoBZProblem(TrivialRep(), f, bz, (fdom, (; β, μ)); kws...)
    return init(prob, bzalg)
end

function update_density!(solver::AutoBZCore.AutoBZCache; β, μ=zero(inv(oneunit(β))))
    fdom = solver.p[1]
    solver.p = (fdom, (; β, μ))
    solver.cacheval.dom = get_safe_fermi_function_limits(β, fdom...)
    return
end

function get_safe_fermi_function_limits(β, lb, ub; kwargs...)
    l, u = fermi_function_limits(β; kwargs...)
    if l < lb
        @warn "At β=$β, the interpolant limits the desired frequency window from below"
        l = oftype(l, lb)
    end
    if u > ub
        @warn "At β=$β, the interpolant limits the desired frequency window from above"
        u = oftype(u, ub)
    end
    l, u
end
