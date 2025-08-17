function _DynamicalTransportDistributionSolver(fun::F, Σ::AbstractSelfEnergy, fdom, falg, hv::AbstractVelocityInterp, bz, bzalg, linalg; β, Ω, n, μ=zero(Ω), scale_inner=nothing, inner_kws=nothing, kws...) where {F}
    dom = get_safe_fermi_window_limits(Ω, β, fdom...)
    s = 1 # inv((dom[2]-dom[1])*fermi_window_maximum(β, Ω)) # or area under the window function
    # the right choice depends on whether worst-case point-wise error or average is
    # important close to 1) important and could eventually let the user decide
    # And to be rigorous, the scaling should be 1/(dom[2]-dom[1])/fermi_window(β, ω, Ω),
    # although in the tails of the window this uniform error could be dangerous
    _inner_kws = inner_kws === nothing ? _rescale_abstol(something(scale_inner, s); kws...) : inner_kws
    p = (; β, μ, Ω, n)
    _solve! = (solver, ω, (_, (; β, μ, Ω, n))) -> begin
        update_td!(solver; ω₁=ω, ω₂=ω+Ω, μ)
        # TODO rescale inner tolerance based on discussion above
        # if β != p.β || Ω != p.Ω # not ideal, should remember params in solver
        # if haskey(solver.kwargs, :abstol)
        #     _dom = get_safe_fermi_window_limits(Ω, β, fdom...) # not ideal - don't want to
        #     emit warning
        #     solver.kwargs = _rescale_abstol(inv((_dom[2]-_dom[1])*fermi_window_maximum(β, Ω)); kws...)
        # end
        sol = solve!(solver)
        return AutoBZCore.CommonSolutionStats((ω*β)^n * fermi_window(β, ω, Ω) * sol.value, sol.stats)
    end
    td_prob = _TransportDistributionProblem(fun, Σ, hv, bz, linalg; ω₁=zero(Ω), ω₂=Ω, μ, _inner_kws...)
    proto = (float(zero(Ω))*β)^n * fermi_window(β, float(zero(Ω)), Ω) * td_prob.f.prototype * det(bz.B)
    f = CommonSolveIntegralFunction(_solve!, td_prob, _heuristic_bzalg(bzalg, Σ, hv), proto)
    prob = IntegralProblem(f, dom, (fdom, p); kws...)
    return init(prob, falg)
end

function update_kc!(solver::AutoBZCore.IntegralSolver; β, Ω, n, μ=zero(Ω))
    fdom = solver.p[1]
    if solver.p[2].Ω != Ω || solver.p[2].β != β
        solver.dom = get_safe_fermi_window_limits(Ω, β, fdom...)
    end
    solver.p = (fdom, (; β, μ, Ω, n))
    return
end

function _DynamicalTransportDistributionSolver(fun::F, hv::AbstractVelocityInterp, bz, bzalg, Σ::AbstractSelfEnergy, fdom, falg, linalg; β, Ω, n, μ=zero(inv(oneunit(β))), scale_inner=nothing, inner_kws=nothing, kws...) where {F}
    M = evalM2(; Σ, ω₁=float(zero(Ω)), ω₂=float(Ω), μ)
    k = SVector(period(hv))
    hvk = hv(k)
    g = gauge(hv)
    A = g isa Hamiltonian ? Diagonal(hvk[1].values) : hvk[1]
    prob_k = TwoGreensFunctionProblem(A, _to_gauge_twice(g, hvk[1], M...)...)
    alg = TwoGreensFunctionLinearSystem(linalg)
    p = (; β, μ, Ω, n)
    p_k = (fdom, deepcopy(Σ), hvk, p)
    _ksolve! = (solver, ω, (_, Σ, hvk, (; β, μ, Ω, n))) -> begin
        solver.M1, solver.M2, solver.isdistinct = _to_gauge_twice(g, hvk[1], evalM2(; Σ, ω₁=ω, ω₂=ω+Ω, μ)...) # WARN: Σ evaluation may not be threadsafe so need another prob type
        solver.h = g isa Hamiltonian ? Diagonal(hvk[1].values) : hvk[1]
        sol = solve!(solver)
        return (ω*β)^n * fermi_window(β, ω, Ω) * fun(transport_distribution_integrand(hvk[2], sol.G1, sol.G2, sol.isdistinct), hvk..., sol)
    end
    proto = _ksolve!(init(prob_k, alg), zero(fdom[1]+fdom[2])/2, p_k)
    f_k = CommonSolveIntegralFunction(_ksolve!, prob_k, alg, proto)
    V = abs(det(bz.B))
    _inner_kws = inner_kws === nothing ? _rescale_abstol(something(scale_inner, inv(V*nsyms(bz))); kws...) : inner_kws
    fprob = IntegralProblem(f_k, get_safe_fermi_window_limits(Ω, β, fdom...), p_k; _inner_kws...)
    _solve! = (solver, k, hv, p) -> begin
        # if iszero(Ω) && isinf(β)
        #     # we pass in β=4 since fermi_window(4,0,0)=1, the weight of the delta
        #     # function, and also this prevents (0*β)^n from giving NaN when n!=0
        #     return Ω * f.f(Ω, MixedParameters(; Σ, n, β=4*oneunit(β), Ω, μ, hv_k))
        # end
        _fdom, _Σ, = solver.p
        if solver.p[4].Ω != p.Ω || solver.p[4].β != p.β
            solver.dom = get_safe_fermi_window_limits(p.Ω, p.β, _fdom...)
        end
        solver.p = (_fdom, _Σ, hv, p)
        sol = solve!(solver)
        return AutoBZCore.CommonSolutionStats(sol.value, sol.stats)
    end
    f = CommonSolveFourierIntegralFunction(_solve!, fprob, falg, hv, proto*Ω)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kws...)
    return init(prob, _heuristic_bzalg(bzalg, Σ, hv))
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
    KineticCoefficientSolver(hv, bz, bzalg, Σ, [fdom,] falg, [linalg=JLInv()]; n, β, Ω, μ=0, scale_inner=inv(abs(det(bz.B))*nsyms(bz)), kws...)
    KineticCoefficientSolver(Σ, [fdom,] falg, hv, bz, bzalg, [linalg=JLInv()]; n, β, Ω, μ=0, scale_inner=1, kws...)

A solver for kinetic coefficients.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
Use `AutoBZ.update_kc!(solver; β, Ω, μ, n)` to change parameters.
`linalg` selects the algorithm to compute the resolvent.

Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
"""
function KineticCoefficientSolver(Σ::AbstractSelfEnergy, fdom, falg::IntegralAlgorithm, hv::AbstractVelocityInterp, bz, bzalg, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    _DynamicalTransportDistributionSolver((Γ,_...) -> Γ, Σ, fdom, falg, hv, bz, bzalg, linalg; kws...)
end
function KineticCoefficientSolver(Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    KineticCoefficientSolver(Σ, (lb(Σ), ub(Σ)), falg, hv, bz, bzalg, linalg; kws...)
end

function KineticCoefficientSolver(hv::AbstractVelocityInterp, bz, bzalg, Σ::AbstractSelfEnergy, fdom, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    _DynamicalTransportDistributionSolver((Γ,_...) -> Γ, hv, bz, bzalg, Σ, fdom, falg, linalg; kws...)
end
function KineticCoefficientSolver(hv::AbstractVelocityInterp, bz, bzalg, Σ::AbstractSelfEnergy, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    KineticCoefficientSolver(hv, bz, bzalg, Σ, (lb(Σ), ub(Σ)), falg, linalg; kws...)
end

"""
    OpticalConductivitySolver(hv, bz, bzalg, Σ, [fdom,] falg, [linalg=JLInv()]; β, Ω, μ=0, scale_inner=inv(abs(det(bz.B))*nsyms(bz)), kws...)
    OpticalConductivitySolver(Σ, [fdom,] falg, hv, bz, bzalg, [linalg=JLInv()]; β, Ω, μ=0, scale_inner=1, kws...)

A solver for the optical conductivity. For details see [`KineticCoefficientSolver`](@ref)
and note that by default the parameter `n=0`. Use `AutoBZ.update_oc!(solver; β, Ω, μ)` to
change parameters.
"""
OpticalConductivitySolver(args...; kws...) = KineticCoefficientSolver(args...; kws..., n=0)
update_oc!(solver; kws...) = update_kc!(solver; kws..., n=0)


"""
    AuxKineticCoefficientSolver([auxfun], hv, bz, bzalg, Σ, [fdom,] falg, [linalg=JLInv()]; n, β, Ω, μ=0, scale_inner=inv(abs(det(bz.B))*nsyms(bz)), kws...)
    AuxKineticCoefficientSolver([auxfun], Σ, [fdom,] falg, hv, bz, bzalg, [linalg=JLInv()]; n, β, Ω, μ=0, scale_inner=1, kws...)

A solver for kinetic coefficients using an auxiliary integrand.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
The default `auxfun` is the sum of the Green's functions.
Use `AutoBZ.update_auxkc!(solver; β, Ω, μ, n)` to change parameters.
If `fdom` is not specified the default is `(AutoBZ.lb(Σ), AutoBZ.ub(Σ))`.
"""
function AuxKineticCoefficientSolver end

update_auxkc!(args...; kws...) = update_kc!(args...; kws...)


"""
    AuxOpticalConductivitySolver([auxfun], hv, bz, bzalg, Σ, [fdom,] falg, [linalg=JLInv()]; β, Ω, μ=0, scale_inner=inv(abs(det(bz.B))*nsyms(bz)), kws...)
    AuxOpticalConductivitySolver([auxfun], Σ, [fdom,] falg, hv, bz, bzalg, [linalg=JLInv()]; β, Ω, μ=0, scale_inner=1, kws...)

A solver for the optical conductivity. For details see [`AuxKineticCoefficientSolver`](@ref)
and note that by default the parameter `n=0`. Use `AutoBZ.update_auxoc!(solver; β, Ω, μ)` to
change parameters.
"""
AuxOpticalConductivitySolver(args...; kws...) = AuxKineticCoefficientSolver(args...; kws..., n=0)
update_auxoc!(solver; kws...) = update_auxkc!(solver; kws..., n=0)
