function _DynamicalTransportDistributionSolver(fun::F, Σ::AbstractSelfEnergy, fdom, falg, hv::AbstractVelocityInterp, bz, bzalg, linalg; β, Ω, n, μ=zero(Ω), kws...) where {F}
    inner_kws = _rescale_abstol(inv(max(Ω, inv(β))); kws...)
    p = (; β, μ, Ω, n)
    up = (solver, ω, (_, (; β, μ, Ω, n))) -> (update_td!(solver; ω₁=ω, ω₂=ω+Ω, μ) ; return)
    post = (sol, ω, (_, (; β, μ, Ω, n))) -> (ω*β)^n * fermi_window(β, ω, Ω) * sol.value
    td_prob = _TransportDistributionProblem(fun, Σ, hv, bz, linalg; ω₁=zero(Ω), ω₂=Ω, μ, inner_kws...)
    proto = (float(zero(Ω))*β)^n * fermi_window(β, float(zero(Ω)), Ω) * td_prob.f.prototype
    f = CommonSolveIntegralFunction(td_prob, _heuristic_bzalg(bzalg, Σ, hv), up, post, proto)
    prob = IntegralProblem(f, get_safe_fermi_window_limits(Ω, β, fdom...), (fdom, p); kws...)
    return init(prob, falg)
end

function update_kc!(solver::AutoBZCore.IntegralSolver; β, Ω, n, μ=zero(Ω))
    fdom = solver.p[1]
    if solver.p[2].Ω != Ω || solver.p[2].β != β
        solver.dom = get_safe_fermi_window_limits(Ω, β, fdom...)
        # TODO rescale inner tolerance based on domain length
    end
    solver.p = (fdom, (; β, μ, Ω, n))
    return
end

function _DynamicalTransportDistributionSolver(fun::F, hv::AbstractVelocityInterp, bz, bzalg, Σ::AbstractSelfEnergy, fdom, falg, linalg; β, Ω, n, μ=zero(inv(oneunit(β))), kws...) where {F}
    M = evalM2(; Σ, ω₁=float(zero(Ω)), ω₂=float(Ω), μ)
    k = SVector(period(hv))
    hvk = hv(k)
    g = gauge(hv)
    A = g isa Hamiltonian ? Diagonal(hvk[1].values) : hvk[1]
    prob_k = TwoGreensFunctionProblem(A, M...)
    alg = TwoGreensFunctionLinearSystem(linalg)
    p = (; β, μ, Ω, n)
    p_k = (fdom, deepcopy(Σ), hvk, p)
    up_k = (solver, ω, (_, Σ, hvk, (; β, μ, Ω, n))) -> begin
        solver.M1, solver.M2, solver.isdistinct = evalM2(; Σ, ω₁=ω, ω₂=ω+Ω, μ) # WARN: Σ evaluation may not be threadsafe so need another prob type
        solver.h = g isa Hamiltonian ? Diagonal(hvk[1].values) : hvk[1]
        return
    end
    post_k = (sol, ω, (_, Σ, hvk, (; β, μ, Ω, n))) -> (ω*β)^n * fermi_window(β, ω, Ω) * fun(transport_distribution_integrand(hvk[2], sol.G1, sol.G2, sol.isdistinct), hvk..., sol)
    proto = post_k(solve(prob_k, alg), (fdom[1]+fdom[2])/2, p_k)
    f_k = CommonSolveIntegralFunction(prob_k, alg, up_k, post_k, proto)
    V = abs(det(bz.B))
    inner_kws = _rescale_abstol(inv(V*nsyms(bz)); kws...)
    fprob = IntegralProblem(f_k, get_safe_fermi_window_limits(Ω, β, fdom...), p_k; inner_kws...)
    up = (solver, k, hv, p) -> begin
        # if iszero(Ω) && isinf(β)
        #     # we pass in β=4 since fermi_window(4,0,0)=1, the weight of the delta
        #     # function, and also this prevents (0*β)^n from giving NaN when n!=0
        #     return Ω * f.f(Ω, MixedParameters(; Σ, n, β=4*oneunit(β), Ω, μ, hv_k))
        # end
        fdom, Σ, = solver.p
        if solver.p[4].Ω != p.Ω || solver.p[4].β != p.β
            solver.dom = get_safe_fermi_window_limits(p.Ω, p.β, fdom...)
            # TODO rescale inner tolerance based on domain length
        end
        solver.p = (fdom, Σ, hv, p)
        return
    end
    post = (sol, k, h, p) -> sol.value
    f = CommonSolveFourierIntegralFunction(fprob, falg, up, post, hv, proto*Ω)
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
    KineticCoefficientSolver(hv, bz, bzalg, Σ, [fdom,] falg, [linalg=JLInv()]; n, β, Ω, μ=0, kws...)
    KineticCoefficientSolver(Σ, [fdom,] falg, hv, bz, bzalg, [linalg=JLInv()]; n, β, Ω, μ=0, kws...)

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
    OpticalConductivitySolver(hv, bz, bzalg, Σ, [fdom,] falg, [linalg=JLInv()]; β, Ω, μ=0, kws...)
    OpticalConductivitySolver(Σ, [fdom,] falg, hv, bz, bzalg, [linalg=JLInv()]; β, Ω, μ=0, kws...)

A solver for the optical conductivity. For details see [`KineticCoefficientSolver`](@ref)
and note that by default the parameter `n=0`. Use `AutoBZ.update_oc!(solver; β, Ω, μ)` to
change parameters.
"""
OpticalConductivitySolver(args...; kws...) = KineticCoefficientSolver(args...; kws..., n=0)
update_oc!(solver; kws...) = update_kc!(solver; kws..., n=0)


"""
    AuxKineticCoefficientSolver([auxfun], hv, bz, bzalg, Σ, [fdom,] falg, [linalg=JLInv()]; n, β, Ω, μ=0, kws...)
    AuxKineticCoefficientSolver([auxfun], Σ, [fdom,] falg, hv, bz, bzalg, [linalg=JLInv()]; n, β, Ω, μ=0, kws...)

A solver for kinetic coefficients using an auxiliary integrand.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
The default `auxfun` is the sum of the Green's functions.
Use `AutoBZ.update_auxkc!(solver; β, Ω, μ, n)` to change parameters.
If `fdom` is not specified the default is `(AutoBZ.lb(Σ), AutoBZ.ub(Σ))`.
"""
function AuxKineticCoefficientSolver(auxfun::F, Σ::AbstractSelfEnergy, fdom, falg::IntegralAlgorithm, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, linalg::LinearSystemAlgorithm=JLInv(); kws...) where {F}
    _DynamicalTransportDistributionSolver((Γ, h, v, sol) -> AutoBZCore.IteratedIntegration.AuxValue(Γ, auxfun(v, sol.G1, sol.G2)), Σ, fdom, falg, hv, bz, bzalg, linalg; kws...)
end
function AuxKineticCoefficientSolver(Σ::AbstractSelfEnergy, fdom::Tuple, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    AuxKineticCoefficientSolver(_trG_auxfun, Σ, fdom, falg, hv, bz, bzalg, linalg; kws...)
end
function AuxKineticCoefficientSolver(auxfun::F, Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, linalg::LinearSystemAlgorithm=JLInv(); kws...) where {F}
    AuxKineticCoefficientSolver(Σ, (lb(Σ), ub(Σ)), falg, hv, bz, bzalg, linalg; kws...)
end
function AuxKineticCoefficientSolver(Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    AuxKineticCoefficientSolver(_trG_auxfun, Σ, falg, hv, bz, bzalg, linalg; kws...)
end

function AuxKineticCoefficientSolver(auxfun::F, h::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, fdom, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...) where {F}
    _DynamicalTransportDistributionSolver((Γ, h, v, sol) -> AutoBZCore.IteratedIntegration.AuxValue(Γ, auxfun(v, sol.G1, sol.G2)), h, bz, bzalg, Σ, fdom, falg, linalg; kws...)
end
function AuxKineticCoefficientSolver(h::AbstractVelocityInterp, bz, bzalg, Σ::AbstractSelfEnergy, fdom, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    AuxKineticCoefficientSolver(_trG_auxfun, h, bz, bzalg, Σ, fdom, falg, linalg; kws...)
end
function AuxKineticCoefficientSolver(auxfun::F, h::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...) where {F}
    AuxKineticCoefficientSolver(auxfun, h, bz, bzalg, Σ, (lb(Σ), ub(Σ)), falg, linalg; kws...)
end
function AuxKineticCoefficientSolver(h::AbstractVelocityInterp, bz, bzalg, Σ::AbstractSelfEnergy, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    AuxKineticCoefficientSolver(_trG_auxfun, h, bz, bzalg, Σ, falg, linalg; kws...)
end
update_auxkc!(args...; kws...) = update_kc!(args...; kws...)


"""
    AuxOpticalConductivitySolver([auxfun], hv, bz, bzalg, Σ, [fdom,] falg, [linalg=JLInv()]; β, Ω, μ=0, kws...)
    AuxOpticalConductivitySolver([auxfun], Σ, [fdom,] falg, hv, bz, bzalg, [linalg=JLInv()]; β, Ω, μ=0, kws...)

A solver for the optical conductivity. For details see [`AuxKineticCoefficientSolver`](@ref)
and note that by default the parameter `n=0`. Use `AutoBZ.update_auxoc!(solver; β, Ω, μ)` to
change parameters.
"""
AuxOpticalConductivitySolver(args...; kws...) = AuxKineticCoefficientSolver(args...; kws..., n=0)
update_auxoc!(solver; kws...) = update_auxkc!(solver; kws..., n=0)
