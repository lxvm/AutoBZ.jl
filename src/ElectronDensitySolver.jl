function _DynamicalOccupiedGreensSolver(fun::F, Σ::AbstractSelfEnergy, fdom, falg, h::AbstractHamiltonianInterp, bz, bzalg, linalg; β, μ=zero(inv(oneunit(β))), bandwidth=one(μ), kws...) where {F}
    # TODO better estimate the bandwidth since the Fermi function is a semi-infinite window
    V = abs(det(bz.B))
    inner_kws = _rescale_abstol(inv(bandwidth); kws...)
    dos_prob = _GreensProblem(fun, Σ, h, bz, linalg; ω=(fdom[1]+fdom[2])/2, μ, inner_kws...)
    p = (; β, μ)
    proto = dos_prob.f.prototype * V * fermi(β, (fdom[1]+fdom[2])/2)
    # WARN: Σ evaluation in update_greens! may not be threadsafe so need another prob type
    up = (solver, ω, (_, (; β, μ))) -> (update_greens!(solver; ω, μ); return)
    post = (sol, ω, (_, (; β, μ))) -> sol.value*fermi(β, ω)
    f = CommonSolveIntegralFunction(dos_prob, _heuristic_bzalg(bzalg, Σ, h), up, post, proto)
    prob = IntegralProblem(f, get_safe_fermi_function_limits(β, fdom...), (fdom, p); kws...)
    return init(prob, falg)
end

function _rescale_abstol(s; kws...)
    haskey(NamedTuple(kws), :abstol) || return (; kws...)
    return (; kws..., abstol=NamedTuple(kws).abstol*s)
end

function update_density!(solver::AutoBZCore.IntegralSolver; β, μ=zero(inv(oneunit(β))))
    fdom = solver.p[1]
    if β != solver.p[2].β
        solver.dom = get_safe_fermi_function_limits(β, fdom...)
        # TODO rescale inner tolerance based on domain length
    end
    solver.p = (fdom, (; β, μ))
    return
end

function _DynamicalOccupiedGreensSolver(fun::F, h::AbstractHamiltonianInterp, bz, bzalg, Σ::AbstractSelfEnergy, fdom, falg, linalg; β, μ=zero(inv(oneunit(β))), kws...) where {F}
    V = abs(det(bz.B))
    k = period(h)
    hk = h(k)
    g = gauge(h)
    M = evalM(; Σ, ω=(fdom[1]+fdom[2])/2, μ)
    A = g isa Hamiltonian ? M - Diagonal(hk.values) : M -hk
    linprob, rep =  linalg isa LinearSystemAlgorithm ? (LinearSystemProblem(A), UnknownRep()) :
                    linalg isa TraceInverseAlgorithm ? (TraceInverseProblem(A), TrivialRep()) :
                    throw(ArgumentError("$linalg is neither a LinearSystemAlgorithm nor TraceInverseAlgorithm"))
    p = (; β, μ)
    p_k = (fdom, deepcopy(Σ), hk, p)
    up_k = (solver, ω, (_, Σ, hk, (; β, μ))) -> begin
        M = evalM(; Σ, ω, μ) # WARN: Σ evaluation may not be threadsafe so need another prob type
        _hk = g isa Hamiltonian ? Diagonal(hk.values) : hk
        if ismutable(solver.A)
            solver.A .= M .- _hk
        else
            solver.A = M - _hk
        end
        return
    end
    post_k = (sol, ω, (_, Σ, hk, (; β, μ))) -> if sol isa LinearSystemSolution
        inv(sol.value)
    elseif sol isa TraceInverseSolution
        sol.value
    else
        error("$sol is neither a LinearSystemSolution nor TraceInverseSolution")
    end |> fun |> x -> x*fermi(β, ω)
    inner_kws = _rescale_abstol(inv(V*nsyms(bz)); kws...)
    proto = post_k(solve(linprob, linalg), (fdom[1]+fdom[2])/2, p_k)
    f = CommonSolveIntegralFunction(linprob, linalg, up_k, post_k, proto)
    fprob = IntegralProblem(f, get_safe_fermi_function_limits(β, fdom...), p_k; inner_kws...)
    linprob, rep =  linalg isa LinearSystemAlgorithm ? (LinearSystemProblem(A), UnknownRep()) :
                    linalg isa TraceInverseAlgorithm ? (TraceInverseProblem(A), TrivialRep()) :
                    throw(ArgumentError("$linalg is neither a LinearSystemAlgorithm nor TraceInverseAlgorithm"))
    up = (solver, k, h, p) -> begin
        fdom, Σ, = solver.p
        if solver.p[4].β != p.β
            solver.dom = get_safe_fermi_function_limits(p.β, fdom...)
            # TODO rescale inner tolerance based on domain length
        end
        solver.p = (fdom, Σ, h, p)
        return
    end
    post = (sol, k, h, p) -> sol.value
    g = CommonSolveFourierIntegralFunction(fprob, falg, up, post, h, proto*μ)
    prob = AutoBZProblem(rep, g, bz, p; kws...)
    return init(prob, _heuristic_bzalg(bzalg, Σ, h))
end

function update_density!(solver::AutoBZCore.AutoBZCache; β, μ=zero(inv(oneunit(β))))
    solver.p = (; β, μ)
    return
end

"""
    ElectronDensitySolver(h, bz, bzalg, Σ, [fdom,] falg, [trinvalg=JLTrInv()]; β, μ=0, kws...)
    ElectronDensitySolver(Σ, [fdom,] falg, h, bz, bzalg, [trinvalg=JLTrInv()]; β, μ=0, bandwidth=1, kws...)

A solver for the electron density.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
Use `AutoBZ.update_density!(solver; β, μ=0)`.
If `fdom` is not specified the default is `(AutoBZ.lb(Σ), AutoBZ.ub(Σ))`.
The `bandwidth` keyword is used to rescale inner tolerances and should be estimated to
reduce effort.
`trinvalg` may be specified as an algorithm for the inverse trace calculation.

Mathematically, this computes the electron density:
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
To get the density/number of electrons, multiply the result of this integral by `n_sp/det(bz.B)`
"""
function ElectronDensitySolver(Σ::AbstractSelfEnergy, fdom, falg::IntegralAlgorithm, h::AbstractHamiltonianInterp, bz, bzalg, trinvalg::TraceInverseAlgorithm=JLTrInv(); kws...)
    _DynamicalOccupiedGreensSolver(spectral_function, Σ, fdom, falg, h, bz, bzalg, trinvalg; kws...)
end
function ElectronDensitySolver(Σ::AbstractSelfEnergy, falg, h::AbstractHamiltonianInterp, bz, bzalg, trinvalg::TraceInverseAlgorithm=JLTrInv(); kws...)
    ElectronDensitySolver(Σ, (lb(Σ), ub(Σ)), falg, h, bz, bzalg, trinvalg; kws...)
end

function ElectronDensitySolver(h::AbstractHamiltonianInterp, bz, bzalg, Σ::AbstractSelfEnergy, fdom, falg, trinvalg::TraceInverseAlgorithm=JLTrInv(); kws...)
    _DynamicalOccupiedGreensSolver(spectral_function, h, bz, bzalg, Σ, fdom, falg, trinvalg; kws...)
end
function ElectronDensitySolver(h::AbstractHamiltonianInterp, bz, bzalg, Σ::AbstractSelfEnergy, falg, trinvalg::TraceInverseAlgorithm=JLTrInv(); kws...)
    ElectronDensitySolver(h, bz, bzalg, Σ, (lb(Σ), ub(Σ)), falg, trinvalg; kws...)
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
