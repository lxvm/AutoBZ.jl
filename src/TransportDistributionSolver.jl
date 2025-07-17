function transport_distribution_integrand_(vs::SVector{N,V}, Aω₁::A, Aω₂::A) where {N,V,A}
    vsAω₁ = map(v -> v * Aω₁, vs)
    vsAω₂ = map(v -> v * Aω₂, vs)
    return tr_kron(vsAω₁, vsAω₂)
end
function transport_distribution_integrand_(vs::SVector{N,V}, Aω::A) where {N,V,A}
    vsAω = map(v -> v * Aω, vs)
    return tr_kron(vsAω, vsAω)
end
function transport_distribution_integrand(vs, Gω₁, Gω₂, isdistinct)
    if isdistinct
        Aω₁ = spectral_function(Gω₁)
        Aω₂ = spectral_function(Gω₂)
        return transport_distribution_integrand_(vs, Aω₁, Aω₂)
    else
        Aω = spectral_function(Gω₁)
        return transport_distribution_integrand_(vs, Aω)
    end
end

function _evalM2(Σ, ω₁::T, ω₂::T, μ::T) where {T}
    M = _evalM(Σ, ω₁, μ)
    if ω₁ == ω₂
        (M, M, false)
    else
        (M, _evalM(Σ, ω₂, μ), true)
    end
end
evalM2(; Σ, ω₁, ω₂, μ=zero(ω₁)) = _evalM2(Σ, promote(ω₁, ω₂, μ)...)


#=

function TransportDistributionSolver(Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, bzalg; ω₁, ω₂, μ=zero(ω₁), kwargs...)
    p = (Σ, evalM2(; Σ, ω₁, ω₂, μ))
    k = SVector(period(hv))
    hvk = hv(k)
    proto = transport_distribution_integrand(k, hvk, p)
    f = FourierIntegralFunction(transport_distribution_integrand, hv, proto)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kwargs...)
    return init(prob, bzalg)
end

=#
abstract type TwoGreensFunctionAlgorithm end

struct TwoGreensFunctionProblem{H,M,K}
    h1::H
    h2::H
    M1::M
    M2::M
    isdistinct::Bool
    kwargs::K
end
TwoGreensFunctionProblem(h1, h2, M1, M2, isdistinct=(M1!=M2)||(h1!=h2); kws...) = TwoGreensFunctionProblem(h1, h2, M1, M2, isdistinct, kws)

mutable struct TwoGreensFunctionSolver{H,M,K,A,C}
    h1::H
    h2::H
    M1::M
    M2::M
    isdistinct::Bool
    kwargs::K
    alg::A
    cacheval::C
end

struct TwoGreensFunctionSolution{A,S}
    G1::A
    G2::A
    isdistinct::Bool
    retcode::ReturnCode
    stats::S
end

function init(prob::TwoGreensFunctionProblem, alg::TwoGreensFunctionAlgorithm; kws...)
    kwargs = (; prob.kwargs..., kws...)
    cacheval = init_cacheval(prob, alg)
    return TwoGreensFunctionSolver(prob.h1, prob.h2, prob.M1, prob.M2, prob.isdistinct, kwargs, alg, cacheval)
end
solve!(solver::TwoGreensFunctionSolver) = do_twogreenssolve!(solver.h1, solver.h2, solver.M1, solver.M2, solver.isdistinct, solver.alg, solver.cacheval; solver.kwargs...)

struct TwoGreensFunctionLinearSystem{A<:LinearSystemAlgorithm} <: TwoGreensFunctionAlgorithm
    linalg::A
end

function init_cacheval(prob::TwoGreensFunctionProblem, alg::TwoGreensFunctionLinearSystem)
    linprob = LinearSystemProblem(prob.M1-prob.h1)
    return init(linprob, alg.linalg)
end

function do_twogreenssolve!(h1, h2, M1, M2, isdistinct, alg::TwoGreensFunctionLinearSystem, cacheval)
    if ismutable(cacheval.A)
        cacheval.A .= M1 .- h1
    else
        cacheval.A = M1 - h1
    end
    G1 = inv(solve!(cacheval).value)
    if isdistinct
        if ismutable(cacheval.A)
            cacheval.A .= M2 .- h2
        else
            cacheval.A = M2 - h2
        end
        G2 = inv(solve!(cacheval).value)
    else
        G2 = G1
    end
    retcode = Success
    stats = (;)
    return TwoGreensFunctionSolution(G1, G2, isdistinct, retcode, stats)
end

function _to_gauge_twice(g, h, Σ1, Σ2, isdistinct)
    if isdistinct
        (_to_gauge(g, h, Σ1), _to_gauge(g, h, Σ2), isdistinct)
    else
        Σ = _to_gauge(g, h, Σ1)
        (Σ, Σ, isdistinct)
    end
end
function _TransportDistributionProblem(fun::F, Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, linalg::LinearSystemAlgorithm; ω₁, ω₂, μ=zero(ω₁), kws...) where {F}
    p = (Σ, evalM2(; Σ, ω₁, ω₂, μ))
    k = SVector(period(hv))
    hvk = hv(k)
    g = gauge(hv)
    A = g isa Hamiltonian ? Diagonal(hvk[1].values) : hvk[1]
    prob = TwoGreensFunctionProblem(A, A, _to_gauge_twice(g, hvk[1], p[2]...)...)
    alg = TwoGreensFunctionLinearSystem(linalg)
    up = (solver, k, hvk, (Σ, p2)) -> begin
        solver.h1 = solver.h2 = g isa Hamiltonian ? Diagonal(hvk[1].values) : hvk[1]
        solver.M1, solver.M2, solver.isdistinct = _to_gauge_twice(g, hvk[1], p2...)
        return
    end
    post = (sol, k, hvk, p) -> fun(transport_distribution_integrand(hvk[2], sol.G1, sol.G2, sol.isdistinct), hvk..., sol)
    proto = post(solve(prob, alg), k, hvk, p)
    f = CommonSolveFourierIntegralFunction(prob, alg, up, post, hv, proto)
    return AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kws...)
end

function update_td!(solver; ω₁, ω₂, μ=zero(ω₁))
    Σ = solver.p[1]
    solver.p = (Σ, evalM2(; Σ, ω₁, ω₂, μ))
    return
end

"""
    TransportDistributionSolver(Σ, hv::AbstractVelocityInterp, bz, bzalg; ω₁, ω₂, μ=0, kws...)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_td!(solver; ω₁, ω₂, μ=0)` to update the parameters.
"""
function TransportDistributionSolver(Σ, hv, bz, bzalg, linalg=JLInv(); kws...)
    prob = _TransportDistributionProblem((Γ,_...) -> Γ, Σ, hv, bz, linalg; kws...)
    return init(prob, _heuristic_bzalg(bzalg, Σ, hv))
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
function AuxTransportDistributionSolver(auxfun::F, Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, bzalg, linalg=JLInv(); kws...) where {F}
    prob = _TransportDistributionProblem((Γ, h, v, sol) -> AutoBZCore.IteratedIntegration.AuxValue(Γ, auxfun(v, sol.G1, sol.G2)), Σ, hv, bz, linalg; kws...)
    return init(prob, _heuristic_bzalg(bzalg, Σ, hv))
end
AuxTransportDistributionSolver(Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, bzalg, linalg=JLInv(); kws...) = AuxTransportDistributionSolver(_trG_auxfun, Σ, hv, bz, bzalg, linalg; _trG_kws(; kws...)...)
_trG_auxfun(vs, Gω₁, Gω₂) = tr(Gω₁) + tr(Gω₂)
function _trG_kws(; kws...)
    (!haskey(kws, :abstol) || !(kws[:abstol] isa AutoBZCore.IteratedIntegration.AuxValue)) && @warn "pick a sensible default auxiliary tolerance"
    return (; kws...)
end
update_auxtd!(solver; kws...) = update_td!(solver; kws...)
