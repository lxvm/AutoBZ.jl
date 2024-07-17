function transport_distribution_integrand_(vs::SVector{N,V}, Aω₁::A, Aω₂::A) where {N,V,A}
    vsAω₁ = map(v -> v * Aω₁, vs)
    vsAω₂ = map(v -> v * Aω₂, vs)
    return tr_kron(vsAω₁, vsAω₂)
end
function transport_distribution_integrand_(vs::SVector{N,V}, Aω::A) where {N,V,A}
    vsAω = map(v -> v * Aω, vs)
    return tr_kron(vsAω, vsAω)
end

function transport_distribution_integrand(k, v, (Σ, (Mω₁, Mω₂, isdistinct)))
    h, vs = v
    if isdistinct
        Aω₁ = spectral_function(h, Mω₁)
        Aω₂ = spectral_function(h, Mω₂)
        return transport_distribution_integrand_(vs, Aω₁, Aω₂)
    else
        Aω = spectral_function(h, Mω₁)
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
function TransportDistributionSolver(Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, bzalg; ω₁, ω₂, μ=zero(ω₁), kwargs...)
    p = (Σ, evalM2(; Σ, ω₁, ω₂, μ))
    k = SVector(period(hv))
    hvk = hv(k)
    proto = transport_distribution_integrand(k, hvk, p)
    f = FourierIntegralFunction(transport_distribution_integrand, hv, proto)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kwargs...)
    return init(prob, bzalg)
end


abstract type TwoSpectralFunctionAlgorithm end

struct TwoSpectralFunctionProblem{H,M,K}
    h::H
    M1::M
    M2::M
    isdistinct::Bool
    kwargs::K
end
TwoSpectralFunctionProblem(h, M1, M2, isdistinct=M1==M2; kws...) = TwoSpectralFunctionProblem(h, M1, M2, isdistinct, kws)

mutable struct TwoSpectralFunctionSolver{H,M,K,A,C}
    h::H
    M1::M
    M2::M
    isdistinct::Bool
    kwargs::K
    alg::A
    cacheval::C
end

struct TwoSpectralFunctionSolution{A,S}
    A1::A
    A2::A
    retcode::ReturnCode
    stats::S
end

function init(prob::TwoSpectralFunctionProblem, alg::TwoSpectralFunctionAlgorithm; kws...)
    kwargs = (; prob.kwargs..., kws...)
    cacheval = init_cacheval(prob, alg)
    return TwoSpectralFunctionSolver(prob.h, prob.M1, prob.M2, prob.isdistinct, kwargs, alg, cacheval)
end
solve!(solver::TwoSpectralFunctionSolver) = do_twospectralsolve!(solver.h, solver.M1, solver.M2, solver.isdistinct, solver.alg, solver.cacheval; solver.kwargs...)

struct SpectralFunctionLinearSystem{A<:LinearSystemAlgorithm} <: TwoSpectralFunctionAlgorithm
    linalg::A
end

function init_cacheval(prob::TwoSpectralFunctionProblem, alg::SpectralFunctionLinearSystem)
    linprob = LinearSystemProblem(prob.M1-prob.h)
    return init(linprob, alg.linalg)
end

function do_twospectralsolve!(h, M1, M2, isdistinct, alg::SpectralFunctionLinearSystem, cacheval)
    ismutable(cacheval.A) && error("spectral function not implemented for mutable types")
    cacheval.A = M1 - h
    A1 = spectral_function(inv(solve!(cacheval).value))
    if isdistinct
        cacheval.A = M2 - h
        A2 = spectral_function(inv(solve!(cacheval).value))
    else
        A2 = A1
    end
    retcode = Success
    stats = (;)
    return TwoSpectralFunctionSolution(A1, A2, retcode, stats)
end


function TransportDistribution2Solver(hv::AbstractVelocityInterp, bz, bzalg, linalg=JLInv(); Σ, ω₁, ω₂, μ=zero(ω₁), kws...)
    p = evalM2(; Σ, ω₁, ω₂, μ)
    k = SVector(period(hv))
    hvk = hv(k)
    g = gauge(hv)
    linprob = TwoSpectralFunctionProblem(g isa Hamiltonian ? Diagonal(hvk[1].values) : hvk[1], p...)
    up = (solver, k, hvk, (M1, M2, isdistinct)) -> begin
        solver.h = g isa Hamiltonian ? Diagonal(hvk[1].values) : hvk[1]
        solver.M1 = M1
        solver.M2 = M2
        solver.isdistinct = isdistinct
        return
    end
    post = (sol, k, hvk, p) -> transport_distribution_integrand_(hvk[2], sol.A1, sol.A2)
    proto = transport_distribution_integrand(k, hvk, p)
    f = CommonSolveFourierIntegralFunction(linprob, SpectralFunctionLinearSystem(linalg), up, post, hv, proto)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kws...)
    return init(prob, bzalg)
end
