
#=
TODO
Calculate the convolution ∫_BZ dk G(ω, k) G(ω,q-k)
=#
function twogreensbzconvolutionintegrand(Gk, Gkmq)
    Gqmk = conj.(Gkmq)
    return Gk * Gqmk
end

function _TwoGreensBZConvolutionProblem(fun::F, Σ::AbstractSelfEnergy, h::AbstractHamiltonianInterp, bz, linalg::LinearSystemAlgorithm; q, ω, μ=zero(ω), kws...) where {F}
    p = (q, Σ, evalM(; Σ, ω, μ))
    k = SVector(period(h))
    h2 = ManyFourierSeries(h, TranslatedFourierSeries(h, Tuple(-q)); period=FourierSeriesEvaluators.period(h))
    h2k = h2(k)
    g = gauge(h)
    prob = TwoGreensFunctionProblem(g isa Hamiltonian ? Diagonal(h2k[1].values) : h2k[1], g isa Hamiltonian ? Diagonal(h2k[2].values) : h2k[2], _to_gauge(g, h2k[1], p[3]), _to_gauge(g, h2k[2], p[3]), iszero(q))
    alg = TwoGreensFunctionLinearSystem(linalg)
    up = (solver, k, h2k, (q, Σ, M)) -> begin
        solver.h1 = g isa Hamiltonian ? Diagonal(h2k[1].values) : h2k[1]
        solver.h2 = g isa Hamiltonian ? Diagonal(h2k[2].values) : h2k[2]
        solver.M1 = g isa Hamiltonian ? _to_gauge(g, h2k[1], M) : M
        solver.M2 = g isa Hamiltonian ? _to_gauge(g, h2k[2], M) : M
        solver.isdistinct = iszero(q)
        return
    end
    post = (sol, k, h2k, p) -> fun(twogreensbzconvolutionintegrand(sol.G1, sol.G2), k, h2k, p, sol)
    proto = post(solve(prob, alg), k, h2k, p)
    f = CommonSolveFourierIntegralFunction(prob, alg, up, post, h2, proto)
    return AutoBZProblem(UnknownRep(), f, bz, p; kws...)
end

function TwoGreensBZConvolutionSolver(Σ, h, bz, bzalg, linalg=JLInv(); kws...)
    prob = _TwoGreensBZConvolutionProblem((conv, _...) -> conv, Σ, h, bz, linalg; kws...)
    return init(prob, _heuristic_bzalg(bzalg, Σ, h))
end

function update_2g!(solver; q, ω, μ=zero(ω))
    Σ = solver.p[2]
    solver.p = (q, Σ, evalM(; Σ, ω, μ))
    return
end