function TrGloc2Solver(h::AbstractHamiltonianInterp, bz, bzalg, trinvalg=JLTrInv(); ω, Σ, μ=zero(ω), kws...)
    p = evalM(; ω, Σ, μ)
    k = SVector(period(h))
    hk= h(k)
    g = gauge(h)
    trinvprob = TraceInverseProblem(g isa Hamiltonian ? p - Diagonal(hk.values) : p-hk)
    up = (solver, k, hk, p) -> begin
        _hk = g isa Hamiltonian ? Diagonal(hk.values) : hk
        if ismutable(solver.A)
            solver.A .= p .- _hk
        else
            solver.A = p - _hk
        end
        return
    end
    post = (sol, k, hk, p) -> sol.value
    proto = post(solve(trinvprob, trinvalg), k, hk, p)
    f = CommonSolveFourierIntegralFunction(trinvprob, trinvalg, up, post, h, proto)
    prob = AutoBZProblem(TrivialRep(), f, bz, p; kws...)
    return init(prob, bzalg)
end
