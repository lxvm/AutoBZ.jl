function Gloc2Solver(h::AbstractHamiltonianInterp, bz, bzalg, linalg=JLInv(); ω, Σ, μ=zero(ω), kws...)
    p = evalM(; ω, Σ, μ)
    k = SVector(period(h))
    hk= h(k)
    g = gauge(h)
    linprob = LinearSystemProblem(g isa Hamiltonian ? p - Diagonal(hk.values) : p-hk)
    up = (solver, k, hk, p) -> begin
        _hk = g isa Hamiltonian ? Diagonal(hk.values) : hk
        if ismutable(solver.A)
            solver.A .= p .- _hk
        else
            solver.A = p - _hk
        end
        return
    end
    post = (sol, k, hk, p) -> inv(sol.value)
    proto = post(solve(linprob, linalg), k, hk, p)
    f = CommonSolveFourierIntegralFunction(linprob, linalg, up, post, h, proto)
    prob = AutoBZProblem(UnknownRep(), f, bz, p; kws...)
    return init(prob, bzalg)
end
