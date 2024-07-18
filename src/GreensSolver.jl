# shift energy by chemical potential, but not self energy
_evalM(Σ::Union{AbstractMatrix,UniformScaling}, ω, μ) = (ω+μ)*I - Σ
_evalM(Σ::AbstractSelfEnergy, ω, μ) = _evalM(Σ(ω), ω, μ)
evalM(; Σ, ω, μ=zero(ω)) = _evalM(Σ, ω, μ)

function update_greens!(solver; ω, μ=zero(ω))
    Σ = solver.p[1]
    solver.p = (Σ, evalM(; ω, Σ, μ))
    return
end

function _GreensProblem(fun::F, Σ::AbstractSelfEnergy, h::AbstractHamiltonianInterp, bz, linalg; ω, μ=zero(ω), kws...) where {F}
    p = (deepcopy(Σ), evalM(; ω, Σ, μ))
    k = SVector(period(h))
    hk= h(k)
    g = gauge(h)
    A = g isa Hamiltonian ? p[2] - Diagonal(hk.values) : p[2]-hk
    linprob, rep =  linalg isa LinearSystemAlgorithm ? (LinearSystemProblem(A), UnknownRep()) :
                    linalg isa TraceInverseAlgorithm ? (TraceInverseProblem(A), TrivialRep()) :
                    throw(ArgumentError("$linalg is neither a LinearSystemAlgorithm nor TraceInverseAlgorithm"))
    up = (solver, k, hk, (Σ, p)) -> begin
        _hk = g isa Hamiltonian ? Diagonal(hk.values) : hk
        if ismutable(solver.A)
            solver.A .= p .- _hk
        else
            solver.A = p - _hk
        end
        return
    end
    post = (sol, k, hk, p) -> if sol isa LinearSystemSolution
        inv(sol.value)
    elseif sol isa TraceInverseSolution
        sol.value
    else
        error("$sol is neither a LinearSystemSolution nor TraceInverseSolution")
    end |> fun
    proto = post(solve(linprob, linalg), k, hk, p)
    f = CommonSolveFourierIntegralFunction(linprob, linalg, up, post, h, proto)
    return AutoBZProblem(rep, f, bz, p; kws...)
end
function _GreensSolver(fun::F, Σ, h, bz, bzalg, linalg; kws...) where {F}
    prob = _GreensProblem(fun, Σ, h, bz, linalg; kws...)
    return init(prob, _heuristic_bzalg(bzalg, Σ, h))
end

"""
    GlocSolver(Σ, h, bz, bzalg, [linalg=JLInv()]; ω, μ=0, kws...)

Green's function integrands accepting a self energy Σ that can either be a matrix or a
function of ω (see the self energy section of the documentation for examples)
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_gloc!(solver; ω, μ=0)` to change the parameters.
The `linalg` argument sets the linear system solver
"""
function GlocSolver(Σ, h, bz, bzalg, alg::LinearSystemAlgorithm=JLInv(); kws...)
    _GreensSolver(identity, Σ, h, bz, bzalg, alg; kws...)
end
update_gloc!(solver; kws...) = update_greens!(solver; kws...)

"""
    TrGlocSolver(Σ, h, bz, bzalg, [trinvalg=JLTrInv()]; ω, μ=0, kws...)

Green's function integrands accepting a self energy Σ that can either be a matrix or a
function of ω (see the self energy section of the documentation for examples)
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_trgloc!(solver; ω, μ=0)` to change the parameters.
"""
function TrGlocSolver(Σ, h, bz, bzalg, alg::TraceInverseAlgorithm=JLTrInv(); kws...)
    _GreensSolver(identity, Σ, h, bz, bzalg, alg; kws...)
end
update_trgloc!(solver; kws...) = update_greens!(solver; kws...)

function spectral_function(G::AbstractMatrix)
    T = real(eltype(G)) # get floating point type from input
    imtwo = complex(zero(one(T)), one(T)+one(T))
    return (G - G')/(-imtwo*pi)   # skew-Hermitian part
end
spectral_function(G::Union{Number,Diagonal}) = -imag(G)/pi # optimization

"""
    DOSSolver(Σ, h, bz, bzalg, [trinvalg=JLTrInv()]; ω, μ=0, kws...)

Green's function integrands accepting a self energy Σ that can either be a matrix or a
function of ω (see the self energy section of the documentation for examples)
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_dos!(solver; ω, μ=0)` to change the parameters.
"""
function DOSSolver(Σ, h, bz, bzalg, alg::TraceInverseAlgorithm=JLTrInv(); kws...)
    _GreensSolver(spectral_function, Σ, h, bz, bzalg, alg; kws...)
end
update_dos!(solver; kws...) = update_greens!(solver; kws...)
