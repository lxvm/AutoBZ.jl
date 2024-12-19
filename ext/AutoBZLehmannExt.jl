module AutoBZLehmannExt

using AutoBZ
using AutoBZ: AbstractHamiltonianInterp, TraceInverseAlgorithm
using Lehmann
import CommonSolve: init, solve!
using LinearAlgebra

struct LehmannProblem{G,B,P}
    g::G
    β::B
    p::P
end

mutable struct LehmannSolver{G,B,P,A,D,M,C}
    g::G
    β::B
    p::P
    alg::A
    dlrgrid::D
    Gmatdata::M
    cacheval::C
    isfresh::Bool # whether β has been modified
end
function Base.setproperty!(solver::LehmannSolver, name::Symbol, x)
    if name == :β
        setfield!(solver, :isfresh, true)
    end
    return setfield!(solver, name, x)
end

struct LehmannSolution{V,S}
    value::V
    retcode::AutoBZCore.ReturnCode
    stats::S
end

function AutoBZ.ElectronDensitySolver(Σ::AbstractSelfEnergy, falg::LehmannJL, h::AbstractHamiltonianInterp, bz, bzalg, trinvalg::TraceInverseAlgorithm=JLTrInv(); β, μ=zero(inv(oneunit(β))), scale_inner=nothing, kws...)
    # TODO better estimate the bandwidth since the Fermi function is a semi-infinite window
    bandwidth = oneunit(μ)
    inner_kws = AutoBZ._rescale_abstol(something(scale_inner, inv(bandwidth)); kws...)
    kprob = AutoBZ._GreensProblem(identity, Σ, h, bz, trinvalg; ω=complex(zero(μ)), μ, inner_kws...)
    kalg = AutoBZ._heuristic_bzalg(bzalg, Σ, h)
    up = (solver, ω, μ) -> AutoBZ.update_greens!(solver; ω, μ)
    post = (sol, ω, μ) -> sol.value
    proto = kprob.f.prototype * det(bz.B)
    fprob = LehmannProblem(CommonSolveFunction(kprob, kalg, up, post, proto), β, μ)
    return init(fprob, falg)
end

function init(prob::LehmannProblem, alg::LehmannJL)
    (; g, β, p) = prob
    dlrgrid = DLRGrid(alg.Λ, β, alg.rtol, alg.isFermi; alg.kws...)
    isfresh = false
    cacheval, prototype = _init_cacheval(g, p)
    Gmatdata = Vector{typeof(prototype)}(undef, length(dlrgrid.ωn))
    return LehmannSolver(g, β, p, alg, dlrgrid, Gmatdata, cacheval, isfresh)
end

function solve!(solver::LehmannSolver)
    (; g, β, p, alg, dlrgrid, Gmatdata, cacheval, isfresh) = solver
    if isfresh
        dlrgrid = DLRGrid(alg.Λ, β, alg.rtol, alg.isFermi; alg.kws...)
        solver.dlrgrid = dlrgrid
        solver.isfresh = false
    end
    resize!(Gmatdata, length(dlrgrid.ωn))
    _batcheval!(g, Gmatdata, dlrgrid.ωn, p, cacheval)
    dlrcoeff = matfreq2dlr(dlrgrid, Gmatdata)
    Gtaudata = dlr2tau(dlrgrid, dlrcoeff, [β])
    value = real(-Gtaudata[1])
    retcode = AutoBZCore.Success
    stats = (; )
    return LehmannSolution(value, retcode, stats)
end

struct CommonSolveFunction{P,A,U,F,T}
    prob::P
    alg::A
    update!::U
    post::F
    prototype::T
end

function _init_cacheval(g::CommonSolveFunction, p)
    solver = init(g.prob, g.alg)
    prototype = g.prototype
    return solver, prototype
end
function _batcheval!(g::CommonSolveFunction, Gmatdata, ωn, p, solver)
    f = ω -> begin
        g.update!(solver, im*ω, p)
        sol = solve!(solver)
        return g.post(sol, im*ω, p)
    end
    map!(f, Gmatdata, ωn)
    return
end
function AutoBZ.update_density!(solver::LehmannSolver; β, μ=zero(inv(oneunit(β))))
    solver.p = μ
    if β != solver.β
        solver.β = β
    end
    return
end

end
