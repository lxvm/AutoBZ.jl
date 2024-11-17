function transport_function_integrand((h, vs)::Tuple{Eigen,SVector}; β, μ)
    f′ = Diagonal(β .* fermi′.(β .* (h.values .- μ)))
    f′vs = map(v -> f′*v, vs)
    return tr_kron(vs, f′vs)
end
transport_function_integrand(k, hv, p) = transport_function_integrand(hv; p...)
function update_tf!(solver; β, μ=zero(inv(oneunit(β))))
    β != solver.p.β && solver.alg isa AutoPTR && @warn "changing β does not update AutoPTR heuristics"
    # solver.alg = _heuristic_bzalg(solver.alg, π/β, solver.f.s)
    # TODO also reinit the AutoPTR cache?
    solver.p = (; β, μ)
    return
end
function transport_function_integrand_lorentzian((h, vs); β, μ)
    A = spectral_function(_inv((μ-im/β)*I-h))
    Avs = map(v -> A*v, vs)
    return tr_kron(vs, f′vs)
end
transport_function_integrand_lorentzian(k, hv, p) = transport_function_integrand_lorentzian(hv; p...)


"""
    TransportFunctionSolver(hv::AbstractVelocityInterp, bz, bzalg; β, μ=0, kernel=:fermi, kws...)

A function whose integral over the BZ gives the transport function, proportional
to the Drude weight,
```math
D_{\\alpha\\beta} = \\sum_{nm} \\int_{\\text{BZ}} dk f'(\\epsilon_{nk}-\\mu) \\nu_{n\\alpha}(k) \\nu_{m\\beta}(k)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distribution.
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_tf!(solver; β, μ=0)` to update the parameters.

If the keyword `kernel` is set to `:lorentzian` then the following is computed
```math
D_{\\alpha\\beta} = \\sum_{nm} \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_{n\\alpha}(k) A(k, \\mu) \\nu_{m\\beta}(k)]
```
"""
function TransportFunctionSolver(hv::AbstractVelocityInterp, bz, bzalg; β, μ=zero(inv(oneunit(β))), kernel=:fermi, kws...)
    p = (; β, μ)
    k = SVector(period(hv))
    hvk = hv(k)
    f = if kernel == :fermi
        @assert gauge(hv) isa Hamiltonian
        proto = transport_function_integrand(k, hvk, p)
        FourierIntegralFunction(transport_function_integrand, hv, proto)
    elseif kernel == :lorentzian
        A = gauge(hv) isa Hamiltonian ? (μ+im/β)*I - Diagonal(hvk[1].values) : (μ+im/β)*I-hvk[1]
        linprob =  LinearSystemProblem(A)
        linalg = JLInv()
        up = (solver, k, hvk, (; β, μ)) -> begin
            M = (μ+im/β)*I
            _hk = gauge(hv) isa Hamiltonian ? Diagonal(hvk[1].values) : hvk[1]
            if ismutable(solver.A)
                solver.A .= M .- _hk
            else
                solver.A = M - _hk
            end
            return
        end
        post = (sol, k, hvk, p) -> begin
            G = inv(sol.value)
            A = spectral_function(G)
            vs = hvk[2]
            Avs = map(v -> A*v, vs)
            return tr_kron(vs, Avs)
        end
        proto = post(solve(linprob, linalg), k, hvk, p)
        CommonSolveFourierIntegralFunction(linprob, linalg, up, post, hv, proto)
    else
        error("kernel $kernel not recognized")
    end
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kws...)
    return init(prob, _heuristic_bzalg(bzalg, π/β, hv))
end
