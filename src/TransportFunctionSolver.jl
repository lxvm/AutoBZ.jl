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

"""
    TransportFunctionSolver(hv::AbstractVelocityInterp, bz, bzalg; β, μ=0, kws...)

A function whose integral over the BZ gives the transport function, proportional
to the Drude weight,
```math
D_{\\alpha\\beta} = \\sum_{nm} \\int_{\\text{BZ}} dk f'(\\epsilon_{nk}-\\mu) \\nu_{n\\alpha}(k) \\nu_{m\\beta}(k)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distribution.
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_tf!(solver; β, μ=0)` to update the parameters.
"""
function TransportFunctionSolver(hv::AbstractVelocityInterp, bz, bzalg; β, μ=zero(inv(oneunit(β))), kws...)
    @assert gauge(hv) isa Hamiltonian
    p = (; β, μ)
    k = SVector(period(hv))
    hvk = hv(k)
    proto = transport_function_integrand(k, hvk, p)
    f = FourierIntegralFunction(transport_function_integrand, hv, proto)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kws...)
    return init(prob, _heuristic_bzalg(bzalg, π/β, hv))
end
