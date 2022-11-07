iterated_pre_eval(w::WannierIntegrand, x) = WannierIntegrand(w.f, contract(w.s, x), w.p)

iterated_pre_eval(D::DOSIntegrand, k) = DOSIntegrand(contract(D.H, k), D.M)

iterated_pre_eval(g::GammaIntegrand, k) = GammaIntegrand(contract(g.HV, k), g.Mω, g.MΩ)

iterated_pre_eval(f::OCIntegrand, k) = OCIntegrand(contract(f.HV, k), f.Σ, f.Ω, f.β, f.μ)

infer_f(::T, _) where {T<:Union{DOSIntegrand,GammaIntegrand,OCIntegrand}} = (eltype(T), Base.promote_op(norm, eltype(T)))