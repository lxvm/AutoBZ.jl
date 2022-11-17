iterated_pre_eval(w::WannierIntegrand, x) = WannierIntegrand(w.f, contract(w.s, x), w.p)

iterated_pre_eval(D::DOSIntegrand, k) = DOSIntegrand(contract(D.H, k), D.M)
iterated_pre_eval(D::DOSIntegrand{<:AbstractFourierSeries3D}, k, dim) = (contract!(D.H, k, dim); return D)

iterated_pre_eval(g::GammaIntegrand, k) = GammaIntegrand(contract(g.HV, k), g.Mω, g.MΩ)

iterated_pre_eval(f::OCIntegrand, k) = OCIntegrand(contract(f.HV, k), f.Σ, f.Ω, f.β, f.μ)
iterated_pre_eval(f::OCIntegrand{<:AbstractFourierSeries3D}, k, dim) = (contract!(f.HV, k, dim-1); return f)

infer_f(::T, _) where {T<:Union{DOSIntegrand,GammaIntegrand,OCIntegrand,EquispaceOCIntegrand,AutoEquispaceOCIntegrand}} = (eltype(T), Base.promote_op(norm, eltype(T)))