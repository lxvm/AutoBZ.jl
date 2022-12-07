iterated_pre_eval(w::WannierIntegrand, x) = WannierIntegrand(w.f, contract(w.s, x), w.p)
iterated_pre_eval(w::WannierIntegrand{<:Any,<:AbstractFourierSeries3D}, x, dim) = (contract!(w.s, x, dim); return w)


iterated_pre_eval(D::DOSIntegrand, k) = DOSIntegrand(contract(D.H, k), D.M)
iterated_pre_eval(D::DOSIntegrand{<:AbstractFourierSeries3D}, k, dim) = (contract!(D.H, k, dim); return D)

iterated_pre_eval(Γ::TransportIntegrand, k, dim) = (contract!(Γ.HV, k, dim); return Γ)

# dim - 1 to accomodate the innermost frequency integral
iterated_pre_eval(A::KineticIntegrand, k, dim) = (contract!(A.HV, k, dim-1); return A)

infer_f(::T, _) where {T<:Union{DOSIntegrand,TransportIntegrand,KineticIntegrand,EquispaceKineticIntegrand,AutoEquispaceKineticIntegrand}} = (eltype(T), Base.promote_op(norm, eltype(T)))