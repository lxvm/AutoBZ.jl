iterated_pre_eval(w::WannierIntegrand, x) = WannierIntegrand(w.f, contract(w.s, x), w.p)

iterated_pre_eval(g::GreensFunction, x) = GreensFunction(contract(g.H, x), g.M)
iterated_pre_eval(A::SpectralFunction, x) = SpectralFunction(iterated_pre_eval(A.G, x))
iterated_pre_eval(D::DOSIntegrand, x) = DOSIntegrand(iterated_pre_eval(D.A, x))

iterated_pre_eval(g::GammaIntegrand, k) = GammaIntegrand(contract(g.HV, k), g.Mω, g.MΩ)

iterated_pre_eval(f::OCIntegrand, k) = OCIntegrand(contract(f.HV, k), f.Σ, f.Ω, f.β, f.μ)
