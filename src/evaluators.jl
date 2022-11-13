export WannierEvaluator

struct WannierEvaluator{F,S<:AbstractFourierSeries,L<:IntegrationLimits,N} <: Function
    f::F
    s::S
    l::L
    order::Int
    atol::Float64
    rtol::Float64
    norm::N
end
function WannierEvaluator(f, s, l; order=4, atol=nothing, rtol=nothing, norm=norm)
    atol_ = something(atol, eltype(l))/nsyms(l)
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(l)))) : zero(eltype(l)))
    WannierEvaluator(f, s, l, order, atol_, rtol_, norm)
end 
(w::WannierEvaluator)(p) = first(iterated_integration(WannierIntegrand(w.f, w.s, p), w.l; order=w.order, atol=w.atol, rtol=w.rtol))


export DOSEvaluator, GammaEvaluator, OCEvaluator

struct DOSEvaluator{H<:AbstractFourierSeries,E,L,N,S}
    H::H
    Σ::E
    μ::Float64
    l::L
    order::Int
    atol::Float64
    rtol::Float64
    norm::N
    segbufs::S
end
function DOSEvaluator(H::TH, Σ, μ, l; order=4, atol=nothing, rtol=nothing, norm=norm) where TH
    Tfx = eltype(DOSIntegrand{TH}); Tnfx = Base.promote_op(norm, Tfx)
    segbufs = alloc_segbufs(eltype(l), Tfx, Tnfx, ndims(l))
    atol_ = something(atol, zero(Tnfx))/nsyms(l)
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(Tnfx))) : zero(Tnfx))
    DOSEvaluator(H, Σ, μ, l, order, atol_, rtol_, norm, segbufs)
end
(D::DOSEvaluator)(ω) = first(iterated_integration(DOSIntegrand(D.H, ω, D.Σ, D.μ), D.l; order=D.order, atol=D.atol, rtol=D.rtol, norm=D.norm, segbufs=D.segbufs))
