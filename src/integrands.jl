export FourierIntegrand

"""
    FourierIntegrand(f, s::AbstractFourierSeries, ps...)

A type generically representing an integrand `f` whose entire dependence on the
variables of integration is in a Fourier series `s`, and which may also accept
some input parameters `ps`. The caller must know that their function, `f`, will
be evaluated at many points, `x`, in the following way: `f(s(x), ps...)`.
Therefore the caller is expected to know the type of `s(x)` (hint: `eltype(s)`)
and the layout of the parameters in the tuple `ps`. Additionally, `f` is assumed
to be type-stable and the inference can be queried by calling `eltype` on the
`FourierIntegrand`.
"""
struct FourierIntegrand{TF,TS<:AbstractFourierSeries,TP<:Tuple}
    f::TF
    s::TS
    p::TP
end
FourierIntegrand{F}(s, ps...) where {F<:Function} = FourierIntegrand(F.instance, s, ps) # allows dispatch by aliases
FourierIntegrand(f, s, ps...) = FourierIntegrand(f, s, ps)
(f::FourierIntegrand)(x) = evaluate_integrand(f, f.s(x))
evaluate_integrand(f::FourierIntegrand, s_x) = f.f(s_x, f.p...)
Base.eltype(::Type{FourierIntegrand{F,S,P}}) where {F,S,P} = Base.promote_op(F.instance, eltype(S), P.parameters...)

iterated_pre_eval(f::FourierIntegrand, x) = FourierIntegrand(f.f, contract(f.s, x), f.p)

"""
    equispace_pre_eval(f::FourierIntegrand, l::IntegrationLimits, npt)

This function will evaluate the Fourier series and integration weights needed
for equispace integration of `f` at `npt` points per dimension. `l` should
contain the relevant symmetries needed for IBZ integration, if desired.
"""
equispace_pre_eval(f::FourierIntegrand, l, npt) = pre_eval_contract(f.s, l, npt)
