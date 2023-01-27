export ThunkIntegrand, FourierIntegrand, IteratedFourierIntegrand

"""
    ThunkIntegrand(f, x)

Store `f` and `x` to evaluate `f(x)` at a later time. Employed by
`iterated_integration` for generic integrands that haven't been specialized to
use `iterated_pre_eval`.
"""
struct ThunkIntegrand{T,d,X}
    f::T
    x::SVector{d,X}
end

(f::ThunkIntegrand)(x) = f.f(vcat(x, f.x))

"""
    thunk(f, x)

Delay the computation of f(x). Needed to normally evaluate an integrand in
nested integrals, a setting in which the values of the variables of integration
are passed one at a time. Importantly, `thunk` assumes that the variables of
integration are passed from the outermost to the innermost. For example, to
evaluate `f([1, 2])`, call `thunk(f, 2)(1)`.

This behavior is consistent with `CubicLimits`, but may come as a surprise if
implementing new `IntegrationLimits`.
"""
thunk(f, x) = ThunkIntegrand(f, SVector(x))
thunk(f::ThunkIntegrand, x) = ThunkIntegrand(f.f, vcat(x, f.x))

iterated_pre_eval(f, x) = thunk(f, x)


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
struct FourierIntegrand{F,S<:AbstractFourierSeries,P<:Tuple}
    f::F
    s::S
    p::P
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
equispace_pre_eval(f::FourierIntegrand, l, npt) = fourier_pre_eval(f.s, l, npt)



"""
Integrand type similar to `FourierIntegrand`, but allowing nested integrand
functions. Only the innermost integrand is allowed to depend on parameters, but
this could be changed.
"""
struct IteratedFourierIntegrand{F<:Tuple,S<:AbstractFourierSeries,P<:Tuple}
    f::F
    s::S
    p::P
    function IteratedFourierIntegrand(f::F, s::S, p::P) where {F,S,P<:Tuple}
        @assert length(f) == ndims(s) "need same number of integrands as variables of integration"
        new{F,S,P}(f, s, p)
    end
end

IteratedFourierIntegrand(f, s, ps...) = IteratedFourierIntegrand(f, s, ps)

function iterated_pre_eval(f::IteratedFourierIntegrand, x)
    FourierIntegrand(f.f, contract(f.s, x), f.p)
end

iterated_integrand(f::IteratedFourierIntegrand, x, ::Type{Val{1}}) = f(x)
iterated_integrand(f::IteratedFourierIntegrand, y, ::Type{Val{d}}) where d = f.f[d](y)

(f::IteratedFourierIntegrand)(x::SVector{N}) where N = evaluate_integrand(f, f.s(x), Val{N}())
(f::IteratedFourierIntegrand)(x::Number) = evaluate_integrand(f, f.s(x), Val{1}()) # innermost integral needs this interface
evaluate_integrand(f::IteratedFourierIntegrand, s_x, ::Val{1}) = f.f[1](s_x, f.p...)
evaluate_integrand(f::IteratedFourierIntegrand, y, ::Val{dim}) where dim = f.f[dim](evaluate_integrand(f, y, Val{dim-1}())) # f may depend on other parameters?
evaluate_integrand(f::IteratedFourierIntegrand, s_x, ::Colon) = ∘(reverse(f.f)...)(s_x, f.p...)
Base.eltype(::Type{T}) where {F,S,T<:IteratedFourierIntegrand{F,S}} = Base.promote_op(evaluate_integrand, T, eltype(S), Colon)