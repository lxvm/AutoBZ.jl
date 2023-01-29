export AbstractFourierIntegrand
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
    AbstractFourierIntegrand{S<:AbstractFourierSeries}

Supertype representing Fourier integrands
"""
abstract type AbstractFourierIntegrand{S<:AbstractFourierSeries} end

# interface

function finner end # the function acting in the innermost integral
ftotal(f::AbstractFourierIntegrand) = f.f # the collection of functions
series(f::AbstractFourierIntegrand) = f.s # the Fourier series
params(f::AbstractFourierIntegrand) = f.p # collection of additional parameters

# abstract methods

@generated iterated_pre_eval(f::T, x) where {T<:AbstractFourierIntegrand} =
    :($(nameof(T))(ftotal(f), contract(series(f), x), params(f)))

(f::AbstractFourierIntegrand)(x) = finner(f)(series(f)(x), params(f)...)

equispace_integrand(f::AbstractFourierIntegrand, s_x) = finner(f)(s_x, params(f)...)

function equispace_rule(f::AbstractFourierIntegrand, bz::AbstractBZ, npt)
    rule = Vector{Tuple{fourier_type(series(f),domain_type(bz)),domain_type(bz)}}(undef, 0)
    equispace_rule!(rule, f, bz, npt)
end
equispace_rule!(rule, f::AbstractFourierIntegrand, bz::AbstractBZ, npt) =
    fourier_rule!(rule, series(f), bz, npt)

# implementations

"""
    FourierIntegrand(f, s::AbstractFourierSeries, ps...)

A type generically rerulesenting an integrand `f` whose entire dependence on the
variables of integration is in a Fourier series `s`, and which may also accept
some input parameters `ps`. The caller must know that their function, `f`, will
be evaluated at many points, `x`, in the following way: `f(s(x), ps...)`.
Therefore the caller is expected to know the type of `s(x)` (hint: `eltype(s)`)
and the layout of the parameters in the tuple `ps`. Additionally, `f` is assumed
to be type-stable and the inference can be queried by calling `eltype` on the
`FourierIntegrand`.
"""
struct FourierIntegrand{F,S,P<:Tuple} <: AbstractFourierIntegrand{S}
    f::F
    s::S
    p::P
end
FourierIntegrand{F}(s, ps...) where {F<:Function} = FourierIntegrand(F.instance, s, ps) # allows dispatch by aliases
FourierIntegrand(f, s, ps...) = FourierIntegrand(f, s, ps)

finner(f::FourierIntegrand) = ftotal(f)


"""
    IteratedFourierIntegrand(fs::Tuple, s::AbstractFourierSeries, ps...)

Integrand type similar to `FourierIntegrand`, but allowing nested integrand
functions `fs` with `fs[1]` the innermost function. Only the innermost integrand
is allowed to depend on parameters, but this could be implemented to allow the
inner function to also be multivariate Fourier series.

!!! note "Incompatibility with symmetries"
    In practice, it is only safe to use the output of this integrand when
    integrated over a domain with symmetries when the functions`fs` preserve the
    periodicity of the functions being integrated, such as linear functions.
    When used as an equispace integrand, this type does each integral one
    variable at a time applying PTR to each dimension. Note that 
"""
struct IteratedFourierIntegrand{F<:Tuple,S,P<:Tuple} <: AbstractFourierIntegrand{S}
    f::F
    s::S
    p::P
end
IteratedFourierIntegrand{F}(s, ps...) where {F<:Tuple{Vararg{Function}}} =
    IteratedFourierIntegrand(tuple(map(f -> f.instance, F.parameters)...), s, ps) # allows dispatch by aliases
IteratedFourierIntegrand(f, s, ps...) = IteratedFourierIntegrand(f, s, ps)

finner(f::IteratedFourierIntegrand) = f.f[1]

# iterated customizations

iterated_integrand(f::IteratedFourierIntegrand, x, ::Type{Val{1}}) = f(x)
iterated_integrand(f::IteratedFourierIntegrand, y, ::Type{Val{d}}) where d = f.f[d](y)
iterated_integrand(_::IteratedFourierIntegrand, y, ::Type{Val{0}}) = y

# equispace customizations

function equispace_evalrule(f::IteratedFourierIntegrand{F,S}, rule::Vector) where {F,N,S<:AbstractFourierSeries{N}}
    @warn "Do not trust an iterated integrand with equispace integration unless for linear integrands with full BZ"
    equispace_evalrule(f, rule)
end
@generated function equispace_evalrule(f::F, rule::AbstractArray{Tuple{T,W},N}) where {N,S<:AbstractFourierSeries{N},F<:IteratedFourierIntegrand{<:Any,S},T,W}
    I_N = Symbol(:I_, N)
    quote
        npt = 
        # infer return types of individual integrals
        T_1 = Base.promote_op(*, Base.promote_op(equispace_integrand, F, T), W)
        Base.Cartesian.@nexprs $(N-1) d -> T_{d+1} = Base.promote_op(equispace_integrand, F, T_d)
        # compute quadrature
        $I_N = zero($(Symbol(:T_, N)))
        Base.Cartesian.@nloops $N i rule (d -> d==1 ? nothing : I_{d-1} = zero(T_{d-1})) (d -> d==1 ? nothing : I_d += iterated_integrand(f, I_{d-1}, Val{d})) begin
            I_1 += equispace_integrand(f, rule[Base.Cartesian.@ncall $N equispace_index npt d -> i_d][1])
        end
        iterated_integrand(f, $I_N, Val{0})
    end
end