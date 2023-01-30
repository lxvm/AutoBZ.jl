"""
    AbstractFourierIntegrand{S<:AbstractFourierSeries} <: AbstractIteratedIntegrand

Supertype representing Fourier integrands
"""
abstract type AbstractFourierIntegrand{d,S<:AbstractFourierSeries} <: AbstractIteratedIntegrand{d} end

# interface

function finner end # the function acting in the innermost integral
ftotal(f::AbstractFourierIntegrand) = f.f # the collection of functions
series(f::AbstractFourierIntegrand) = f.s # the Fourier series
params(f::AbstractFourierIntegrand) = f.p # collection of additional parameters

# abstract methods

iterated_integrand(_::AbstractFourierIntegrand, y, ::Type{Val{d}}) where d = y
iterated_integrand(f::AbstractFourierIntegrand, x, ::Type{Val{1}}) =
    finner(f)(series(f)(x), params(f)...)

@generated iterated_pre_eval(f::T, x, dim::Int) where {d,T<:AbstractFourierIntegrand{d}} =
    :($(nameof(T)){$d}(ftotal(f), contract(series(f), x, dim), params(f)))
iterated_value(f::AbstractFourierIntegrand) = series(f)

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

A type generically representing an integrand `f` whose entire dependence on the
variables of integration is in a Fourier series `s`, and which may also accept
some input parameters `ps`. The caller must know that their function, `f`, will
be evaluated at many points, `x`, in the following way: `f(s(x), ps...)`.
Therefore the caller is expected to know the type of `s(x)` (hint: `eltype(s)`)
and the layout of the parameters in the tuple `ps`. Additionally, `f` is assumed
to be type-stable, and is compatible with the equispace integration routines.
"""
struct FourierIntegrand{F,d,S,P} <: AbstractFourierIntegrand{d,S}
    f::F
    s::S
    p::P
    FourierIntegrand{d}(f::F, s::S, p::P) where {d,F,S<:AbstractFourierSeries,P<:Tuple} =
        new{F,d,S,P}(f,s,p)
end
FourierIntegrand{F}(s::AbstractFourierSeries{N}, ps) where {F<:Function,N} = FourierIntegrand{N}(F.instance, s, ps) # allows dispatch by aliases
FourierIntegrand{d}(f, s, ps...) where d = FourierIntegrand{d}(f, s, ps)

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
struct IteratedFourierIntegrand{F,d,S,P} <: AbstractFourierIntegrand{d,S}
    f::F
    s::S
    p::P
    IteratedFourierIntegrand{d}(f::F, s::S, p::P) where {d,F<:Tuple,S<:AbstractFourierSeries,P<:Tuple} =
        new{F,d,S,P}(f,s,p)
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
    equispace_evalrule_(f, rule)
end
@generated function equispace_evalrule_(f::F, rule::Vector{Tuple{T,W}}) where {N,S<:AbstractFourierSeries{N},FF,F<:IteratedFourierIntegrand{FF,S},T,W}
    I_N = Symbol(:I_, N)
    quote
        npt = round(Int, length(rule)^(1/$N)) # implicitly assuming FBZ
        # infer return types of individual integrals
        T_1 = Base.promote_op(*, Base.promote_op(equispace_integrand, F, T), W)
        Base.Cartesian.@nexprs $(N-1) d -> T_{d+1} = Base.promote_op(iterated_integrand, F, T_d, Type{Val{d+1}})
        # compute quadrature
        $I_N = zero($(Symbol(:T_, N)))
        Base.Cartesian.@nloops $N i rule (d -> d==1 ? nothing : I_{d-1} = zero(T_{d-1})) (d -> d==1 ? nothing : I_d += iterated_integrand(f, I_{d-1}, Val{d})) begin
            idx = Base.Cartesian.@ncall $N equispace_index npt d -> i_d
            I_1 += rule[idx][2]*equispace_integrand(f, rule[idx][1]) # implicitly assuming outer functions are linear
        end
        iterated_integrand(f, $I_N, Val{0})
    end
end