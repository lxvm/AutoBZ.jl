"""
    AbstractIteratedIntegrand{d}

Supertype for integrands compatible with iterated integration of `d` variables.
"""
abstract type AbstractIteratedIntegrand{d} end

# interface

function iterated_pre_eval end # passes in the current variable of integration to the inner integrals as a parameter
function iterated_integrand end # evaluates the integrand at the current variable of integration

# abstract methods

nvars(::AbstractIteratedIntegrand{d}) where d = d
nvars(_) = -1 # defined for functions to escape the dimensions 0 to d

(f::AbstractIteratedIntegrand)(x) = iterated_integrand(f, x, Val{1})

"""
    iterated_value(f, Val{d}) where d

When `d=1`, calls `iterated_value(f)` to yield a pre-evaluated item from f to
pass to an inner integral. This should be specialized for `f`
"""
iterated_value(f::AbstractIteratedIntegrand, ::Type{Val{1}}) = iterated_value(f)
iterated_value(f, _) = f # null op for dim>1
iterated_value(_, ::Type{Val{1}}) = missing # can't specialize on arbitrary functions
iterated_value(::AbstractIteratedIntegrand) = missing # fallback

"""
    iterated_integrand(f::AbstractIteratedIntegrand, y, ::Type{Val{d}}) where d
    iterated_integrand(f::AbstractIteratedIntegrand, y, dim::Int) # fallback

By default, returns `y` which is the result of an interior integral.
Can use dispatch on `f` to implement more kinds of iterated integrals, in which
case the developer should know `d` takes values of `0, 1, ..., ndims(lims)`.
`d=1` evaluates the innermost integral, `d=0` evaluates a function outside the
last integral. Subtypes of `AbstractFourierIntegrand` 
"""
@inline iterated_integrand(f::AbstractIteratedIntegrand, y, ::Type{Val{d}}) where d = iterated_integrand(f, y, d)
@inline iterated_integrand(f, x, ::Type{Val{-1}}) = f(x)

"""
    iterated_pre_eval(f, x, dim)
    iterated_pre_eval(f, x) # fallback

Perform a precomputation on `f` using the value of a variable of integration,
`x`. The default is to store `x` and delay the computation of `f(x)` until all
of the values of the variables of integration are determined at a integration
point. Certain types of functions, such as Fourier series, take can use `x` to
precompute a new integrand for the remaining variables of integration that is
more computationally efficient. This function must return the integrand for the
subsequent integral.
"""
@inline iterated_pre_eval(f::AbstractIteratedIntegrand, x, ::Type{Val{d}}) where d = iterated_pre_eval(f, x, d)
iterated_pre_eval(f::AbstractIteratedIntegrand, x, dim) = iterated_pre_eval(f, x)
iterated_pre_eval(f, x, dim) = f # when f is anything else, leave it


# implementations

"""
    ThunkIntegrand{d}(f)

Store `f` and `x` to evaluate `f(x...)` at a later time. Employed by
`iterated_integration` for generic integrands that haven't been specialized to
use `iterated_pre_eval`. Note that `x isa Tuple`, so the function arguments
needs to expect the behavior. `d` is a parameter specifying the number of
variables in the vector input of `f`. This is good for integrands like `∫∫∫f`.
"""
struct ThunkIntegrand{d,F,X} <: AbstractIteratedIntegrand{d}
    f::F
    x::X
    ThunkIntegrand{d}(f::F, x::X) where {d,F,N,T,X<:NTuple{N,T}} =
        new{d,F,X}(f, x) 
end
ThunkIntegrand{d}(f) where {d,T} = ThunkIntegrand{d}(f, ())

iterated_integrand(f::ThunkIntegrand, x, ::Type{Val{1}}) = f.f(x, f.x...)
iterated_integrand(_::ThunkIntegrand, x, ::Type{Val{d}}) where d = x
iterated_pre_eval(f::ThunkIntegrand{d}, x) where d =
    ThunkIntegrand{d}(f.f, (x, f.x...))


# experimental and untested symbolic language for IAI integrals like  ∫(I1 * ∫(I2 + I3), I4)

"""
    AssociativeOpIntegrand(op, I::AbstractIteratedIntegrand...)

!!! warning "Experimental"
    This may not work or may change

Constructor for a collection of integrands reduced by an associative `op`.
"""
struct AssociativeOpIntegrand{L,O,T} <: AbstractIteratedIntegrand{1}
    op::O
    terms::T
    AssociativeOpIntegrand(op::O,terms::T) where {L,O,T<:Tuple{Vararg{Any,L}}} =
        new{L,O,T}(op, terms)
end
AssociativeOpIntegrand(op, I...) = AssociativeOpIntegrand(op, I)

iterated_integrand(f::AssociativeOpIntegrand{L}, x, ::Type{Val{1}}) where L =
    reduce(f.op, ntuple(n -> iterated_integrand(f.terms[n], x, Val{nvars(f.terms[n])}), Val{L}()))
iterated_pre_eval(f::AssociativeOpIntegrand{L}, x, ::Type{Val{1}}) where L =
    AssociativeOpIntegrand(f.op, ntuple(n -> iterated_pre_eval(f.terms[n], x, Val{nvars(f.terms[n])}), Val{L}()))


"""
    IteratedIntegrand(fs...; f0=identity)

!!! warning "Experimental"
    This may not work or may change

Represents a nested integral of the form
`f0(∫dxN fN(xN, ... ∫dx2 f2(x2, ..., xN, ∫dx1 f1(x1, ..., xN)) ... )))`
so the functions need to know the arguments and their layout, since the
variables and inner integrals are passed as vectors.
"""
struct IteratedIntegrand{d,F0,F,X,L} <: AbstractIteratedIntegrand{d}
    f0::F0
    f::F
    x::X
    levels::L
    IteratedIntegrand{d}(f0::F0, f::F, x::X, levels::L) where {d,F0,F<:Tuple,N,T,X<:SVector{N,T},L<:Tuple{Vararg{Int}}} =
        new{d,F0,F,X,L}(f0, f, x, levels)
end
IteratedIntegrand(fs::AbstractIteratedIntegrand...; f0=identity) =
    IteratedIntegrand{sum(nvars, fs; init=0)}(f0, fs, SVector{0}(), cumsum(map(nvars, fs)))

iterated_integrand(f::IteratedIntegrand, y, ::Type{Val{0}}) = f.f0(y)
function iterated_integrand(f::IteratedIntegrand{d}, y, ::Type{Val{dim}}) where {d,dim}
    i = findfirst(<=(dim), levels); j = d-levels[i]+levels[1] # number of variables below current one
    iterated_integrand(f.f[i], (f.x, skipmissing(f.f)...), y, Val{j}) # how to choose which variables to evaluate?
end
# since IteratedIntegrand will return a tuple of the coordinate and inner
# integrand make the default behavior to pass the inner integrand
# e.g. ∫dx ∫dy f(x,y) -> (x,Ix) -> 
@inline iterated_integrand(f::AbstractIteratedIntegrand, _, y, ::Type{Val{d}}) where d =
    iterated_integrand(f, y, d)
# for a bare function, evaluate f(coordinate, inner_integral)
# e.g ∫dx exp(x + ∫dy_0^x sin(xy)) becomes (x, Ix) -> exp(x + Ix) 
@inline iterated_integrand(f, x, y, dim) = f(x, y)


function iterated_pre_eval(f::IteratedIntegrand{d}, x, ::Type{Val{dim}}) where {d,dim}
    i = findfirst(<=(dim), levels); j = d-levels[i]+levels[1] # number of variables below current one
    fs = setindex(f.f, iterated_value(iterated_pre_eval(f.f[i], x, Val{j}), Val{j}), i) #
    x = vcat(x, f.x) # keep the outer variables and only modify the current one
    IteratedIntegrand{d}(f.f0, fs, x, f.levels)
end

const ∫ = IteratedIntegrand