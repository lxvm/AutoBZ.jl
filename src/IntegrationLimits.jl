export IntegrationLimits, lower, upper, box, vol, nsyms, symmetrize, symmetries,
    CubicLimits, CompositeLimits

"""
    IntegrationLimits{d}

Represents a set of integration limits over `d` variables.
Realizations of this type should implement `lower` and `upper`, which return the
lower and upper limits of integration along some dimension, `rescale` which
represents the number of symmetries of the BZ which are used by the realization
to reduce the BZ (the integrand over the limits gets multiplied by this factor),
and a functor that accepts a single numeric argument and returns another
realization of that type (in order to do nested integration). Thus the
realization is also in control of the order of variables of integration and must
coordinate this behavior with their integrand.
Instances should also be static structs.
"""
abstract type IntegrationLimits{d} end

# Interface for IntegrationLimits types

"""
    box(::IntegrationLimits)

Return an iterator of tuples that for each dimension returns a tuple with the
lower and upper limits of the integration domain without symmetries applied.
"""
function box end

"""
    lower(::IntegrationLimits)

Return the lower limit of the next variable of integration. If a vector is
returned, then the integration routine may attempt to multidimensional
integration.
"""
function lower end

"""
    upper(::IntegrationLimits)

Return the upper limit of the next variable of integration. If a vector is
returned, then the integration routine may attempt to multidimensional
integration.
"""
function upper end

"""
    nsyms(::IntegrationLimits)

Return the number of symmetries that the parametrization has used to reduce the
volume of the integration domain.
"""
function nsyms end

"""
    symmetries(::IntegrationLimits)

Return an iterator over the symmetry transformations that the parametrization
has used to reduce the volume of the integration domain.
"""
function symmetries end

"""
    discretize_equispace(::IntegrationLimits, ::Integer)

Return an iterator of 2-tuples containing integration nodes and weights that
correspond to an equispace integration grid with the symmetry transformations
applied to it.
"""
function discretize_equispace end

# Generic methods for IntegrationLimits

"""
    (::IntegrationLimits)(x::SVector)

Compute the outermost `length(x)` variables of integration.
Realizations of type `T<:IntegrationLimits` only have to implement a method with
signature `(::T)(::Number)`.
"""
(l::IntegrationLimits)(x::SVector) = l(last(x))(pop(x))
(l::IntegrationLimits)(::SVector{0}) = l

"""
    symmetrize(::IntegrationLimits, x)
    symmetrize(::IntegrationLimits, xs...)

Transform `x` by the symmetries of the parametrization used to reduce the
domain, thus mapping the value of `x` on the parametrization to the full domain.
When the integrand is a scalar, this is equal to `nsyms(l)*x`.
When the integrand is a vector, this is `sum(S*x for S in symmetries(l))`.
When the integrand is a matrix, this is `sum(S*x*S' for S in symmetries(l))`.
"""
symmetrize(l::IntegrationLimits, xs...) = map(x -> symmetrize(l, x), xs)
symmetrize(l::IntegrationLimits, x) = symmetrize_(x, nsyms(l), symmetries(l))
symmetrize_(x::Number, nsym, syms) = nsym*x
symmetrize_(x::AbstractArray{<:Any,0}, nsym, syms) = symmetrize_(only(x), nsym, syms)
function symmetrize_(x::AbstractVector, nsym, syms)
    r = zero(x)
    for S in syms
        r += S * x
    end
    r
end
function symmetrize_(x::AbstractMatrix, nsym, syms)
    r = zero(x)
    for S in syms
        r += S * x * S'
    end
    r
end

"""
    vol(::IntegrationLimits)

Return the volume of the full domain without the symmetries applied
"""
vol(l::IntegrationLimits) = prod(u-l for (l, u) in box(l))

"""
    ndims(::IntegrationLimits{d})

Returns `d`. This is a type-based rule.
"""
Base.ndims(::T) where {T<:IntegrationLimits} = ndims(T)
Base.ndims(::Type{<:IntegrationLimits{d}}) where {d} = d

nsyms(l::IntegrationLimits) = length(collect(symmetries(l)))

function discretize_equispace(l, npt)
    flag, wsym, nsym = discretize_equispace_(l, npt)
    T = SVector{ndims(l),Float64}
    out = Vector{Tuple{T,Int}}(undef, nsym)
    ps = box(l)
    n = 0
    for i in CartesianIndices(flag)
        if flag[i]
            n += 1
            out[n] = (StaticArrays.sacollect(T, (p[2]-p[1])*(j-1)/npt + p[1] for (j, p) in zip(Tuple(i), ps)), wsym[n])
            n >= nsym && break
        end
    end
    return out
end

@generated function discretize_equispace_(l::T, npt) where {d, T<:IntegrationLimits{d}}
    quote
    xsym = Matrix{Float64}(undef, $d, nsyms(l))
    syms = collect(symmetries(l))
    x = range(-1.0, step=2inv(npt), length=npt)
    flag = ones(Bool, Base.Cartesian.@ntuple $d _ -> npt)
    nsym = 0
    wsym = Vector{Int}(undef, npt^$d)
    Base.Cartesian.@nloops $d i _ -> Base.OneTo(npt) begin
        (Base.Cartesian.@nref $d flag i) || continue
        for (j, S) in enumerate(syms)
            xsym[:, j] = S * (Base.Cartesian.@ncall $d SVector{$d,Float64} k -> x[i_k])
        end
        nsym += 1
        wsym[nsym] = 1
        for j in 2:nsyms(l)
            Base.Cartesian.@nexprs $d k -> begin
                ii_k = 0.5npt * (xsym[k, j] + 1.0) + 1.0
                iii_k = round(Int, ii_k)
                (iii_k - ii_k) > 1e-12 && throw("Inexact index")
            end
            (Base.Cartesian.@nany $d k -> (iii_k > npt)) && continue
            (Base.Cartesian.@nall $d k -> (iii_k == i_k)) && continue
            if (Base.Cartesian.@nref $d flag iii)
                (Base.Cartesian.@nref $d flag iii) = false
                wsym[nsym] += 1
            end
        end
    end
    return flag, wsym, nsym
    end
end

# Implementations of IntegrationLimits

"""
    CubicLimits(a, [b])

Store integration limit information for a hypercube with vertices `a` and `b`.
If `b` is not passed as an argument, then the lower limit defaults to `zero(a)`.
"""
struct CubicLimits{d,T<:Real} <: IntegrationLimits{d}
    l::SVector{d,T}
    u::SVector{d,T}
end
function CubicLimits(l::SVector{d,Tl}, u::SVector{d,Tu}) where {d,Tl<:Real,Tu<:Real}
    T = float(promote_type(Tl, Tu))
    CubicLimits(SVector{d,T}(l), SVector{d,T}(u))
end
CubicLimits(l::Tl, u::Tu) where {Tl<:Real,Tu<:Real} = CubicLimits(SVector{1,Tl}(l), SVector{1,Tu}(u))
CubicLimits(u) = CubicLimits(zero(u), u)

box(c::CubicLimits{d,T}) where {d,T} = StaticArrays.sacollect(SVector{d,Tuple{T,T}}, zip(c.l, c.u))

"""
    lower(::CubicLimits)

Returns the lower limit of the outermost variable of integration.
"""
lower(c::CubicLimits) = last(c.l)

"""
    upper(::CubicLimits)

Returns the upper limit of the outermost variable of integration.
"""
upper(c::CubicLimits) = last(c.u)

"""
    nsyms(::CubicLimits)

Returns 1 because the only symmetry applied to the cube is the identity.
"""
nsyms(::CubicLimits) = 1

"""
    symmetries(::CubicLimits)

Return an identity matrix.
"""
symmetries(::CubicLimits) = tuple(I)

"""
    (::CubicLimits)(x::Number)

Return a CubicLimits of lower dimension with the outermost variable of
integration removed.
"""
(c::CubicLimits)(::Number) = CubicLimits(pop(c.l), pop(c.u))

function discretize_equispace(c::CubicLimits{d,T}, npt) where {d,T}
    ((SVector{d,T}(x...), true) for x in Iterators.product([range(l, step=(u-l)/npt, length=npt) for (l,u) in box(c)]...))
end

"""
    CompositeLimits(::Tuple{Vararg{IntegrationLimits}})

Construct a collection of limits which yields the first limit followed by the
second, and so on.
"""
struct CompositeLimits{T<:Tuple{Vararg{IntegrationLimits}},d} <: IntegrationLimits{d}
    lims::T
    function CompositeLimits(lims::T) where {T<:Tuple{Vararg{IntegrationLimits}}}
        new{T, mapreduce(ndims, +, T.parameters; init=0)}(lims)
    end
end
CompositeLimits(lims::IntegrationLimits...) = CompositeLimits(lims)
(l::CompositeLimits)(x::Number) = CompositeLimits(first(l.lims)(x), Base.rest(l.lims, 2)...)
(l::CompositeLimits{T})(x::Number) where {T<:Tuple{<:IntegrationLimits}} = CompositeLimits(first(l.lims)(x))
(l::CompositeLimits{T})(::Number) where {T<:Tuple{<:IntegrationLimits{1},Vararg{IntegrationLimits}}} = CompositeLimits(Base.rest(l.lims, 2)...)

box(l::CompositeLimits) = Iterators.flatten(reverse(map(box, l.lims)))
lower(l::CompositeLimits) = lower(first(l.lims))
upper(l::CompositeLimits) = upper(first(l.lims))
nsyms(l::CompositeLimits) = prod(nsyms, l.lims)
function symmetrize(l::CompositeLimits, x)
    r = x
    for lim in l.lims
        r = symmetrize(lim, r)
    end
    r
end

# untested
function discretize_equispace(l::CompositeLimits, npt)
    ((vcat([map(y -> y[1], x)]...), prod(y -> y[2], x)) for x in Iterators.product(reverse(map(m -> discretize_equispace(m, npt), l.lims)))...)
end