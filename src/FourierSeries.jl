# Possible TODO for FourierSeries
# - replace dependence on SVector with NTuple for ease of use by new users
# - enable reduction over multiple dims simulatneously (?) may not be used
export AbstractFourierSeries, period, contract, value
export FourierSeries, FourierSeriesDerivative, OffsetFourierSeries, ManyFourierSeries, ManyOffsetsFourierSeries

"""
    AbstractFourierSeries{N}

A supertype for Fourier series that are periodic maps ``\\R^N \\to V`` where
``V`` is any vector space. Typically these can be represented by `N`-dimensional
arrays whose elements belong to the vector space. See the manual section on the
`AbstractFourierSeries` interface.
"""
abstract type AbstractFourierSeries{N} end

"""
    period(f::AbstractFourierSeries{N}) where {N}

Return a `NTuple{N}` whose `m`-th element corresponds to the period of `f`
along its `m`-th input dimension. Typically, these values set the units of
length for the problem.
"""
function period end

"""
    contract(f::AbstractFourierSeries{N}, x::Number, [dim=N]) where {N}

Return another Fourier series of dimension `N-1` by summing over dimension `dim`
of `f` with the phase factors evaluated at `x`. If `N=1`, this function should
return an `AbstractFourierSeries{0}` that stores the evaluated Fourier series,
but has no more input dimensions to contract.

The default of `dim=N` is motivated by preserving memory locality in Julia's
column-major array format.

    contract(f::AbstractFourierSeries{N}, x::SVector{M}) where {N,M}

Contract the outermost indices `M` of `f` in order of `last(x)` to `first(x)`.
If `M>N`, the default behavior is just to try and contract `M` indices, which
will likely lead to an error.
"""
function contract end
contract(f::AbstractFourierSeries, x::SVector) = contract(contract(f, last(x)), pop(x))
contract(f::AbstractFourierSeries, ::SVector{0}) = f

"""
value(::AbstractFourierSeries{0})

Return the evaluated Fourier series whose indices have all been contracted.
Typically, this value has the same units as the Fourier series coefficients.
"""
function value end

"""
    (f::AbstractFourierSeries{N})(x::SVector{N}) where {N}
    (f::AbstractFourierSeries{1})(x::Number)

Evaluate the Fourier series at the given point, which must have the same input
dimension as the Fourier series
"""
(f::AbstractFourierSeries{N})(x::SVector{N}) where {N} = value(contract(f, x))
(f::AbstractFourierSeries{1})(x::Number) = value(contract(f, x))

"""
    FourierSeries(coeffs, period::SVector{N,Float64}) where {N}

Construct a Fourier series whose coefficients are given by the coefficient array
array `coeffs` whose `eltype` should support addition and scalar multiplication,
and whose periodicity on the `i`th axis is given by `period[i]`. This type
represents the Fourier series
```math
f(\\vec{x}) = \\sum_{\\vec{n} \\in \\mathcal I} C_{\\vec{n}} \\exp(i2\\pi\\vec{k}_{\\vec{n}}\\cdot\\overrightarrow{x})
```
where ``i = \\sqrt{-1}`` is the imaginary unit, ``C`` is the array `coeffs`,
``\\mathcal I`` is `CartesianIndices(C)`, ``\\vec{n}`` is a `CartesianIndex` and
``\\vec{k}_{\\vec{n}}`` is equal to ``n_j/p_j`` in the
``j``th position with ``p_j`` the ``j``th element of `period`.
Because of the choice to use Cartesian indices to set the phase factors,
typically the indices of `coeffs` should be specified by using an `OffsetArray`.

    FourierSeries(coeffs::AbstractArray{T,N}, period::Real) where {T,N}

If period is a `Real`, this constructor will infer the number of
input dimensions of the Fourier series from the array dimensionality of the
coefficients, and `period` will become the period of all of the dimensions.
"""
struct FourierSeries{N,T} <: AbstractFourierSeries{N}
    coeffs::T
    period::SVector{N,Float64}
end

FourierSeries(coeffs::AbstractArray{T,N}, period::Real) where {T,N} = FourierSeries(coeffs, fill(period, SVector{N,Float64}))
Base.eltype(::Type{<:FourierSeries{N,T}}) where {N,T} = Base.promote_op(*, ComplexF64, eltype(T))
Base.eltype(::Type{<:FourierSeries{0,T}}) where {T} = T

"""
    contract(f::FourierSeries{N}, x::Number, [dim=N]) where N

Contract index `dim` of the coefficients of `f` at the spatial point `x`.
The default `dim` is the outermost dimension to preserve memory locality.
"""
function contract(f::FourierSeries{N}, x::Number, dim::Int) where N
    dim == N && return contract(f, x)
    1 <= dim <= N || error("Choose dim=$dim in 1:$N")
    C = f.coeffs
    @inbounds ϕ = x/f.period[dim]
    C′ = mapreduce(+, CartesianIndices(C); dims=dim) do i
        @inbounds C[i]*cispi(2*i.I[dim]*ϕ)
    end
    FourierSeries(dropdims(C′; dims=dim), deleteat(f.period, dim))
end
# Fourier series evaluation by recurrence is faster, but we only implement
# contraction of the outermost dimension
function contract(f::FourierSeries{1}, x::Number)
    C = contract_(f.coeffs, x / last(f.period))
    FourierSeries(C, pop(f.period)) # for consistency with N>1 edit C -> fill(C)
    # however, allocating a 0-dimensional array slows things down so I have
    # treat FourierSeries{0} as a special case since I store the value in coeffs
end
function contract_(C::AbstractVector, ϕ::Number)
    -2first(axes(C,1))+1 == size(C,1) || throw("array indices are not of form -n:n")
    @inbounds r = C[0]
    if size(C,1) > 1
        z₀ = cispi(2ϕ)
        z = one(z₀)
        @inbounds for n in Base.OneTo(last(axes(C,1)))
            z *= z₀
            r += z*C[n] + conj(z)*C[-n]
        end
    end
    r
end
# N-D implementation of recurrence
@generated function contract(f::FourierSeries{N}, x::Number) where N
quote
    C_ = f.coeffs
    C = (Base.Cartesian.@nref $N C_ i -> i == $N ? 0 : :)
    contract!(C, f.coeffs, x / last(f.period))
    FourierSeries(C, pop(f.period))
end
end
@generated function contract!(r, C::AbstractArray{<:Any,N}, ϕ::Number) where N
quote
    -2first(axes(C,$N))+1 == size(C,$N) || throw("array indices are not of form -n:n")
    @inbounds Base.Cartesian.@nloops $(N-1) i r begin
        (Base.Cartesian.@nref $(N-1) r i) = (Base.Cartesian.@nref $N C d -> d == $N ? 0 : i_d)
    end
    if size(C,$N) > 1
        z₀ = cispi(2ϕ)
        z = one(z₀)
        for n in Base.OneTo(last(axes(C,$N)))
            z *= z₀
            z̄ = conj(z)
            @inbounds Base.Cartesian.@nloops $(N-1) i r begin
                (Base.Cartesian.@nref $(N-1) r i) += z*(Base.Cartesian.@nref $N C d -> d == $N ? n : i_d) + z̄*(Base.Cartesian.@nref $N C d -> d == $N ? -n : i_d)
            end
        end
    end
end
end


function (f::FourierSeries{N})(x::SVector{N}) where {N}
    C = f.coeffs
    ϕ = x ./ f.period
    sum(CartesianIndices(C), init=zero(eltype(f))) do i
        C[i] * cispi(2*dot(ϕ, convert(SVector, i)))
    end
end
(f::FourierSeries{1})(x::SVector{1}) = f(only(x))
(f::FourierSeries{1})(x::Number) = contract_(f.coeffs, x/only(f.period))

value(f::FourierSeries{0}) = f.coeffs
period(f::FourierSeries) = f.period

"""
    FourierSeriesDerivative(f::FourierSeries{N}, a::SVector{N}) where {N}

Represent the differential of Fourier series `f` by a multi-index `a` of
derivatives, e.g. `[1,2,...]`, whose `i`th entry represents the order of
differentiation on the `i`th input dimension of `f`. Mathematically, this means
```math
\\left( \\prod_{j=1}^N \\partial_{x_j}^{a_j} \\right) f(\\vec{x}) = \\sum_{\\vec{n} \\in \\mathcal I} \\left( \\prod_{j=1}^N (i 2\\pi k_j)^{a_j} \\right) C_{\\vec{n}} \\exp(i2\\pi\\vec{k}_{\\vec{n}}\\cdot\\overrightarrow{x})
```
where ``\\partial_{x_j}^{a_j}`` represents the ``a_j``th derivative of ``x_j``,
``i = \\sqrt{-1}`` is the imaginary unit, ``C`` is the array `coeffs`,
``\\mathcal I`` is `CartesianIndices(C)`, ``\\vec{n}`` is a `CartesianIndex` and
``\\vec{k}_{\\vec{n}}`` is equal to ``n_j/p_j`` in the ``j``th position with
``p_j`` the ``j``th element of `period`. Because of the choice to use Cartesian
indices to set the phase factors, typically the indices of `coeffs` should be
specified by using an `OffsetArray`. Also, note that termwise differentiation of
the Fourier series results in additional factors of ``i2\\pi`` which should be
anticipated for the use case. Also, note that this type can be used to represent
fractional differentiation or integration by suitably choosing the ``a_j``s.

This is a 'lazy' representation of the derivative because instead of
differentiating by computing all of the Fourier coefficients of the derivative
upon constructing the object, the evaluator waits until it contracts the
differentiated dimension to evaluate the new coefficients.
"""
struct FourierSeriesDerivative{N,T<:FourierSeries{N},Ta} <: AbstractFourierSeries{N}
    f::T
    a::SVector{N,Ta}
end

Base.eltype(::Type{<:FourierSeriesDerivative{N,T}}) where {N,T} = eltype(T)

"""
    contract(f::FourierSeriesDerivative{N}, x::Number, [dim=N]) where {N}

Contract index `dim` of the coefficients of `f` at the spatial point `x`.
The default `dim` is the outermost dimension to preserve memory locality.
"""
function contract(dv::FourierSeriesDerivative{N}, x::Number, dim::Int) where N
    dim == N && return contract(dv, x)
    1 <= dim <= N || error("Choose dim=$dim in 1:$N")
    C = dv.f.coeffs
    @inbounds imk = im*2*pi/dv.f.period[dim]
    @inbounds ϕ = x/dv.f.period[dim]
    @inbounds a = dv.a[dim]
    C′ = mapreduce(+, CartesianIndices(C); dims=dim) do i
        @inbounds C[i]*(cispi(2*i.I[dim]*ϕ)*((imk*i.I[dim])^a))
    end
    f = FourierSeries(dropdims(C′; dims=dim), deleteat(dv.f.period, dim))
    FourierSeriesDerivative(f, deleteat(dv.a, dim))
end
# 1D recurrence is still faster
function contract(dv::FourierSeriesDerivative{1}, x::Number)
    C = contract_(dv.f.coeffs, x, last(dv.f.period), last(dv.a))
    f = FourierSeries(C, pop(dv.f.period)) # for consistency with higher dimensions, C -> fill(C)
    FourierSeriesDerivative(f, pop(dv.a))
end
function contract_(C::AbstractVector, x, a₀, a)
    -2first(axes(C,1))+1 == size(C,1) || throw("array indices are not of form -n:n")
    @inbounds r = (0^a)*C[0]
    if size(C,1) > 1
        imk = im*2*pi/a₀
        z₀ = cispi(2*x/a₀)
        z = one(z₀)
        @inbounds for n in Base.OneTo(last(axes(C,1)))
            z *= z₀
            r += (((imk*n)^a)*z)*C[n] + (((-imk*n)^a)*conj(z))*C[-n]
        end
    end
    r
end
# N-D implementation of recurrence
@generated function contract(dv::FourierSeriesDerivative{N}, x::Number) where N
quote
    C_ = dv.f.coeffs
    C = (Base.Cartesian.@nref $N C_ i -> i == $N ? 0 : :)
    contract!(C, dv.f.coeffs, x, last(dv.f.period), last(dv.a))
    f = FourierSeries(C, pop(dv.f.period))
    FourierSeriesDerivative(f, pop(dv.a))
end
end
@generated function contract!(r, C::AbstractArray{<:Any,N}, x, a₀, a) where {N}
quote
    -2first(axes(C,$N))+1 == size(C,$N) || throw("array indices are not of form -n:n")
    c₀ = 0^a
    @inbounds Base.Cartesian.@nloops $(N-1) i r begin
        (Base.Cartesian.@nref $(N-1) r i) = c₀*(Base.Cartesian.@nref $N C d -> d == $N ? 0 : i_d)
    end
    if size(C,$N) > 1
        imk = im*2*pi/a₀
        z₀ = cispi(2*x/a₀)
        z = one(z₀)
        for n in Base.OneTo(last(axes(C,$N)))
            z *= z₀
            c = ((imk*n)^a)*z
            c̄ = ((-imk*n)^a)*conj(z)
            @inbounds Base.Cartesian.@nloops $(N-1) i r begin
                (Base.Cartesian.@nref $(N-1) r i) += c*(Base.Cartesian.@nref $N C d -> d == $N ? n : i_d) + c̄*(Base.Cartesian.@nref $N C d -> d == $N ? -n : i_d)
            end
        end
    end
end
end

"""
    (f::FourierSeriesDerivative)(x)

Evaluate `f` at the point `x`.
"""
function (dv::FourierSeriesDerivative{N})(x::SVector{N}) where {N}
    C = dv.f.coeffs
    imk = (im*2*pi) ./ dv.f.period
    ϕ = x ./ dv.f.period
    sum(CartesianIndices(C), init=zero(eltype(dv))) do i
        idx = convert(SVector, i)
        @inbounds C[i] * (cispi(2*dot(ϕ, idx)) * prod((imk.*idx) .^ dv.a))
    end
end
(dv::FourierSeriesDerivative{1})(x::SVector{1}) = dv(only(x))
(dv::FourierSeriesDerivative{1})(x::Number) = contract_(dv.f.coeffs, x, only(dv.f.period), only(dv.a))

value(dv::FourierSeriesDerivative{0}) = value(dv.f)
period(dv::FourierSeriesDerivative) = period(dv.f)

"""
    OffsetFourierSeries(f::AbstractFourierSeries{N}, q::SVector{N,Float64}) where {N}

Represent a Fourier series whose argument is offset by the vector ``\\vec{q}``
and evaluates it as ``f(\\vec{x}-\\vec{q})``.
"""
struct OffsetFourierSeries{N,T<:AbstractFourierSeries{N}} <: AbstractFourierSeries{N}
    f::T
    q::SVector{N,Float64}
end
contract(f::OffsetFourierSeries, x::Number) = OffsetFourierSeries(contract(f.f, x-last(f.q)), pop(f.q))
period(f::OffsetFourierSeries) = period(f.f)
value(f::OffsetFourierSeries{0}) = value(f.f)
Base.eltype(::Type{OffsetFourierSeries{N,T}}) where {N,T} = eltype(T)

"""
    ManyFourierSeries(fs::AbstractFourierSeries{N}...) where {N}

Represents a tuple of Fourier series of the same dimension and periodicity and
contracts them all simultaneously.
"""
struct ManyFourierSeries{N,T<:Tuple{Vararg{AbstractFourierSeries{N}}}} <: AbstractFourierSeries{N}
    fs::T
    period::SVector{N,Float64}
end
function ManyFourierSeries(fs::AbstractFourierSeries{N}...) where {N}
    @assert all(map(==(period(fs[1])), map(period, Base.tail(fs)))) "all periods should match"
    ManyFourierSeries(fs, period(fs[1]))
end
contract(fs::ManyFourierSeries, x::Number) = ManyFourierSeries(map(f -> contract(f, x), fs.fs), pop(fs.period))
period(fs::ManyFourierSeries) = fs.period
value(fs::ManyFourierSeries{0}) = map(value, fs.fs)
Base.eltype(::Type{ManyFourierSeries{N,T}}) where {N,T} = map(eltype, Tuple(T.parameters))

"""
    ManyOffsetsFourierSeries(f, qs..., [origin=true])

Represent a Fourier series evaluated at many different points, and contract them
all simultaneously, returning them in the order the `qs` were passed, i.e.
`(f(x-qs[1]), f(x-qs[2]), ...)`
The `origin` keyword decides whether or not to evaluate ``f`` without an offset,
and if `origin` is true, the value of ``f`` evaluated without an offset will be
returned in the first position of the output.
"""
struct ManyOffsetsFourierSeries{N,T<:AbstractFourierSeries{N},Q} <: AbstractFourierSeries{N}
    f::T
    qs::NTuple{Q,SVector{N,Float64}}
end

function ManyOffsetsFourierSeries(f::AbstractFourierSeries{N}, qs::SVector{N,Float64}...; origin=true) where {N}
    qs_ = ifelse(origin, (fill(0.0, SVector{N,Float64}),), ())
    ManyOffsetsFourierSeries(f, (qs_..., qs...))
end
function contract(f::ManyOffsetsFourierSeries, x::Number)
    fs = map(q -> OffsetFourierSeries(contract(f.f, x-last(q)), pop(q)), f.qs)
    ManyFourierSeries(fs, pop(period(f)))
end
period(f::ManyOffsetsFourierSeries) = period(f.f)
value(f::ManyOffsetsFourierSeries{0}) = value(f.f)
Base.eltype(::Type{ManyOffsetsFourierSeries{N,T,Q}}) where {N,T,Q} = ntuple(_ -> eltype(T), Val{Q}())

