# Possible TODO for FourierSeries
# - replace dependence on SVector with NTuple for ease of use by new users
# - enable reduction over multiple dims simulatneously (?) may not be used
export FourierSeries, contract, FourierSeriesDerivative, ManyFourierSeries

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
"""
(f::AbstractFourierSeries{N})(x::SVector{N}) where {N} = value(contract(f, x))
(f::AbstractFourierSeries{1})(x::Number) = value(contract(f, x))

"""
    value(::AbstractFourierSeries{0})

Return the evaluated Fourier series whose indices have all been contracted.
Typically, this value has the same units as the Fourier series coefficients.
"""
function value end

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
"""
struct FourierSeries{N,T} <: AbstractFourierSeries{N}
    coeffs::T
    period::SVector{N,Float64}
end

"""
    FourierSeries(coeffs::AbstractArray{T,N}, period::Real) where {T,N}

If period is a `Real`, this constructor will infer the number of
input dimensions of the Fourier series from the array dimensionality of the
coefficients, and `period` will become the period of all of the dimensions.
"""
FourierSeries(coeffs::AbstractArray{T,N}, period::Real) where {T,N} = FourierSeries(coeffs, fill(period, SVector{N,Float64}))
Base.eltype(::Type{<:FourierSeries{N,T}}) where {N,T} = eltype(T)
Base.eltype(::Type{<:FourierSeries{0,T}}) where {T} = T

"""
    contract(f::FourierSeries{N}, x::Number, [dim=N]) where {N}

Contract index `dim` of the coefficients of `f` at the spatial point `x`.
The default `dim` is the outermost dimension to preserve memory locality.
"""
function contract(f::FourierSeries{N}, x::Number; dim::Int=N) where {N}
    1 <= dim <= N || error("Choose dim=$dim in 1:$N")
    C = f.coeffs
    @inbounds imϕ = 2π*im*x/f.period[dim]
    C′ = mapreduce(+, CartesianIndices(C); dims=dim) do i
        @inbounds C[i]*exp(i.I[dim]*imϕ)
    end
    idx = sacollect(SVector{N-1,Int}, i >= dim ? i+1 : i for i in 1:N-1)
    FourierSeries(reshape(C′, @inbounds axes(C)[idx]), @inbounds f.period[idx])
end
# evaluation by recurrence is faster for 1D Fourier series evaluation
function contract(f::FourierSeries{1}, x::Number)
    C = contract_(f.coeffs, 2π*x / last(f.period))
    FourierSeries(C, pop(f.period)) # for consistency with N>1 edit C -> fill(C)
    # however, allocating a 0-dimensional array slows things down so I have
    # treat FourierSeries{0} as a special case since I store the value in coeffs
end
function contract_(C::AbstractVector, ϕ::Number)
    -2first(axes(C,1))+1 == size(C,1) || throw("array indices are not of form -n:n")
    @inbounds r = C[0]
    if size(C,1) > 1
        z₀ = exp(im*ϕ)
        z = one(z₀)
        for i in 1:last(axes(C,1))
            z *= z₀
            @inbounds r += z*C[i] + conj(z)*C[-i]
        end
    end
    r
end
#= N-D implementation of recurrence, which is slower so disabled
function contract(f::FourierSeries{N,T}, x::Number) where {N,T}
    C = contract_(f.coeffs, 2π*x / last(f.period))
    FourierSeries(C, pop(f.period))
end
function contract_(C::AbstractArray{<:Any,N}, ϕ::Number) where {N}
    -2first(axes(C,N))+1 == size(C,N) || throw("array indices are not of form -n:n")
    ax = CartesianIndices(axes(C)[1:N-1])
    @inbounds r = view(C, ax, 0)
    if size(C,N) > 1
        z₀ = exp(im*ϕ)
        z = one(z₀)
        for i in 1:last(axes(C,N))
            z *= z₀
            @inbounds r += z*view(C, ax, i) + conj(z)*view(C, ax, -i)
        end
    end
    r
end
=#

"""
    (f::FourierSeries)(x)

Evaluate `f` at the point `x`.
"""
function (f::FourierSeries)(x::AbstractVector)
    C = f.coeffs
    imϕ = (2π*im) .* x ./ f.period
    sum(CartesianIndices(C), init=zero(eltype(f))) do i
        C[i] * exp(dot(imϕ, convert(SVector, i)))
    end
end
(f::FourierSeries{1})(x::SVector{1}) = f(only(x))
(f::FourierSeries{1})(x::Number) = contract_(f.coeffs, 2π*x/only(f.period))

value(f::FourierSeries{0}) = f.coeffs
period(f::FourierSeries) = f.period

"""
    FourierSeriesDerivative(f::FourierSeries{N}, a::SVector{N}) where {N}

Represent the differential of Fourier series `f` by a multi-index `a` of
derivatives, e.g. `[1,2,...]`, whose .


Construct a Fourier series whose coefficients are given by the coefficient array
array `coeffs` whose `eltype` should support addition and scalar multiplication,
and whose periodicity on the `i`th axis is given by `period[i]`. This type
represents the Fourier series
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
function contract(dv::FourierSeriesDerivative{N}, x::Number, dim::Int) where {N}
    1 <= dim <= N || error("Choose dim=$dim in 1:$N")
    C = dv.f.coeffs
    @inbounds imk = im*2π/dv.f.period[dim]
    imϕ = imk*x
    @inbounds a = dv.a[dim]
    C′ = mapreduce(+, CartesianIndices(C); dims=dim) do i
        @inbounds C[i]*(exp(i.I[dim]*imϕ)*((imk*i.I[dim])^a))
    end
    idx = sacollect(SVector{N-1,Int}, i >= dim ? i+1 : i for i in 1:N-1)
    f = FourierSeries(reshape(C′, @inbounds axes(C)[idx]), dv.f.period[idx])
    FourierSeriesDerivative(f, @inbounds dv.a[idx])
end
# 1D recurrence is still faster
function contract(dv::FourierSeriesDerivative{1}, x::Number)
    C = contract_(dv.f.coeffs, x, 2π/last(dv.f.period), last(dv.a))
    f = FourierSeries(C, pop(dv.f.period)) # for consistency with higher dimensions, C -> fill(C)
    FourierSeriesDerivative(f, pop(dv.a))
end
function contract_(C::AbstractVector, x, k, a)
    -2first(axes(C,1))+1 == size(C,1) || throw("array indices are not of form -n:n")
    @inbounds r = (0^a)*C[0]
    if size(C,1) > 1
        imk = im*k
        z₀ = exp(imk*x)
        z = one(z₀)
        for i in 1:last(axes(C,1))
            z *= z₀
            @inbounds r += (((imk*i)^a)*z)*C[i] + (((-imk*i)^a)*conj(z))*C[-i]
        end
    end
    r
end
#= N-D implementation of recurrence, which is slower so disabled
function contract(dv::FourierSeriesDerivative, x::Number)
    C = contract_(dv.f.coeffs, x, 2π/last(dv.f.period), last(dv.a))
    f = FourierSeries(C, pop(dv.f.period))
    FourierSeriesDerivative(f, pop(dv.a))
end
function contract_(C::AbstractArray{<:Any,N}, x, k, a) where {N}
    -2first(axes(C,N))+1 == size(C,N) || throw("array indices are not of form -n:n")
    ax = CartesianIndices(axes(C)[1:N-1])
    @inbounds r = (0^a)*view(C, ax, 0)
    if size(C,N) > 1
        imk = im*k
        z₀ = exp(imk*x)
        z = one(z₀)
        for i in 1:last(axes(C,N))
            z *= z₀
            @inbounds r += (((imk*i)^a)*z)*view(C, ax, i) + (((-imk*i)^a)*conj(z))*view(C, ax, -i)
        end
    end
    r
end
=#

"""
    (f::FourierSeriesDerivative)(x)

Evaluate `f` at the point `x`.
"""
function (dv::FourierSeriesDerivative)(x::AbstractVector)
    C = dv.f.coeffs
    imk = (2π*im) ./ dv.f.period
    imϕ = imk .* x
    sum(CartesianIndices(C), init=zero(eltype(dv))) do i
        idx = convert(SVector, i)
        @inbounds C[i] * (exp(dot(imϕ, idx)) * prod((imk.*idx) .^ dv.a))
    end
end
(dv::FourierSeriesDerivative{1})(x::SVector{1}) = dv(only(x))
(dv::FourierSeriesDerivative{1})(x::Number) = contract_(dv.f.coeffs, x, 2π/only(dv.f.period), only(dv.a))

value(dv::FourierSeriesDerivative{0}) = value(dv.f)
period(dv::FourierSeriesDerivative) = period(dv.f)

"""

"""
struct OffsetFourierSeries{N,T<:AbstractFourierSeries{N}} <: AbstractFourierSeries{N}
    f::T
    q::SVector{N,Float64}
end

"""
"""
struct ManyFourierSeries{N,T<:Tuple{Vararg{AbstractFourierSeries{N}}}} <: AbstractFourierSeries{N}
    fs::T
end