export FourierSeries, contract, FourierSeriesDerivative

"""
    FourierSeries(coeffs, period) where {N}

Construct a Fourier series whose coefficients are an array with an element type
supporting addition and scalar multiplication, and whose periodicity on the
`i`th axis is given by `period[i]`. If period is a `Number`, 
"""
struct FourierSeries{N,T<:AbstractArray{<:StaticArray,N}}
    coeffs::T
    period::SVector{N,Float64}
end
# Use this struct, which is more performant with `contract` for the 1D case,
# because the `fill` call can be ignored. This would be a breaking change and
# may require type checking in constructors (since `@inbounds` calls depend on
# this 

# likely changes
# first(f.coeffs) -> f.coeffs
# add Base.eltype(::Type{<:FourierSeries{0,T}}) where {T} = T

struct FourierSeries1{N,T}
    coeffs::T
    period::SVector{N,Float64}
end

FourierSeries(coeffs::AbstractArray{T,N}, period::Real) where {T,N} = FourierSeries(coeffs, fill(period, SVector{N,Float64}))
Base.eltype(::Type{<:FourierSeries{N,T}}) where {N,T} = eltype(T)

"""
    contract(f, x::SVector)

Contracts the indices of the argument `f` in order of `last(x)` to `first(x)`.
To be used as a callback in iterated integration for efficient evaluation.
Any type `T` that wants to implement `contract` needs to define a method with
signature `contract(::T, ::Number)`.
"""
contract(f, x::SVector) = contract(contract(f, last(x)), pop(x))
contract(f, ::SVector{0}) = f

"""
    contract(f::FourierSeries, x::Number, [dim::int])

Contract the outermost index of the `f` at point `x`.
"""
function contract(f::FourierSeries{N}, x::Number) where {N}
    C = f.coeffs
    @inbounds imϕ = 2π*im*x/f.period[N]
    C′ = mapreduce(+, CartesianIndices(C); dims=N) do i
        @inbounds C[i]*exp(i.I[N]*imϕ)
    end
    FourierSeries(reshape(C′, @inbounds axes(C)[1:N-1]), pop(f.period))
end
function contract(f::FourierSeries{N}, x::Number, dim::Int) where {N}
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
# Note that the allocation by `fill` takes as long as running `contract_`
function contract(f::FourierSeries{1}, x::Number)
    C = contract_(f.coeffs, 2π*x / last(f.period))
    FourierSeries(fill(C), pop(f.period))
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

"""
    FourierSeriesDerivative(::FourierSeries, ::SVector)

Construct a Fourier series derivative from a multi-index `a` of derivatives,
e.g. `[1,2,...]` and a `FourierSeries`, whose order of derivative on `i`th
axis is the `i`th element of `a`.
"""
struct FourierSeriesDerivative{N,T<:FourierSeries{N},Ta}
    f::T
    a::SVector{N,Ta}
end

Base.eltype(::Type{<:FourierSeriesDerivative{N,T}}) where {N,T} = eltype(T)
function contract(dv::FourierSeriesDerivative{N}, x::Number) where {N}
    C = dv.f.coeffs
    @inbounds imk = im*2π/dv.f.period[N]
    imϕ = imk*x
    @inbounds a = dv.a[N]
    C′ = mapreduce(+, CartesianIndices(C); dims=N) do i
        @inbounds C[i]*(exp(i.I[N]*imϕ)*((imk*i.I[N])^a))
    end
    f = FourierSeries(reshape(C′, @inbounds axes(C)[1:N-1]), pop(dv.f.period))
    FourierSeriesDerivative(f, pop(dv.a))
end
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
function contract(dv::FourierSeriesDerivative{1}, x::Number)
    C = contract_(dv.f.coeffs, x, 2π/last(dv.f.period), last(dv.a))
    f = FourierSeries(fill(C), pop(dv.f.period))
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