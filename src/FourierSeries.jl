export FourierSeries, HermitianFourierSeries, contract, FourierSeriesDerivative

"""
Construct a Fourier series whose coefficients are array-valued (hopefully contiguous in memory) such that the resulting matrix-valued Fourier series is Hermitian.
"""
struct FourierSeries{N,T<:AbstractArray{<:StaticArray,N}}
    coeffs::T
    period::SVector{N,Float64}
end

const HermitianFourierSeries{N,T} = FourierSeries{N,T} where {N,T<:AbstractArray{<:SHermitianCompact,N}}

Base.eltype(::Type{<:FourierSeries{N,T}}) where {N,T} = eltype(T)

"""
Contract the outermost index of the Fourier Series
"""
contract(f, x::SVector{1}) = contract(f, first(x))
function contract(f::FourierSeries{N}, x::Number)  where {N}
    C = f.coeffs
    ϕ = 2π*im*x/last(f.period)
    C′ = mapreduce(+, CartesianIndices(C); dims=N) do i
        C[i]*exp(last(i.I)*ϕ)
    end
    FourierSeries(reshape(C′, axes(C)[1:N-1]), pop(f.period))
end
# performance hack for larger tensors that allocates less
function contract(f::FourierSeries{3}, x::Number)
    N=3
    C = _contract(f.coeffs, x / last(f.period))
    FourierSeries(C, pop(f.period))
end
function _contract(C::AbstractVector, ϕ::Number)
    -2first(axes(C,1))+1 == size(C,1) || throw("array indices are not of form -n:n")
    @inbounds r = C[0]
    if size(C,1) > 1
        z₀ = exp(2π*im*ϕ)
        z = one(z₀)
        for i in 1:last(axes(C,1))
            z *= z₀
            @inbounds r += z*C[i] + conj(z)*C[-i]
        end
    end
    r
end
function _contract(C::AbstractArray{<:Any,3}, ϕ::Number)
    N = 3 # the function body works for any N>1
    -2first(axes(C,N))+1 == size(C,N) || throw("array indices are not of form -n:n")
    ax = CartesianIndices(axes(C)[1:N-1])
    @inbounds r = view(C, ax, 0)
    if size(C,N) > 1
        z₀ = exp(2π*im*ϕ)
        z = one(z₀)
        for i in 1:last(axes(C,N))
            z *= z₀
            @inbounds r += z*view(C, ax, i) + conj(z)*view(C, ax, -i)
        end
    end
    r
end

"""
    (f::FourierSeries)(x)
Evaluate the  Fourier series at the point ``x``
"""
function (f::FourierSeries)(x::AbstractVector)
    C = f.coeffs
    ϕ = (2π*im) .* x ./ f.period
    sum(CartesianIndices(C), init=zero(eltype(f))) do i
        C[i] * exp(dot(ϕ, convert(SVector, i)))
    end
end
(f::FourierSeries{1})(x::SVector{1}) = _contract(f.coeffs, first(x)/first(f.period))


"""
Imitating style of  StaticArraysCore
Could loosen `Int` type restriction to `Number` for fractional or complex derivatives
"""
function check_derivative_parameters(::Type{Val{N}}, ::Type{α}) where {N,α}
    length(α.parameters) == N || throw(ArgumentError("There must be $N derivatives"))
    all(x->isa(x, Int), α.parameters) || throw(ArgumentError("Derivatives must be a tuple of Ints (e.g. Tuple{1,2,3})"))
    return nothing
end

struct FourierSeriesDerivative{N,T<:FourierSeries{N},α<:Tuple}
    ϵ::T
    function FourierSeriesDerivative{α}(ϵ::T) where {N,T<:FourierSeries{N},α<:Tuple}
        check_derivative_parameters(Val{N},α)
        # @show N T α
        new{N,typeof(ϵ),α}(ϵ)
    end
end

Base.eltype(::Type{<:FourierSeriesDerivative{N,T,α}}) where {N,T,α} = eltype(T)

function contract(f::FourierSeriesDerivative{N,T,α}, x::Number) where {N,T,α}
    C = f.ϵ.coeffs
    imk = im*2π/last(f.ϵ.period)
    imϕ = imk*x
    a = last(α.parameters)
    C′ = mapreduce(+, CartesianIndices(C); dims=N) do i
        C[i]*(exp(last(i.I)*imϕ)*((imk*last(i.I))^a))
    end
    ϵ = FourierSeries(reshape(C′, axes(C)[1:N-1]), pop(f.ϵ.period))
    @inbounds FourierSeriesDerivative{Tuple{α.parameters[1:N-1]...}}(ϵ)
end
# performance hack for larger tensors that allocates less
function contract(f::FourierSeriesDerivative{3,T,α}, x::Number) where {T,α}
    N=3
    C = _contract(f.ϵ.coeffs, x, 2π/last(f.ϵ.period), last(α.parameters))
    ϵ = FourierSeries(C, pop(f.ϵ.period))
    @inbounds FourierSeriesDerivative{Tuple{α.parameters[1:N-1]...}}(ϵ)
end
function _contract(C::AbstractVector, x, k, a)
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
function _contract(C::AbstractArray{<:Any,3}, x, k, a)
    N = 3 # the function body works for any N>1
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

"""
    (f::FourierSeriesDerivative)(x)
Evaluate the derivative of the Fourier series at the point ``x``
"""
function (f::FourierSeriesDerivative{N,T,α})(x::AbstractVector) where {N,T,α}
    C = f.ϵ.coeffs
    imk = (2π*im) ./ f.ϵ.period
    imϕ =  k .* x
    sum(CartesianIndices(C), init=zero(eltype(f))) do i
        idx = convert(SVector, i)
        @inbounds C[i] * (exp(dot(imϕ, idx)) * prod((imk.*idx) .^ α.parameters))
    end
end
(f::FourierSeriesDerivative{1,T,Tuple{a}})(x::SVector{1}) where{T,a} = _contract(f.ϵ.coeffs, first(x), 2π/first(f.ϵ.period), a)