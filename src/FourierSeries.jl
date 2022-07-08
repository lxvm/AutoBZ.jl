export FourierSeries, HermitianFourierSeries, contract

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
    C = _contract(f.coeffs, first(x) / last(f.period))
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
