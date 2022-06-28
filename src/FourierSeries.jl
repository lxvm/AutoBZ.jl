export FourierSeries, DOSIntegrand, contract

"""
Construct a Fourier series whose coefficients are matrix valued (hopefully contiguous in memory) such that the resulting matrix-valued Fourier series is Hermitian.
"""
struct FourierSeries{N,T<:AbstractArray{<:SArray,N}}
    coeffs::T
    period::SVector{N,Float64}
end

"""
    (f::FourierSeries)(x)

Evaluate the matrix-valued Fourier series at the point ``x``
"""
function (f::FourierSeries{N,T})(x) where {N,T}
    C = f.coeffs
    ϕ = (2π*im) .* x ./ f.period
    sum(CartesianIndices(C), init=zero(eltype(T))) do i
        C[i] * exp(dot(ϕ, convert(SVector, i)))
    end
end
function (f::FourierSeries{1})(x::SVector{1})
    C = f.coeffs
    # this function is the 1d version of contract
    -2first(axes(C,1))+1 == size(C,1) || throw("array indices are not of form -n:n")
    @inbounds r = C[0]
    if size(C,1) > 1
        z₀ = exp(2π * im * first(x) / first(f.period))
        z = one(z₀)
        for i in 1:last(axes(C,1))
            z *= z₀
            @inbounds r += z*C[i] + conj(z)*C[-i]
        end
    end
    r
end

"""
    DOSIntegrand(ϵ,ω,η,μ)
A function that computes the DOS from the constituent parameters.
"""
struct DOSIntegrand{N,T}
    ϵ::FourierSeries{N,T}
    ω::Float64
    η::Float64
    μ::Float64
end

(f::DOSIntegrand)(k) = hinv(complex(f.ω + f.μ, f.η)*I - f.ϵ(k))

"""
Contract the outermost index of the Fourier Series
"""
contract(f, x::SVector{1}) = contract(f, first(x))
contract(f::DOSIntegrand, x) = DOSIntegrand(contract(f.ϵ, x), f.ω, f.η, f.μ)
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
    C = contract(f.coeffs, first(x) / last(f.period))
    j = SVector{N-1}(1:N-1)
    FourierSeries(C, f.period[j])
end
function contract(C::AbstractArray{<:Any,3}, ϕ::Number)
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
