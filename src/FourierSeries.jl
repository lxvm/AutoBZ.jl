export FourierSeries, DOSIntegrand

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
function (f::FourierSeries{1,T})(x::SVector{1}) where {T}
    C = f.coeffs
    -2first(axes(C,1))+1 == size(C,1) || throw("array indices are not of form -n:n")
    ϕ = 2π * im * first(x) / first(f.period)
    z = exp(ϕ)
    r = C[0]
    if size(C,1) > 1
        @inbounds r += z*C[1] + conj(z)*C[-1]
        z′ = z
        for i in 2:last(axes(C,1))
            z′ *= z
            @inbounds r += z′*C[i] + conj(z′)*C[-i]
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

(f::DOSIntegrand)(k) = inv(complex(f.ω + f.μ, f.η)*I - f.ϵ(k))