export FourierSeries, contract, FourierSeriesDerivative

"""
    FourierSeries(coeffs, period) where {N}

Construct a Fourier series whose coefficients are an array with an element type
supporting addition and scalar multiplication, and whose periodicity on the
`i`th axis is given by `period[i]`.
"""
struct FourierSeries{N,T<:AbstractArray{<:StaticArray,N}}
    coeffs::T
    period::SVector{N,Float64}
end

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
    contract(f::FourierSeries, x::Number)

Contract the outermost index of the `f` at point `x`.
"""
function contract(f::FourierSeries{N}, x::Number)  where {N}
    C = f.coeffs
    imϕ = 2π*im*x/last(f.period)
    C′ = mapreduce(+, CartesianIndices(C); dims=N) do i
        C[i]*exp(last(i.I)*imϕ)
    end
    FourierSeries(reshape(C′, axes(C)[1:N-1]), pop(f.period))
end
# performance hack for larger tensors that allocates less
function contract(f::FourierSeries{3}, x::Number)
    N=3
    C = _contract(f.coeffs, 2π*x / last(f.period))
    FourierSeries(C, pop(f.period))
end
function _contract(C::AbstractVector, ϕ::Number)
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
function _contract(C::AbstractArray{<:Any,3}, ϕ::Number)
    N = 3 # the function body works for any N>1
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
(f::FourierSeries{1})(x::SVector{1}) = _contract(f.coeffs, 2π*first(x)/first(f.period))


"""
    check_derivative_parameters(::Type{Val{N}}, ::Type{α})

Require that `α` be a `Tuple{...}` of `N` `Int`s (like `StaticArraysCore.Size`).
Note: loosen `Int` type restriction to `Real` for fractional derivatives, or
remove for arbitary derivatives so long as `z^a` is defined for `z::Complex` and
`a` in `α`.
"""
function check_derivative_parameters(::Type{Val{N}}, ::Type{α}) where {N,α}
    length(α.parameters) == N || throw(ArgumentError("There must be $N derivatives"))
    all(x->isa(x, Int), α.parameters) || throw(ArgumentError("Derivatives must be a tuple of Ints (e.g. Tuple{1,2,3})"))
    return nothing
end

"""
    FourierSeriesDerivative{α}(::FourierSeries)

Construct a Fourier series derivative from a multi-index `α` of derivatives,
e.g. `Tuple{1,2,...}` and a `FourierSeries`, whose order of derivative on `i`th
axis is the `i`th element of `α`.
"""
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
        # @show ((imk*last(i.I))^a)
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

Evaluate `f` at the point `x`.
"""
function (f::FourierSeriesDerivative{N,T,α})(x::AbstractVector) where {N,T,α}
    C = f.ϵ.coeffs
    imk = (2π*im) ./ f.ϵ.period
    imϕ =  imk .* x
    sum(CartesianIndices(C), init=zero(eltype(f))) do i
        idx = convert(SVector, i)
        # @show (imk.*idx) .^ α.parameters
        @inbounds C[i] * (exp(dot(imϕ, idx)) * prod((imk.*idx) .^ α.parameters))
    end
end
(f::FourierSeriesDerivative{1,T,Tuple{a}})(x::SVector{1}) where{T,a} = _contract(f.ϵ.coeffs, first(x), 2π/first(f.ϵ.period), a)