export equispace_integration

"""
Integrate a function on an equispace grid with the same number of grid points
along each dimension
"""
function equispace_integration(f::T, p::Int) where {T<:Integrand}
    r = zero(eltype(T))
    x = range(0.0, step=inv(p), length=p)
    for k in 1:p
        @inbounds g = contract(f, x[k])
        for j in 1:p
            @inbounds h = contract(g, x[j])
            for i in 1:p
                @inbounds r += h(SVector(x[i]))
            end
        end
    end
    r*inv(p)^3
end

function evaluate_series(f::FourierSeries{3}, p::Int)
    r = Array{eltype(f)}(undef, p, p, p)
    x = range(0.0, step=inv(p), length=p)
    for k in 1:p
        @inbounds g = contract(f, x[k])
        for j in 1:p
            @inbounds h = contract(g, x[j])
            for i in 1:p
                @inbounds r[i,j,k] = h(SVector(x[i]))
            end
        end
    end
    r
end

fft_equispace_integration(f::DOSIntegrand, p::Int) = tr(fft_equispace_integration(f.A, p))
function fft_equispace_integration(A::SpectralFunction, p::Int)
    ϵk = fft_evaluate_series(A.ϵ, p)
    r = zero(eltype(A))
    for i in CartesianIndices(size(ϵk)[3:end])
        r += hinv(complex(A.ω + A.μ, A.η)*I - SMatrix{3,3,ComplexF64}(ϵk[CartesianIndices((3,3)), i]))
    end
    imag(r)*inv(p)^3/(-pi)
end

"""
Evaluate a FourierSeries by FFT. The indices are assumed to be the frequencies,
but these get mapped back modulo `p` to the domain of FFT frequencies, so the
caller is responsible for ensuring that `p` is large enough to capture their
high frequency behavior.
"""
function fft_evaluate_series(f::FourierSeries, p::Int)
    C = f.coeffs
    maximum(size(C)) > p && throw("Inexact: Choose more grid points, $p, per dim than Fourier coefficients per dim, $(size(C))")
    # pad C such that size(C) = (3,3,p,p,p)
    S1 = size(eltype(C))
    S2 = Tuple(fill(p, ndims(C)))
    Z = zeros(ComplexF64, S1..., S2...)
    populate_fourier_coefficients!(Z, C)
    # return Z
    # fft(C) along the dimensions of size p
    fft!(Z, (length(S1)+1):(length(S1)+length(S2)))
    # convert back to StaticArray for efficient inversion
    return Z
    return reshape(reinterpret(SArray{Tuple{S1...},ComplexF64,length(S1),prod(S1)}, Z), S2)
    X = Array{SArray{Tuple{S1...},ComplexF64}}(undef,S2)
    for i in CartesianIndices(S2)
        X[i] = SArray{Tuple{S1...},ComplexF64}(view(Z, CartesianIndices(S1), i))
    end
    X
end

function populate_fourier_coefficients!(Z::AbstractArray, C::AbstractArray{<:StaticArray})
    S1 = size(eltype(C))
    preI = CartesianIndices(S1)
    S2 = size(Z)[(length(S1)+1):end]
    for i in CartesianIndices(C)
        Z[preI, CartesianIndex(_to_fftw_index(i.I, S2))] = C[i]
    end
end

"""
Convert positive/negative indices of Fourier coefficients to those suitable for FFTW.
`i` is a tuple of +/- indices by dimension, and `j` is a tuple of the size of
each dimension.
"""
_to_fftw_index(i::NTuple{N, Int}, j::NTuple{N, Int}) where {N} = mod.(i, j) .+ 1