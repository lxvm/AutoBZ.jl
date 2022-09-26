# singleton

function pre_eval_contract(f::AbstractFourierSeries{d}, l::CubicLimits{d}, npt) where {d}
    @assert period(f) ≈ [x[2] - x[1] for x in box(l)] "Integration region doesn't match integrand period"
    f_xs = Array{eltype(f),d}(undef, ntuple(_ -> npt, Val{d}()))
    pre_eval_contract!(f_xs, d, (), f, box(l), npt)
    return f_xs
end

function pre_eval_contract!(f_xs, d, idx, f, box, npt)
    for i in axes(f_xs, d)
        pre_eval_contract!(f_xs, d-1, (idx..., i), contract(f, (box[d][2]-box[d][1])*(i-1)/npt + box[d][1]), pop(box), npt)
    end
end

function pre_eval_contract!(f_xs, _, idx, f::AbstractFourierSeries{0}, _, _) where {N}
    f_xs[CartesianIndex(idx)] = value(f)
end

function pre_eval_contract(f::AbstractFourierSeries{d}, l::TetrahedralLimits{d}, npt) where {d}
    @assert f.period ≈ [x[2] - x[1] for x in box(l)] "Integration region doesn't match integrand period"
    error("not implemented") # TODO (need to modify discretize_equispace code)
end

pre_eval_contract(G::GreensFunction, l, npt) = pre_eval_contract(G.H, l, npt)
pre_eval_contract(A::SpectralFunction, l, npt) = pre_eval_contract(A.G, l, npt)
pre_eval_contract(D::DOSIntegrand, l, npt) = pre_eval_contract(D.A, l, npt)

pre_eval_contract(f::GammaIntegrand, l, npt) = pre_eval_contract(f.HV, l, npt)

function int_eval_dft(f::Union{DOSIntegrand,GammaIntegrand}, l::CubicLimits, npt, pre)
    sum(x -> [f(x), 1], pre) .* [prod(x -> x[2]-x[1], box(l))/(npt^ndims(l)*nsyms(l)), 1]
end

function pre_eval_fft(f::FourierSeries{d}, l::CubicLimits{d}, npt) where {d}
    @assert f.period ≈ [x[2] - x[1] for x in box(l)] "Integration region doesn't match integrand period"
    # zero pad coeffs out to length npt and wrangle into shape for fft
    # ifft(coeffs)
    error("not implemented") # TODO (need to modify discretize_equispace code)
end

function pre_eval_fft(f::FourierSeries{d}, l::TetrahedralLimits{d}, npt) where {d}
    @assert f.period ≈ [x[2] - x[1] for x in box(l)] "Integration region doesn't match integrand period"
    # zero pad coeffs out to length npt and wrangle into shape for fft
    # ifft(coeffs)
    error("not implemented") # TODO (need to modify discretize_equispace code)
end

#=
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
=#