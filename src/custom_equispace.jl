export pre_eval_contract

function pre_eval_contract(f::AbstractFourierSeries{d}, l::CubicLimits{d}, npt) where {d}
    @assert period(f) ≈ [x[2] - x[1] for x in box(l)] "Integration region doesn't match integrand period"
    f_xs = Vector{Tuple{eltype(f),Int}}(undef, npt^d)
    pre_eval_contract!(f_xs, d, 0, f, box(l), npt)
    return f_xs
end
function pre_eval_contract!(f_xs, d, idx, f, box, npt)
    for i in Base.OneTo(npt)
        pre_eval_contract!(f_xs, d-1, (i-1)+npt*idx, contract(f, (box[d][2]-box[d][1])*(i-1)/npt + box[d][1]), pop(box), npt)
    end
end
function pre_eval_contract!(f_xs, _, idx, f::AbstractFourierSeries{0}, _, _)
    f_xs[idx+1] = (value(f), 1)
end

#=
Using anonymous function expressions is impure so can't use them in @generated
https://docs.julialang.org/en/v1/devdocs/cartesian/#Anonymous-function-expressions-as-macro-arguments
https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions
=#
function pre_eval_contract(f_3::AbstractFourierSeries{3}, l::TetrahedralLimits{3}, npt)
    @assert period(f_3) ≈ [x[2] - x[1] for x in box(l)] "Integration region doesn't match integrand period"
    flag, wsym, nsym = discretize_equispace_(l, npt)
    n = 0
    b = box(l)
    pre = Vector{Tuple{eltype(f_3),Int}}(undef, nsym)
    Base.Cartesian.@nloops 3 i flag j -> f_{j-1} = contract(f_j, (b[j][2]-b[j][1])*(i_j-1)/npt + b[j][1]) begin
        if (Base.Cartesian.@nref 3 flag i)
            n += 1
            pre[n] = (value(f_0), wsym[n])
            n >= nsym && break
        else
            continue
        end
    end
    return pre
end

pre_eval_contract(f::WannierIntegrand, l, npt) = pre_eval_contract(f.s, l, npt)

pre_eval_contract(G::GreensFunction, l, npt) = pre_eval_contract(G.H, l, npt)
pre_eval_contract(A::SpectralFunction, l, npt) = pre_eval_contract(A.G, l, npt)
pre_eval_contract(D::DOSIntegrand, l, npt) = pre_eval_contract(D.A, l, npt)

pre_eval_contract(f::GammaIntegrand, l, npt) = pre_eval_contract(f.HV, l, npt)
pre_eval_contract(f::OCIntegrand, l, npt) = pre_eval_contract(f.HV, l, npt)

function pre_eval_fft(f::FourierSeries{d}, l::CubicLimits{d}, npt) where {d}
    @assert period(f) ≈ [x[2] - x[1] for x in box(l)] "Integration region doesn't match integrand period"
    # zero pad coeffs out to length npt and wrangle into shape for fft
    # ifft(coeffs)
    error("not implemented")
end

function pre_eval_fft(f::FourierSeries{d}, l::TetrahedralLimits{d}, npt) where {d}
    @assert period(f) ≈ [x[2] - x[1] for x in box(l)] "Integration region doesn't match integrand period"
    # zero pad coeffs out to length npt and wrangle into shape for fft
    # ifft(coeffs)
    error("not implemented")
end

function npt_update_sigma(npt, f, atol, rtol)
end

function npt_update_eta(npt, f, atol, rtol)
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