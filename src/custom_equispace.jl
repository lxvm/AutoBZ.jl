export pre_eval_contract
# TODO:
# implement pre_eval_fft
# generalize pre_eval_contract to all IntegrationLimits{d}

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
function pre_eval_contract(f::AbstractFourierSeries3D, l::TetrahedralLimits{3}, npt)
    @assert collect(period(f)) ≈ [x[2] - x[1] for x in box(l)] "Integration region doesn't match integrand period"
    flag, wsym, nsym = discretize_equispace_(l, npt)
    n = 0
    b = box(l)
    pre = Vector{Tuple{eltype(f),Int}}(undef, nsym)
    for k in axes(flag, 3)
        contract!(f, (b[3][2]-b[3][1])*(k-1)/npt, 3)
        for j in axes(flag, 2)
            contract!(f, (b[2][2]-b[2][1])*(j-1)/npt, 2)
            for i in axes(flag, 1)
                if flag[i,j,k]
                    n += 1
                    pre[n] = (f((b[1][2]-b[1][1])*(i-1)/npt), wsym[n])
                    n >= nsym && break
                end
            end
        end
    end
    return pre
end

"""
    equispace_pre_eval(f::WannierIntegrand, l::IntegrationLimits, npt)

This function will evaluate the Fourier series and integration weights needed
for equispace integration of `f` at `npt` points per dimension. `l` should
contain the relevant symmetries needed for IBZ integration, if desired.
"""
equispace_pre_eval(f::WannierIntegrand, l, npt) = pre_eval_contract(f.s, l, npt)

equispace_pre_eval(D::DOSIntegrand, l, npt) = pre_eval_contract(D.H, l, npt)

equispace_pre_eval(f::GammaIntegrand, l, npt) = pre_eval_contract(f.HV, l, npt)


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


function equispace_npt_update(npt, D::DOSIntegrand, atol, rtol)
    η = im_sigma_to_eta(-imag(D.M))
    npt_update_eta(npt, η, atol, rtol)
end

function equispace_npt_update(npt, g::GammaIntegrand, atol, rtol)
    ηω = im_sigma_to_eta(-imag(g.Mω))
    ηΩ = im_sigma_to_eta(-imag(g.MΩ))
    npt_update_eta(npt, min(ηω, ηΩ), atol, rtol)
end

im_sigma_to_eta(x::UniformScaling) = -x.λ

"""
    npt_update_eta(npt, η, atol, rtol)

Implements the heuristics for incrementing kpts suggested by Kaye et al.
"""
function npt_update_eta(npt, η, atol, rtol)
    npt == 0 && return round(Int, 6/η)
    npt + max(50, round(Int, 2.3/η))
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