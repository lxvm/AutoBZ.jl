function adaptive_integration_sweep(T, f, ϵ, L, ωs, η::Number; μ=12.3958, callback=contract, kwargs...)
    times = Vector{Float64}(undef, length(ωs))
    ints = Vector{T}(undef, length(ωs))
    errs = Vector{Float64}(undef, length(ωs))

    for (i, ω) in enumerate(ωs)
        g = f(ϵ, ω, η, μ)
        t = time()
        @inbounds ints[i], errs[i] = iterated_integration(g, L; callback=callback, kwargs...)
        @inbounds times[i] = time() - t
    end
    ints, errs, times
end

"""atol ~ c₁*exp(-c₂*η*n), where c₁>0, 1≤c₂≤d may depend on ω and d"""
initial_grid_heuristic(η, atol, rtol) = round(Int, -log(0.1rtol)*inv(η*2.0))
"""Want to choose a p that gets another digit of accuracy from the integrand"""
refine_grid_heuristic(p, s, η)::Int = round(Int, p*s)

function equispace_integration_sweep(T, f, ϵ, ωs, η::Number; μ=12.3958, s=1.2, atol=0, rtol=1e-3, kwargs...)
    int1s = Vector{T}(undef, length(ωs))
    int2s = Vector{T}(undef, length(ωs))
    errs = Vector{Float64}(undef, length(ωs))
    times = Vector{Float64}(undef, length(ωs))
    p1s = Vector{Int}(undef, length(ωs))
    p2s = Vector{Int}(undef, length(ωs))

    t = time()
    p1 = initial_grid_heuristic(η, atol, rtol)
    r1, wsym1 = evaluate_series_ibz(ϵ, p1)
    
    p2 = refine_grid_heuristic(p1, s, η)
    r2, wsym2 = evaluate_series_ibz(ϵ, p2)
    t₀ = time() - t
    for (i, ω) in enumerate(ωs)
        t = time()
        @inbounds errs[i], int1s[i], p1, r1, wsym1, int2s[i], p2, r2, wsym2 = resolve_integrand(T, f(ϵ, ω, η, μ), ϵ, p1, r1, wsym1, p2, r2, wsym2, s, η; atol=atol, rtol=rtol, kwargs...)
        @inbounds times[i] = time() - t
        @inbounds p1s[i] = p1
        @inbounds p2s[i] = p2
    end
    times[1] += t₀
    int1s, int2s, errs, times, p1s, p2s
end

function resolve_integrand(T, f, ϵ::FourierSeries{N}, p1, r1, wsym1, p2, r2, wsym2, s, η; atol=0, rtol=1e-3) where {N}
    int1 = evaluate_integrand_ibz(T, f, p1, r1, wsym1, N)
    int2 = evaluate_integrand_ibz(T, f, p2, r2, wsym2, N)
    err = norm(int1 - int2)
    while err > max(rtol*norm(int1), atol)
        p1 = p2
        r1 = r2
        wsym1 = wsym2
        int1 = int2

        p2 = refine_grid_heuristic(p1, s, η)
        r2, wsym2 = evaluate_series_ibz(ϵ, p2)
        int2 = evaluate_integrand_ibz(T, f, p2, r2, wsym2, N)
        err = norm(int1 - int2)
    end
    err, int1, p1, r1, wsym1, int2, p2, r2, wsym2
end

function evaluate_series_ibz(ϵ::FourierSeries, p::Int)
    flag, wsym = cubic_ibz(p)
    npts = count(flag)
    r = Array{eltype(ϵ)}(undef, npts)
    evaluate_series_ibz!(r, ϵ, p, flag)
    r, first(wsym, npts)
end

function evaluate_integrand_ibz(T, f, p, r, wsym, d)
    int = zero(T)
    for i in eachindex(r)
        @inbounds int += wsym[i] * f(r[i])
    end
    int*inv(p)^d
end

"""
Integrate a function on an equispace grid with the same number of grid points
along each dimension
"""
function equispace_integration(T, f, p::Int; callback=thunk)
    r = zero(T)
    x = range(0.0, step=inv(p), length=p)
    for k in 1:p
        @inbounds g = callback(f, x[k])
        for j in 1:p
            @inbounds h = callback(g, x[j])
            for i in 1:p
                @inbounds r += h(SVector(x[i]))
            end
        end
    end
    r*inv(p)^3
end
# TODO check this is right scaling and length
equispace_integration(T, f, p::Int, ::CubicLimits) = equispace_integration(T, f, p)

function evaluate_integrand(T, f, p::Int; callback=thunk)
    r = Array{T}(undef, p, p, p)
    x = range(0.0, step=inv(p), length=p)
    for k in 1:p
        @inbounds g = callback(f, x[k])
        for j in 1:p
            @inbounds h = callback(g, x[j])
            for i in 1:p
                @inbounds r[i,j,k] = h(SVector(x[i]))
            end
        end
    end
    r
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