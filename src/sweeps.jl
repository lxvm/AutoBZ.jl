function integration_sweep(::typeof(iterated_integration), f::Type{<:Integrand}, ϵ::T, ωs, η::Number; μ=12.3958, kwargs...) where {N,T<:FourierSeries{N}}
    times = Vector{Float64}(undef, length(ωs))
    ints = Vector{eltype(f{N,T})}(undef, length(ωs))
    errs = Vector{Float64}(undef, length(ωs))

    for (i, ω) in enumerate(ωs)
        g = f(ϵ, ω, η, μ)
        t = time()
        @inbounds ints[i], errs[i] = iterated_integration(g, TetrahedralLimits(); kwargs...)
        @inbounds times[i] = time() - t
    end
    ints, errs, times
end

"""atol ~ c₁*exp(-c₂*η*n), where c₁>0, 1≤c₂≤d may depend on ω and d"""
initial_grid_heuristic(η, atol, rtol) = round(Int, -log(0.1rtol)*inv(η*2.0))
"""Want to choose a p that gets another digit of accuracy from the integrand"""
refine_grid_heuristic(p, s, η)::Int = round(Int, p*s)

function integration_sweep(::typeof(equispace_integration), f::Type{<:Integrand}, ϵ::T, ωs, η::Number; μ=12.3958, s=1.2, atol=0, rtol=1e-3, kwargs...) where {N,T<:FourierSeries{N}}
    int1s = Vector{eltype(f{N,T})}(undef, length(ωs))
    int2s = Vector{eltype(f{N,T})}(undef, length(ωs))
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
        @inbounds errs[i], int1s[i], p1, r1, wsym1, int2s[i], p2, r2, wsym2 = resolve_integrand(f(ϵ, ω, η, μ), ϵ, p1, r1, wsym1, p2, r2, wsym2, s, η; atol=atol, rtol=rtol, kwargs...)
        @inbounds times[i] = time() - t
        @inbounds p1s[i] = p1
        @inbounds p2s[i] = p2
    end
    times[1] += t₀
    int1s, int2s, errs, times, p1s, p2s
end

function resolve_integrand(f::Integrand, ϵ::FourierSeries, p1, r1, wsym1, p2, r2, wsym2, s, η; atol=0, rtol=1e-3)
    int1 = evaluate_integrand_ibz(f, p1, r1, wsym1)
    int2 = evaluate_integrand_ibz(f, p2, r2, wsym2)
    err = norm(int1 - int2)
    while err > max(rtol*norm(int1), atol)
        p1 = p2
        r1 = r2
        wsym1 = wsym2
        int1 = int2

        p2 = refine_grid_heuristic(p1, s, η)
        r2, wsym2 = evaluate_series_ibz(ϵ, p2)
        int2 = evaluate_integrand_ibz(f, p2, r2, wsym2)
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

function evaluate_integrand_ibz(f::Integrand{d}, p, r, wsym) where {d}
    int = zero(eltype(f))
    for i in eachindex(r)
        @inbounds int += wsym[i] * f(r[i])
    end
    int*inv(p)^d
end