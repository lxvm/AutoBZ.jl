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