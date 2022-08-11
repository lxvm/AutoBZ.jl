export equispace_integration

"""
    equispace_integration()

Evaluates a function on an equispace grid 
"""
equispace_integration(f, a, b; kwargs...) = equispace_integration(f, CubicLimits(a, b); kwargs...)
function equispace_integration(f, l::IntegrationLimits; maxevals=typemax(Int64), eval_pts=generic_eval_pts, eval_int=generic_eval_int, atol=0.0, rtol=1e-3, np_init=np_init_heuristic, np_incr=np_incr_heuristic)
    np1 = np_init(f, atol, rtol)
    np2 = np_incr(np1, f, atol, rtol)
    int, err = equispace_integration_(f, l, eval_pts, eval_int, atol, rtol, maxevals)
    int, err
    # symmetrize(l, int, err)
    # TODO: call integration on a more refined grid and compute error estimate
    # TODO: return integral and error estimate
    # TODO: think about how to allow custom function evaluators, e.g. fft
    # TODO: write a equispace_integration! function that reuses data and keeps
    # equispace_integration self-contained
    # TODO: write separate functions for simple equispace integration w/o error
    # TODO: write an interface for IntegrationLimits using points and weights
    # TODO: make equispace integration recursive like adaptive for callbacks,
    # though this might not make sense since grid refinement on the BZ has to
    # happen for all dimensions at the same time for IBZ to work
end

function equispace_integration_(f, l, r, eval_pts, eval_int, np_incr, atol, rtol, maxevals)

    int1 = eval_int(f, np1, r1);
    int2 = eval_int(f, np2, r2);
    err = norm(int1 - int2)
    while err > max(rtol*norm(int1), atol)
        np2_ = np_incr(np2, f, atol, rtol)
        np2_ > maxevals && break
        
        np1 = np2
        r1 = r2
        int1 = int2
        
        np2 = np2_
        r2 = eval_pts(f, np2)
        int2 = eval_int(f, np2, r2)

        err = norm(int1 - int2)
    end
end

"""A sensible initial number of grid points for PTR"""
np_init_heuristic(f, atol, rtol)::Int = 10
"""Want to choose p so PTR gets another digit of accuracy from the integrand"""
np_incr_heuristic(np, f, atol, rtol)::Int = np + 50

function generic_eval_pts(f, np)
    
end
function generic_eval_int(f, np, r2)

end