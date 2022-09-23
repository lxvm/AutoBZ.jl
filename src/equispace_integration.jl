export equispace_integration

"""
    equispace_integration(f, a, b; kwargs)
    equispace_integration(f, l::IntegrationLimits; npt1=0, pre1=0, npt2=0, pre2=0, pre_eval=generic_pre_eval, int_eval=generic_int_eval, atol=0.0, rtol=1e-3, npt_update=generic_npt_update, maxevals=typemax(Int64))

Automatically evaluates the integral of `f` over domain `l` to within the
requested error tolerances `atol` and `rtol`. Allows optional precomputed data
at two levels of grid refinement `npt1`, `pre1` and `npt2`, `pre2` as well as 
customizable precomputation with `pre_eval` and evaluation/summation `int_eval`.
Moreover, a function defining an update strategy for `npt` can be passed as
`npt_update`.
"""
equispace_integration(f, a, b; kwargs...) = equispace_integration(f, CubicLimits(a, b); kwargs...)
function equispace_integration(f, l::IntegrationLimits; npt1=0, pre1=0, npt2=0, pre2=0, pre_eval=generic_pre_eval, int_eval=generic_int_eval, atol=0.0, rtol=1e-3, npt_update=generic_npt_update, maxevals=typemax(Int64))
    if npt1 == pre1 == npt2 == pre2 == 0
        npt1 = npt_update(npt1, f, atol, rtol)
        pre1 = pre_eval(f, l, npt1)
        npt2 = npt_update(npt1, f, atol, rtol)
        pre2 = pre_eval(f, l, npt2)
    end
    int, err, other = equispace_integration_(f, l, npt1, pre1, npt2, pre2, pre_eval, int_eval, npt_update, atol/nsyms(l), rtol, maxevals)
    return (symmetrize(l, int, err)..., other)
end

function equispace_integration_(f, l, npt1, pre1, npt2, pre2, pre_eval, int_eval, npt_update, atol, rtol, maxevals)
    int1, numevals1 = int_eval(f, l, npt1, pre1)
    int2, numevals2 = int_eval(f, l, npt2, pre2)
    numevals = numevals1 + numevals2
    int1norm = norm(int1)
    err = norm(int1 - int2)

    while true
        (err ≤ max(rtol*int1norm, atol) || numevals ≥ maxevals || !isfinite(int1norm)) && break
        # update coarse result with finer result
        npt1 = npt2
        pre1 = pre2
        int1 = int2
        # evaluate integral on finer grid
        npt2 = npt_update(npt1, f, atol, rtol)
        pre2 = pre_eval(f, l, npt2)
        int2, numevals_= int_eval(f, l, npt2, pre2)
        numevals += numevals_
        # self-convergence error estimate
        int1norm = norm(int1)
        err = norm(int1 - int2)
    end
    return int1, err, (npt1=npt1, pre1=pre1, npt2=npt2, pre2=pre2)
end

"""
    generic_npt_update(npt::Integer, f, atol, rtol)

Returns a new `npt` to try and get another digit of accuracy from PTR.
This fallback option is a heuristic, since the scaling of the error is generally
problem-dependent.
"""
generic_npt_update(npt, f, atol, rtol)::Int = npt + 20

"""
    generic_pre_eval(f, l, npt)

Precomputes the grid points and weights to use for equispace quadrature of `f`
on the domain `l` while applying the relevant symmetries to `l` to reduce the
number of evaluation points.
"""
generic_pre_eval(f, l, npt) = discretize_equispace(l, npt)

"""
    generic_int_eval(f, l, npt, pre)

Sums the values of `f` on the precomputed grid points with corresponding weights
and also counts the number of evaluations.
"""
generic_int_eval(f, l, npt, pre) = sum(x -> [x[2]*f(x[1]), 1], pre) .* [prod(x -> x[2]-x[1], box(l))/(npt^ndims(l)*nsyms(l)), 1]