export equispace_npt_update, equispace_pre_eval, equispace_int_eval,
    equispace_integration, automatic_equispace_integration

"""
    equispace_npt_update(npt, f, atol, rtol)

Returns a new `npt` to try and get another digit of accuracy from PTR.
This fallback option is a heuristic, since the scaling of the error is generally
problem-dependent.
"""
equispace_npt_update(npt, f, atol, rtol)::Int = npt + 20

"""
    equispace_pre_eval(f, l, npt)

Precomputes the grid points and weights to use for equispace quadrature of `f`
on the domain `l` while applying the relevant symmetries to `l` to reduce the
number of evaluation points.
"""
equispace_pre_eval(f, l, npt) = discretize_equispace(l, npt)

"""
    equispace_int_eval(f, pre, dvol)

Sums the values of `f` on the precomputed grid points with corresponding integer
weights and also multiplies by the mesh cell volume.
"""
equispace_int_eval(f, pre, dvol) = dvol * sum(x -> x[2]*f(x[1]), pre)

"""
    equispace_integration(f, l, npt; pre=nothing, equispace_pre_eval=generic_equispace_pre_eval, equispace_int_eval=generic_equispace_int_eval)

Evaluate the integral of `f` over domain `l` using an equispace grid of `npt`
points per dimension, optionally using precomputation `pre`
"""
function equispace_integration(f, l, npt; pre=nothing)
    pre === nothing && (pre = equispace_pre_eval(f, l, npt))
    equispace_integration_(f, l, vol(l)/(npt^ndims(l)*nsyms(l)), pre)
end
function equispace_integration_(f, l, dvol, pre)
    int = equispace_int_eval(f, pre, dvol)
    symmetrize(l, int), (pre=pre, )
end

"""
    automatic_equispace_integration(f, a, b; kwargs)
    automatic_equispace_integration(f, l::IntegrationLimits; npt1=0, pre1=0, npt2=0, pre2=0, atol=0.0, rtol=1e-3 maxevals=typemax(Int64))

Automatically evaluates the integral of `f` over domain `l` to within the
requested error tolerances `atol` and `rtol`. Allows optional precomputed data
at two levels of grid refinement `npt1`, `pre1` and `npt2`, `pre2`.
"""
automatic_equispace_integration(f, a, b; kwargs...) = automatic_equispace_integration(f, CubicLimits(a, b); kwargs...)
function automatic_equispace_integration(f, l::IntegrationLimits; npt1=0, pre1=nothing, npt2=0, pre2=nothing, atol=0.0, rtol=1e-3, maxevals=typemax(Int64))
    if npt1 == npt2 == 0
        npt1 = equispace_npt_update(npt1, f, atol, rtol)
        pre1 = equispace_pre_eval(f, l, npt1)
        npt2 = equispace_npt_update(npt1, f, atol, rtol)
        pre2 = equispace_pre_eval(f, l, npt2)
    end
    automatic_equispace_integration_(f, l, npt1, pre1, npt2, pre2, atol, rtol, maxevals)
end

function automatic_equispace_integration_(f, l, npt1, pre1, npt2, pre2, atol, rtol, maxevals)
    vol_sym = vol(l)/nsyms(l) # rescale by nsym since weights from generic_equispace_pre_eval lack this factor
    int1 = equispace_int_eval(f, pre1, vol_sym/npt1^ndims(l))
    int2 = equispace_int_eval(f, pre2, vol_sym/npt2^ndims(l))
    numevals = length(pre1) + length(pre2)
    int1norm = norm(int1)
    err = norm(int1 - int2)
    atol_sym = atol/nsyms(l) # this tolerance on err_ibz is stricter than atol on err_fbz by triangle inequality

    while true
        (err ≤ max(rtol*int1norm, atol_sym) || numevals ≥ maxevals || !isfinite(int1norm)) && break
        # update coarse result with finer result
        npt1 = npt2
        pre1 = pre2
        int1 = int2
        # evaluate integral on finer grid
        npt2 = equispace_npt_update(npt1, f, atol, rtol)
        pre2 = equispace_pre_eval(f, l, npt2)
        int2 = equispace_int_eval(f, pre2, vol_sym/npt2^ndims(l))
        numevals += length(pre2)
        # self-convergence error estimate
        int1norm = norm(int1)
        err = norm(int1 - int2)
    end
    return symmetrize(l, int1, err)..., (npt1=npt1, pre1=pre1, npt2=npt2, pre2=pre2)
end