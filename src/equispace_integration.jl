export equispace_integration, automatic_equispace_integration

"""
    equispace_integration(f, l, npt; pre=nothing, pre_eval=generic_pre_eval, int_eval=generic_int_eval)

Evaluate the integral of `f` over domain `l` using an equispace grid of `npt`
points per dimension, optionally using precomputation `pre`
"""
function equispace_integration(f, l, npt; pre=nothing, pre_eval=generic_pre_eval, int_eval=generic_int_eval)
    pre === nothing && (pre = pre_eval(f, l, npt))
    equispace_integration_(f, l, vol(l)/npt^ndims(l), pre, int_eval)
end
function equispace_integration_(f, l, dvol, pre, int_eval)
    int = int_eval(f, pre, dvol)
    symmetrize(l, int), (pre=pre, )
end

"""
    automatic_equispace_integration(f, a, b; kwargs)
    automatic_equispace_integration(f, l::IntegrationLimits; npt1=0, pre1=0, npt2=0, pre2=0, pre_eval=generic_pre_eval, int_eval=generic_int_eval, atol=0.0, rtol=1e-3, npt_update=generic_npt_update, maxevals=typemax(Int64))

Automatically evaluates the integral of `f` over domain `l` to within the
requested error tolerances `atol` and `rtol`. Allows optional precomputed data
at two levels of grid refinement `npt1`, `pre1` and `npt2`, `pre2` as well as 
customizable precomputation with `pre_eval` and evaluation/summation `int_eval`.
Moreover, a function defining an update strategy for `npt` can be passed as
`npt_update`.
"""
automatic_equispace_integration(f, a, b; kwargs...) = automatic_equispace_integration(f, CubicLimits(a, b); kwargs...)
function automatic_equispace_integration(f, l::IntegrationLimits; npt1=0, pre1=nothing, npt2=0, pre2=nothing, pre_eval=generic_pre_eval, int_eval=generic_int_eval, atol=0.0, rtol=1e-3, npt_update=generic_npt_update, maxevals=typemax(Int64))
    if npt1 == npt2 == 0
        npt1 = npt_update(npt1, f, atol, rtol)
        pre1 = pre_eval(f, l, npt1)
        npt2 = npt_update(npt1, f, atol, rtol)
        pre2 = pre_eval(f, l, npt2)
    end
    automatic_equispace_integration_(f, l, npt1, pre1, npt2, pre2, pre_eval, int_eval, npt_update, atol, rtol, maxevals)
end

function automatic_equispace_integration_(f, l, npt1, pre1, npt2, pre2, pre_eval, int_eval, npt_update, atol, rtol, maxevals)
    vol = vol(l)
    int1 = int_eval(f, pre1, vol/npt1^ndims(l))
    int2 = int_eval(f, pre2, vol/npt2^ndims(l))
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
        npt2 = npt_update(npt1, f, atol, rtol)
        pre2 = pre_eval(f, l, npt2)
        int2 = int_eval(f, pre2, vol/npt2^ndims(l))
        numevals += length(pre2)
        # self-convergence error estimate
        int1norm = norm(int1)
        err = norm(int1 - int2)
    end
    return symmetrize(l, int1, err)..., (npt1=npt1, pre1=pre1, npt2=npt2, pre2=pre2)
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
    generic_int_eval(f, pre, dvol)

Sums the values of `f` on the precomputed grid points with corresponding integer
weights and also multiplies by the mesh cell volume.
"""
generic_int_eval(f, pre, dvol) = dvol * sum(x -> x[2]*f(x[1]), pre)