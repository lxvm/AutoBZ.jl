export equispace_integration, automatic_equispace_integration
export equispace_npt_update, equispace_pre_eval, equispace_rule, equispace_int_eval, equispace_integrand

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
number of evaluation points. Should return a vector of tuples with the
integration weight in the first position and the precomputation in the second.
This output is passed to the argument `pre` of `equispace_int_eval`.
"""
function equispace_pre_eval(f, bz::AbstractBZ, npt)
    flag, wsym, nsym = equispace_rule(bz, npt)
    T = SVector{ndims(bz),domain_type(bz)}
    out = Vector{Tuple{T,Int}}(undef, nsym)
    ps = box(bz)
    n = 0
    for i in CartesianIndices(flag)
        if flag[i]
            n += 1
            out[n] = (StaticArrays.sacollect(T, (p[2]-p[1])*(j-1)/npt + p[1] for (j, p) in zip(Tuple(i), ps)), wsym[n])
            n >= nsym && break
        end
    end
    return out
end

equispace_pre_eval(f, bz::FullBZ, npt)
    ((SVector{ndims(bz),domain_type(bz)}(x...), true) for x in Iterators.product([range(l, step=(u-l)/npt, length=npt) for (l,u) in box(bz)]...))



"""
    equispace_rule(bz, npt)

Returns `flag`, `wsym`, and `nsym` containing a mask for the nodes of an
`npt` symmetrized PTR quadrature rule, and the corresponding weights
(see Algorithm 3. in [Kaye et al.](http://arxiv.org/abs/2211.12959)).
"""
@generated function equispace_rule(l::T, npt) where {d, T<:AbstractBZ{d}}
    quote
    xsym = Matrix{Float64}(undef, $d, nsyms(l))
    syms = collect(symmetries(l))
    x = range(-1.0, step=2inv(npt), length=npt)
    flag = ones(Bool, Base.Cartesian.@ntuple $d _ -> npt)
    nsym = 0
    wsym = Vector{Int}(undef, npt^$d)
    Base.Cartesian.@nloops $d i _ -> Base.OneTo(npt) begin
        (Base.Cartesian.@nref $d flag i) || continue
        for (j, S) in enumerate(syms)
            xsym[:, j] = S * (Base.Cartesian.@ncall $d SVector{$d,Float64} k -> x[i_k])
        end
        nsym += 1
        wsym[nsym] = 1
        for j in 2:nsyms(l)
            Base.Cartesian.@nexprs $d k -> begin
                ii_k = 0.5npt * (xsym[k, j] + 1.0) + 1.0
                iii_k = round(Int, ii_k)
                (iii_k - ii_k) > 1e-12 && throw("Inexact index")
            end
            (Base.Cartesian.@nany $d k -> (iii_k > npt)) && continue
            (Base.Cartesian.@nall $d k -> (iii_k == i_k)) && continue
            if (Base.Cartesian.@nref $d flag iii)
                (Base.Cartesian.@nref $d flag iii) = false
                wsym[nsym] += 1
            end
        end
    end
    return flag, wsym, nsym
    end
end


"""
    equispace_int_eval(f, bz, npt, pre=equispace_pre_eval(f, l, npt))

Sums the values of `f` on the precomputed grid points with corresponding
quadrature weights and also multiplies by the mesh cell volume to obtain the
integral on the precomputed domain represented by `pre`, obtained from
`equispace_pre_eval`. Evaluation of `f` is done by the function
`equispace_integrand` in order to create a function boundary between the
quadrature and the integrand evaluation.

!!! note "This routine computes the IBZ integral"
    For getting the full integral from this result, use [`AutoBZ.symmetrize`](@ref)
"""
function equispace_int_eval(f, bz::AbstractBZ, npt, pre=equispace_pre_eval(f, bz, npt))
    vol_sym = vol(bz)/nsyms(bz)
    dvol = vol_sym/npt^ndims(bz)
    int = dvol * sum(x -> x[2]*equispace_integrand(f, x[1]), pre)
    int, pre
end

"""
    equispace_integrand(f, x)

By default, this calls `f(x)`, however the caller may dispatch on the type of
`f` if they would like to specialize this function together with
`equispace_pre_eval` so that `x` is a more useful precomputation (e.g. a Fourier
series evaluated at a grid point).
"""
equispace_integrand(f, x) = f(x)

"""
    equispace_integration(f, bz, npt; pre=equispace_pre_eval(f, l, npt))

Evaluate the integral of `f` over the `bz` using an equispace grid of `npt`
points per dimension, optionally using precomputation `pre`
"""
function equispace_integration(f, bz::AbstractBZ, npt; pre=equispace_pre_eval(f, bz, npt))
    int, = equispace_int_eval(f, bz, npt, pre)
    symmetrize(bz, int), (pre=pre, )
end

"""
    automatic_equispace_integration(f, bz::AbstractBZ; atol=0.0, rtol=sqrt(eps()), maxevals=typemax(Int64))

Automatically evaluates the integral of `f` over the `bz` to within the
requested error tolerances `atol` and `rtol`. Allows optional precomputed data
at two levels of grid refinement `npt1`, `pre1` and `npt2`, `pre2`.
"""
function automatic_equispace_integration(f, bz::AbstractBZ;
    atol=nothing, rtol=nothing, maxevals=typemax(Int64),
    npt1=equispace_npt_update(0, f, atol, rtol),
    pre1=equispace_pre_eval(f, bz, npt1),
    npt2=equispace_npt_update(npt1, f, atol, rtol),
    pre2=equispace_pre_eval(f, bz, npt2),
)
    T = domain_type(bz)
    atol_ = something(atol, zero(T))/nsyms(bz)
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(T)) : zero(T))
    int, err, pre = automatic_equispace_integration_(f, bz, npt1, pre1, npt2, pre2, atol_, rtol_, maxevals)
    symmetrize(bz, int, err)..., pre
end

function automatic_equispace_integration_(f, l, npt1, pre1, npt2, pre2, atol, rtol, maxevals)
    int1, = equispace_int_eval(f, l, npt1, pre1)
    int2, = equispace_int_eval(f, l, npt1, pre2)
    numevals = length(pre1) + length(pre2)
    int2norm = norm(int2)
    err = norm(int1 - int2)

    while true
        (err ≤ max(rtol*int2norm, atol) || numevals ≥ maxevals || !isfinite(err)) && break
        # update coarse result with finer result
        npt1 = npt2
        pre1 = pre2
        int1 = int2
        # evaluate integral on finer grid
        npt2 = equispace_npt_update(npt1, f, atol, rtol)
        int2, pre2 = equispace_int_eval(f, l, npt2)
        numevals += length(pre2)
        # self-convergence error estimate
        int2norm = norm(int2)
        err = norm(int1 - int2)
    end
    return int2, err, (npt1=npt1, pre1=pre1, npt2=npt2, pre2=pre2)
end