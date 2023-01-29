export equispace_integration, automatic_equispace_integration
export equispace_npt_update, equispace_rule, equispace_rule!, equispace_integrand, equispace_evalrule

"""
    equispace_npt_update(f, npt, [increment=20])

Returns `npt + increment` to try and get another digit of accuracy from PTR.
This fallback option is a heuristic, since the scaling of the error is generally
problem-dependent, so it is appropriate to specialize this method based on the
integrand type 
"""
equispace_npt_update(f, npt, increment=20) = npt + increment

"""
    equispace_dvol(bz::AbstractBZ, npt::Integer)

Return the symmetry-reduced volume of an equispace discretization point.
"""
equispace_dvol(bz::AbstractBZ, npt::Integer) = vol(bz)/nsyms(bz)/npt^ndims(bz)

"""
    equispace_rule(f, bz, npt)

Precomputes the grid points and weights to use for equispace quadrature of `f`
on the domain `bz` while applying the relevant symmetries to `l` to reduce the
number of evaluation points. Should return a vector of tuples with the
integration weight in the first position and the precomputation in the second.
This output is passed to the argument `pre` of `equispace_evalrule`.
"""
function equispace_rule(f, bz::AbstractBZ, npt)
    pre = Vector{Tuple{SVector{ndims(bz),domain_type(bz)},domain_type(bz)}}(undef, 0)
    equispace_rule!(pre, f, bz, npt)
end

"""
    equispace_rule!(pre::Vector, f, bz::AbstractBZ, npt)

In-place version of [`AutoBZ.equispace_rule`](@ref). This function may still
allocate if it needs to resize `pre`
"""
equispace_rule!(pre, f, bz::AbstractBZ, npt) = equispace_rule!(pre, bz, npt)
function equispace_rule!(pre, fbz::FullBZ, npt)
    resize!(pre, npt^ndims(fbz))
    dvol = equispace_dvol(bz, npt)
    for (n, (x, w)) in equispace_rule(fbz, npt)
        pre[n] = (x, dvol*w)
    end
    pre
end
function equispace_rule!(pre, bz::AbstractBZ, npt)
    flag, wsym, nsym = equispace_rule(bz, npt)
    T = SVector{ndims(bz),domain_type(bz)}
    resize!(pre, nsym)
    box = boundingbox(bz)
    dvol = equispace_dvol(bz, npt)
    n = 0
    for i in CartesianIndices(flag)
        flag[i] || continue
        n += 1
        pre[n] = (StaticArrays.sacollect(T, (p[2]-p[1])*(j-1)/npt + p[1] for (j, p) in zip(Tuple(i), box)), dvol*wsym[n])
        n >= nsym && break
    end
    pre
end

"""
    equispace_rule(fbz::FullBZ, npt)

Returns a generator of the coordinates in a uniform grid and the corresponding
unit integration weights.
"""
equispace_rule(fbz::FullBZ, npt) =
    ((SVector{ndims(fbz),domain_type(fbz)}(x...), true) for x in Iterators.product([range(l, step=(u-l)/npt, length=npt) for (l,u) in boundingbox(fbz)]...))


"""
    equispace_rule(bz::AbstractBZ, npt)

Returns `flag`, `wsym`, and `nsym` containing a mask for the nodes of an
`npt` symmetrized PTR quadrature rule, and the corresponding integer weights
(see Algorithm 3. in [Kaye et al.](http://arxiv.org/abs/2211.12959)).
"""
@generated function equispace_rule(bz::T, npt) where {d,T<:AbstractBZ{d}}
    quote
    xsym = Matrix{Float64}(undef, $d, nsyms(bz))
    syms = symmetries(bz)
    nsbz = nsyms(bz)
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
        for j in 2:nsbz
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
    equispace_evalrule(f, rule)

Sums the values of `f` on the precomputed grid points with corresponding
quadrature weights to obtain the integral on the precomputed domain represented
by `rule`, obtained from `equispace_rule`. Evaluation of `f` is done by the
function `equispace_integrand` in order to create a function boundary between
the quadrature and the integrand evaluation.

!!! note "This routine computes the IBZ integral" For getting the full integral
    from this result, use [`AutoBZ.symmetrize`](@ref)
"""
equispace_evalrule(f, rule) = sum(x -> x[2]*equispace_integrand(f, x[1]), rule)


"""
    equispace_integrand(f, x)

By default, this calls `f(x)`, however the caller may dispatch on the type of
`f` if they would like to specialize this function together with
`equispace_rule` so that `x` is a more useful precomputation (e.g. a Fourier
series evaluated at a grid point).
"""
equispace_integrand(f, x) = f(x)

"""
    equispace_integration(f, bz, npt; pre=equispace_rule(f, l, npt))

Evaluate the integral of `f` over the `bz` using an equispace grid of `npt`
points per dimension, optionally using precomputation `pre`
"""
function equispace_integration(f, bz::AbstractBZ; kwargs...)
    buf = equispace_integration_kwargs(f, bz; kwargs...)
    int = equispace_evalrule(f, buf.rule)
    symmetrize(bz, int), buf
end
equispace_integration_kwargs(f, bz; npt=equispace_npt_update(f,0), rule=equispace_rule(f, bz, npt)) =
    (npt=npt, rule=rule)

"""
    automatic_equispace_integration(f, bz::AbstractBZ; atol=0.0, rtol=sqrt(eps()), maxevals=typemax(Int64))

Automatically evaluates the integral of `f` over the `bz` to within the
requested error tolerances `atol` and `rtol`. Allows optional precomputed data
at two levels of grid refinement `npt1`, `rule1` and `npt2`, `rule2` when passed
as keywords.
"""
function automatic_equispace_integration(f, bz::AbstractBZ; kwargs...)
    int, err, buf = automatic_equispace_integration_(f, bz, automatic_equispace_integration_kwargs(f, bz; kwargs...)...)
    symmetrize(bz, int, err)..., buf
end
function automatic_equispace_integration_kwargs(f, bz;
    atol=nothing, rtol=nothing, maxevals=typemax(Int64),
    npt1=equispace_npt_update(f, 0),
    rule1=equispace_rule(f, bz, npt1),
    npt2=equispace_npt_update(f, npt1),
    rule2=equispace_rule(f, bz, npt2),
)
    T = domain_type(bz)
    atol_ = something(atol, zero(T))/nsyms(bz)
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(T)) : zero(T))
    (npt1=npt1, rule1=rule1, npt2=npt2, rule2=rule2, atol=atol_, rtol=rtol_, maxevals=maxevals)
end

function automatic_equispace_integration_(f, bz, npt1, rule1, npt2, rule2, atol, rtol, maxevals)
    int1 = equispace_evalrule(f, rule1)
    int2 = equispace_evalrule(f, rule2)
    numevals = length(rule1) + length(rule2)
    int2norm = norm(int2)
    err = norm(int1 - int2)
    while true
        (err ≤ max(rtol*int2norm, atol) || numevals ≥ maxevals || !isfinite(err)) && break
        # update coarse result with finer result
        int1 = int2
        npt1 = npt2
        resize!(rule1, length(rule2))
        copyto!(rule1, rule2)
        # evaluate integral on finer grid
        npt2 = equispace_npt_update(f, npt1)
        rule2 = equispace_rule!(rule2, f, bz, npt2)
        int2 = equispace_evalrule(f, rule2)
        numevals += length(rule2)
        # self-convergence error estimate
        int2norm = norm(int2)
        err = norm(int1 - int2)
    end
    return int2, err, (npt1=npt1, rule1=rule1, npt2=npt2, rule2=rule2)
end

"""
    equispace_index(npt, i::Int...)

Return the linear index from the Cartesian index
"""
equispace_index(npt, i::Int...) = equispace_index(npt, i) + 1
equispace_index(npt, i::NTuple{N,Int}) where N = i[1] - 1 + npt*equispace_index(npt, i[2:N])
equispace_index(npt, i::Tuple{}) = 0