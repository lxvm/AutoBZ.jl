# TODO: allow an arbitrary shift of the grid


"""
    equispace_rule(f, a, b, npt)

Returns an iterator over the grid points for dimension `n` equispace grid on the
hypercube with extremal vertices `a` and `b`. The points along dimension `n` are
`range(a[n], b[n], length=npt+1)[1:npt]`.
"""
equispace_rule(_, a::SVector{N}, b::SVector{N}, npt) where N =
    Iterators.product(ntuple(n -> range(a[n], b[n], length=npt+1)[1:npt], Val(N))...)

"""
    equispace_evalrule(f, a, b, npt)

Sums the values of `f` on an equispace grid of `npt` points per dimension on the
hypercube with extremal vertices `a` and `b`.

!!! note "For developers"

Evaluation of `f` is done by the function [`equispace_integrand`](@ref) in order
to create a function boundary between the quadrature and the integrand
evaluation.
"""
equispace_evalrule(f, a::SVector{N}, b::SVector{N}, npt::Integer) where N =
    sum(x -> equispace_integrand(f, x), equispace_rule(f, a, b, npt)) * prod(b-a) / npt^N

"""
    equispace_integrand(f, x) = f(x)

!!! note "For developers"
    The caller may dispatch on the type of `f` if they would like to specialize
    this function together with [`equispace_rule`](@ref) so that `x` is
    a more useful precomputation (e.g. a Fourier series evaluated at `x`).
"""
equispace_integrand(f, x) = f(x)

"""
    equispace_integration(f, a, b; npt=equispace_npt_update(f, 0), rule=equispace_rule(f, l, npt))

Evaluates the integral of a periodic function `f` over the limits `l` using an
equispace integration rule of `npt` points per dimension When the keyword `npt`
is provided, `rule`, in which case `npt` is ignored.

The limits of integration `l` can be any tuple or vector whose entries are
floating-point numbers and give the period of `f` for each variable. The
integration points will be of the same type as `l`, and see
[`equispace_integrand`](@ref) 

Returns `(int, buf)` where `int` is the result and 
"""
function equispace_integration(f, a::AbstractVector{T}, b::AbstractVector{S}; kwargs...) where {T,S}
    length(a) == length(b) || throw(DimensionMismatch("endpoints $a and $b must have the same length"))
    F = float(promote_type(T, S))
    a_ = SVector{length(a),F}(a)
    b_ = SVector{length(a),F}(b)
    buf = equispace_integration_kwargs(f, a_, b_; kwargs...)
    int = equispace_evalrule(f, a_, b_, buf.npt)
    int, buf
end
equispace_integration_kwargs(f, _, _; npt=equispace_npt_update(f,0)) =
    (npt=npt,)

"""
    automatic_equispace_integration(f, l; atol=0.0, rtol=sqrt(eps()), maxevals=typemax(Int64))

Automatically evaluates the integral of `f` over the `l` to within the
requested error tolerances `atol` and `rtol`. Allows optional precomputed data
at two levels of grid refinement `npt1`, `rule1` and `npt2`, `rule2` when passed
as keywords.
"""
function automatic_equispace_integration(f, l_; kwargs...)
    l = float(l_)
    automatic_equispace_integration_(f, l, automatic_equispace_integration_kwargs(f, l; kwargs...)...)
end
function automatic_equispace_integration_kwargs(f, l;
    atol=nothing, rtol=nothing, maxevals=typemax(Int64),
    npt1=equispace_npt_update(f, 0),
    rule1=equispace_rule(f, l, npt1),
    npt2=equispace_npt_update(f, npt1),
    rule2=equispace_rule(f, l, npt2),
)
    T = eltype(l)
    atol_ = something(atol, zero(T))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(T)) : zero(T))
    (npt1=npt1, rule1=rule1, npt2=npt2, rule2=rule2, atol=atol_, rtol=rtol_, maxevals=maxevals)
end

function automatic_equispace_integration_(f, a, b, npt1, npt2, atol, rtol, maxevals)
    int1 = equispace_evalrule(f, a, b, npt1)
    int2 = equispace_evalrule(f, a, b, npt2)
    numevals = npt1^length(a) + npt2^length(a)
    int2norm = norm(int2)
    err = norm(int1 - int2)
    while true
        (err ≤ max(rtol*int2norm, atol) || numevals ≥ maxevals || !isfinite(err)) && break
        # update coarse result with finer result
        int1 = int2
        npt1 = npt2
        # evaluate integral on finer grid
        npt2 = equispace_npt_update(f, npt1)
        int2 = equispace_evalrule(f, a, b, npt2)
        numevals += npt2^length(a)
        # self-convergence error estimate
        int2norm = norm(int2)
        err = norm(int1 - int2)
    end
    return int2, err, (npt1=npt1, npt2=npt2, numevals=numevals)
end

"""
    equispace_index(npt, i::Int...)

Return the linear index from the Cartesian index assuming a grid with `npt`
points per axis.
"""
equispace_index(npt, i::Int...) = equispace_index(npt, i) + 1
equispace_index(npt, i::NTuple{N,Int}) where N = i[1] - 1 + npt*equispace_index(npt, i[2:N])
equispace_index(npt, i::Tuple{}) = 0