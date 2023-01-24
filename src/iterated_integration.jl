export iterated_integration
export iterated_tol_update, iterated_integrand, iterated_pre_eval, iterated_segs

"""
    iterated_tol_update(f, l, atol, rtol)

Choose a new set of error tolerances for the next inner integral. By default
returns `(atol, rtol)` unchanged.
"""
iterated_tol_update(f, l, atol, rtol, dim) = (atol, rtol)

"""
    iterated_integrand(f, y, dim)

By default, returns `y` which is the result of an interior integral.
"""
@inline iterated_integrand(f, y, ::Type{Val{d}}) where d = iterated_integrand(f, y, d)
@inline iterated_integrand(f, y, dim) = y

"""
    iterated_pre_eval(f, x, dim)

Perform a precomputation on `f` using the value of a variable of integration,
`x`. The default is to store `x` and delay the computation of `f(x)` until all
of the values of the variables of integration are determined at a integration
point. Certain types of functions, such as Fourier series, take can use `x` to
precompute a new integrand for the remaining variables of integration that is
more computationally efficient. This function must return the integrand for the
subsequent integral.
"""
@inline iterated_pre_eval(f, x, ::Type{Val{d}}) where d = iterated_pre_eval(f, x, d)
iterated_pre_eval(f, x, dim) = iterated_pre_eval(f, x)

"""
    iterated_segs(f, l, d, initdivs)

Returns a `Tuple` of integration nodes that are passed to `QuadGK` to initialize
the segments for adaptive integration. By default, returns `initdivs` equally
spaced panels on `(lower(l), upper(l))`. If `f` is localized, specializing this
function can also help avoid errors when `QuadGK` fails to adapt.
"""
function iterated_segs(f, l, d, ::Val{initdivs}) where initdivs
    lb, ub = limits(l, d)
    r = range(lb, ub, length=initdivs+1)
    ntuple(i -> r[i], Val{initdivs+1}())
end

"""
    iterated_integration(f, ::IntegrationLimits; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdivs=ntuple(i -> Val(1), Val{d}()), segbufs=nothing)
    iterated_integration(f, a, b; kwargs...)

Calls `QuadGK` to perform iterated 1D integration of `f` over a domain
parametrized by `IntegrationLimits`. In the case two points `a` and `b` are
passed, the integration region becomes the hypercube with those extremal
vertices.

Returns a tuple `(I, E)` of the estimated integral and estimated error.

Keyword options include a relative error tolerance `rtol` (if `atol==0`,
defaults to `sqrt(eps)` in the precision of the norm of the return type), an
absolute error tolerance `atol` (defaults to 0), a maximum number of function
evaluations `maxevals` for each nested integral (defaults to `10^7`), and the
`order` of the integration rule (defaults to 7).

The algorithm is an adaptive Gauss-Kronrod integration technique: the integral
in each interval is estimated using a Kronrod rule (`2*order+1` points) and the
error is estimated using an embedded Gauss rule (`order` points). The interval
with the largest error is then subdivided into two intervals and the process is
repeated until the desired error tolerance is achieved. This 1D procedure is
applied recursively to each variable of integration in an order determined by
`l` to obtain the multi-dimensional integral.

Unlike `quadgk`, this routine does not allow infinite limits of integration nor
unions of intervals to avoid singular points of the integrand. However, the
`initdivs` keyword allows passing a tuple of integers which specifies the
initial number of panels in each `quadgk` call at each level of integration.

In normal usage, `iterated_integration` will allocate segment buffers. You can
instead pass a preallocated buffer allocated using `alloc_segbufs` as the segbuf
argument. This buffer can be used across multiple calls to avoid repeated
allocation.
"""
iterated_integration(f, a, b; kwargs...) = iterated_integration(f, CubicLimits(a, b); kwargs...)
function iterated_integration(f, l::IntegrationLimits{d}; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=10^7, initdivs=ntuple(i -> Val(1), Val{d}()), segbufs=nothing) where d
    Tfx = Base.promote_op(f, SVector{ndims(l),eltype(l)})
    Tnfx = Base.promote_op(norm, Tfx)
    segbufs_ = segbufs === nothing ? alloc_segbufs(eltype(l), Tfx, Tnfx, ndims(l)) : segbufs
    atol_ = something(atol, zero(Tnfx))/nsyms(l)
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(Tnfx))) : zero(Tnfx))
    int, err = iterated_integration_(Val{d}, f, l, order, atol_, rtol_, maxevals, norm, initdivs, segbufs_)
    symmetrize(l, int, err)
end

function iterated_integration_(::Type{Val{1}}, f, l, order, atol, rtol, maxevals, norm, initdivs, segbufs)
    QuadGK.do_quadgk(f, iterated_segs(f, l, 1, initdivs[1]), order, atol, rtol, maxevals, norm, segbufs[1])
end
function iterated_integration_(::Type{Val{d}}, f, l, order, atol, rtol, maxevals, norm, initdivs, segbufs) where d
    # avoid runtime dispatch when capturing variables
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    f_ = let f=f, l=l, order=order, atol=atol, rtol=rtol, maxevals=maxevals, norm=norm, initdivs=initdivs, segbufs=segbufs
        x -> iterated_integrand(f, first(iterated_integration_(Val{d-1}, iterated_pre_eval(f, x, Val{d}), l(x, d), order, iterated_tol_update(f, l, atol, rtol, d)..., maxevals, norm, initdivs, segbufs)), Val{d})
    end
    QuadGK.do_quadgk(f_, iterated_segs(f, l, d, initdivs[d]), order, atol, rtol, maxevals, norm, segbufs[d])
end

"""
    alloc_segbufs(eltype_l, typeof_fx, typeof_nfx, ndims_l)

This helper function will allocate enough segment buffers as are needed for an
`iterated_integration` call of integrand `f` and integration limits `l`.
`eltype_l` should be `eltype(l)`, `typeof_fx` should be the return type of the
integrand `f`, `typeof_nfx` should be the type of the norm of a value of `f`,
and `ndims_l` should be `ndims(l)`.
"""
alloc_segbufs(eltype_l, typeof_fx, typeof_nfx, ndims_l) = ntuple(_ -> QuadGK.alloc_segbuf(eltype_l, typeof_fx, typeof_nfx), Val{ndims_l}())