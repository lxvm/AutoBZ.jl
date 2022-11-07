export tree_integration, iterated_integration,
    iterated_tol_update, iterated_pre_eval

"""
    tree_integration(f, a, b)
    tree_integration(f, ::CubicLimits)

Calls `HCubature` to perform multi-dimensional integration of `f` over a cube.
"""
tree_integration(f, a, b; kwargs...) = hcubature(f, a, b; kwargs...)
tree_integration(f, c::CubicLimits; kwargs...) = tree_integration(f, c.l, c.u; kwargs...)

"""
    iterated_tol_update(f, l, atol, rtol)

Choose a new set of error tolerances for the next inner integral. By default
returns `(atol, rtol)` unchanged.
"""
iterated_tol_update(f, l, atol, rtol) = (atol, rtol)

"""
    iterated_pre_eval(f, x)

Perform a precomputation on `f` using the value of a variable of integration,
`x`. The default is to store `x` and delay the computation of `f(x)` until all
of the values of the variables of integration are determined at a integration
point. Certain types of functions, such as Fourier series, take can use `x` to
precompute a new integrand for the remaining variables of integration that is
more computationally efficient.
"""
iterated_pre_eval(f, x) = thunk(f, x)

"""
    ThunkIntegrand(f, x)

Store `f` and `x` to evaluate `f(x)` at a later time. Employed by
`iterated_integration` for generic integrands that haven't been specialized to
use `iterated_pre_eval`.
"""
struct ThunkIntegrand{T,d,X}
    f::T
    x::SVector{d,X}
end

(f::ThunkIntegrand)(x) = f.f(vcat(x, f.x))

"""
    thunk(f, x)

Delay the computation of f(x). Needed to normally evaluate an integrand in
nested integrals, a setting in which the values of the variables of integration
are passed one at a time. Importantly, `thunk` assumes that the variables of
integration are passed from the outermost to the innermost. For example, to
evaluate `f([1, 2])`, call `thunk(f, 2)(1)`.

This behavior is consistent with `CubicLimits`, but may come as a surprise if
implementing new `IntegrationLimits`.
"""
thunk(f, x) = ThunkIntegrand(f, SVector(x))
thunk(f::ThunkIntegrand, x) = ThunkIntegrand(f.f, vcat(x, f.x))

"""
    iterated_integration(f, ::IntegrationLimits; order=4, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), segbufs=nothing)
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
`order` of the integration rule (defaults to 4).

The algorithm is an adaptive Gauss-Kronrod integration technique: the integral
in each interval is estimated using a Kronrod rule (`2*order+1` points) and the
error is estimated using an embedded Gauss rule (`order` points). The interval
with the largest error is then subdivided into two intervals and the process is
repeated until the desired error tolerance is achieved. This 1D procedure is
applied recursively to each variable of integration in an order determined by
`l` to obtain the multi-dimensional integral.

Unlike `quadgk`, this routine does not allow infinite limits of integration nor
unions of intervals to avoid singular points of the integrand.

In normal usage, `iterated_integration` will allocate segment buffers. You can
instead pass a preallocated buffer allocated using `alloc_segbufs` as the segbuf
argument. This buffer can be used across multiple calls to avoid repeated
allocation.
"""
iterated_integration(f, a, b; kwargs...) = iterated_integration(f, CubicLimits(a, b); kwargs...)
function iterated_integration(f, l::IntegrationLimits; order=4, atol=nothing, rtol=nothing, norm=norm, maxevals=10^7, segbufs=nothing)
    Tfx, Tnfx = infer_f(f, SVector{ndims(l),eltype(l)})
    segbufs_ = segbufs === nothing ? alloc_segbufs(eltype(l), Tfx, Tnfx, ndims(l)) : segbufs
    atol_ = something(atol, zero(Tnfx))/nsyms(l)
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(Tnfx))) : zero(Tnfx))
    int, err = iterated_integration_(f, l, order, atol_, rtol_, maxevals, norm, segbufs_)
    symmetrize(l, int, err)
end
function iterated_integration_(f, l::IntegrationLimits{1}, order, atol, rtol, maxevals, norm, segbufs)
    QuadGK.do_quadgk(f, (lower(l), upper(l)), order, atol, rtol, maxevals, norm, segbufs[1])
end
function iterated_integration_(f, l::IntegrationLimits, order, atol, rtol, maxevals, norm, segbufs)
    QuadGK.do_quadgk((lower(l), upper(l)), order, atol, rtol, maxevals, norm, segbufs[1]) do x
        first(iterated_integration_(iterated_pre_eval(f, x), l(x), order, iterated_tol_update(f, l, atol, rtol)..., maxevals, norm, Base.tail(segbufs)))
    end
end

"""
    infer_f(f, Tx)

Evaluates `f(zero(Tx))` and `norm(f(zero(Tx)))` and returns their types. If the
type of the range of `f` is known apriori, this method is meant to be specialized.
"""
function infer_f(f, Tx)
    fx = evaluate_integrand(f, zero(Tx))
    nfx = norm(fx)
    typeof(fx), typeof(nfx)
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