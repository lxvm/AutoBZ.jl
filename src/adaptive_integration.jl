export tree_integration, iterated_integration,
    iterated_tol_update, iterated_pre_eval

"""
    tree_integration(f, a, b)
    tree_integration(f, ::CubicLimits)

Calls `HCubature` to perform multi-dimensional integration of `f` over a cube.
"""
tree_integration(f, a, b; callback=nothing, kwargs...) = hcubature(f, a, b; kwargs...)
tree_integration(f, c::CubicLimits; kwargs...) = tree_integration(f, c.l, c.u; kwargs...)

"""
    iterated_tol_update(f, l, atol, rtol)

Choose a new set of error tolerances for the next inner integral. By default
returns `(atol, rtol)` unchanged.
"""
iterated_tol_update(f, l, atol, rtol) = (atol, rtol)

"""
    iterated_pre_eval(f, x)

Perform a precomputation
"""
iterated_pre_eval(f, x) = thunk(f, x)

"""
    iterated_integration(f, ::IntegrationLimits)

Calls `HCubature` to perform iterated 1D integration of `f` over a domain
parametrized by `IntegrationLimits`.
"""
iterated_integration(f, a, b; kwargs...) = iterated_integration(f, CubicLimits(a, b); kwargs...)
#=
function iterated_integration(f, l::IntegrationLimits; atol=0.0, rtol=sqrt(eps()), norm=norm, kwargs...)
    T = typeof(lower(l))
    fx = f(zero(SVector{ndims(l),T}))
    segbufs = ntuple(_ -> QuadGK.alloc_segbuf(T, typeof(fx), typeof(norm(fx))), Val{ndims(l)}())
    int, err = iterated_integration_(f, l, atol/nsyms(l), rtol, segbufs; norm=norm, kwargs...)
    symmetrize(l, int, err)
end

function iterated_integration_(f, l::IntegrationLimits{1}, atol, rtol, segbufs; kwargs...) 
    quadgk(f, lower(l), upper(l); atol=atol, rtol=rtol, segbuf=segbufs[1], kwargs...)
end
function iterated_integration_(f, l::IntegrationLimits, atol, rtol, segbufs; kwargs...)
    quadgk(lower(l), upper(l); atol=atol, rtol=rtol, segbuf=segbufs[1], kwargs...) do x
        first(iterated_integration_(iterated_pre_eval(f, x), l(x), iterated_tol_update(f, l, atol, rtol)..., Base.tail(segbufs); kwargs...))
    end
end
=#
# TODO: draft v2 using QuadGK.do_quadgk since v1 is horribly slow
function iterated_integration(f, l::IntegrationLimits; order=7, atol=0.0, rtol=sqrt(eps()), norm=norm, maxevals=typemax(Int))
    T = typeof(lower(l))
    fx = f(zero(SVector{ndims(l),T}))
    segbufs = ntuple(_ -> QuadGK.alloc_segbuf(T, typeof(fx), typeof(norm(fx))), Val{ndims(l)}())
    int, err = iterated_integration_(f, l, order, atol/nsyms(l), rtol, maxevals, norm, segbufs)
    symmetrize(l, int, err)
end
function iterated_integration_(f, l::IntegrationLimits{1}, order, atol, rtol, maxevals, nrm, segbufs)
    QuadGK.do_quadgk(f, (lower(l), upper(l)), order, atol, rtol, maxevals, nrm, segbufs[1])
end
function iterated_integration_(f, l::IntegrationLimits, order, atol, rtol, maxevals, nrm, segbufs)
    QuadGK.do_quadgk((lower(l), upper(l)), order, atol, rtol, maxevals, nrm, segbufs[1]) do x
        first(iterated_integration_(iterated_pre_eval(f, x), l(x), order, iterated_tol_update(f, l, atol, rtol)..., maxevals, nrm, Base.tail(segbufs)))
    end
end
# TODO: draft v3 using QuadGK.adapt?
#=
function iterated_integration(f, l::IntegrationLimits; order=7, atol=0.0, rtol=sqrt(eps()), norm=norm, maxevals=typemax(Int))
    x,w,gw = cachedrule(eltype(l),order)
    int, err = iterated_integration_(f, l, atol/nsyms(l), rtol, segbufs)
    symmetrize(l, int, err)
end
function iterated_integration_(f, l::IntegrationLimits{1}, atol, rtol, segbufs) 
    QuadGK.adapt(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm)
end
function iterated_integration_(f, l::IntegrationLimits, atol, rtol, segbufs)
    QuadGK.adapt(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm) do x
        first(iterated_integration_(iterated_pre_eval(f, x), l(x), iterated_tol_update(f, l, atol, rtol)..., Base.tail(segbufs); kwargs...))
    end
end
=#