"""
    EquiBaryInterp

A module to perform local barycentric Lagrange interpolation of data on
equispaced grids as in the paper by Berrut and Trefethen (2004) SIAM review
https://doi.org/10.1137/S0036144502417715
"""
module EquiBaryInterp

export barycentric_weights, BaryPoly, LocalEquiBaryInterp

"""
    barycentric_weights(x::AbstractVector{T}) where {T<:Real}

Computes barycentric weights for the nodes `x`.
"""
function barycentric_weights(x::AbstractVector{T}) where {T<:Real}
    T[inv(prod(x[i] - x[j] for j in 1:N if j != i)) for i in 1:length(x)]
end
barycentric_weights(x::AbstractRange) = equi_bary_weights(length(x)-1)
"""
    equi_bary_weights(n::Integer)

Computes barycentric weights for `n+1` equispace nodes.
"""
function equi_bary_weights(n::T) where {T<:Integer}
    T[(-1)^j * binomial(n, j) for j in 0:n]
end


"""
    bary_kernel(x, xs, ys, ws)

Computes the value of the barycentric Lagrange polynomial interpolant with nodes
`xs`, values `ys`, and weights `ws` at point `x`.
"""
function bary_kernel(x_, xs::AbstractVector{T}, ys, ws) where T
    # unroll the first loop to get the right types
    x = convert(T, x_)
    Δx = x - xs[1]
    iszero(Δx) && return ys[1]
    q = l = ws[1]/Δx
    p = l * ys[1]
    @inbounds for i in 2:length(ws)
        Δx = x - xs[i]
        iszero(Δx) && return ys[i]
        q += l = ws[i]/Δx
        p += l * ys[i]
    end
    p/q
end


"""
    BaryPoly(x, y)
    BaryPoly(x, y, w)

Constructs a barycentric Lagrange polynomial from the data `y` sampled on `x`.
"""
struct BaryPoly{Tx,Ty,Tw} <: Function
    x::Vector{Tx}
    y::Vector{Ty}
    w::Vector{Tw}
end
BaryPoly(x, y) = BaryPoly(x, y, barycentric_weights(x))
function (b::BaryPoly)(x)
    first(b.x) <= x <= last(b.x) || throw(ArgumentError("x is out of range of interpolant"))
    bary_kernel(x, b.x, b.y, b.w)
end


"""
    LocalEquiBaryInterp(x::AbstractVector, y::AbstractVector, [degree=8])
    LocalEquiBaryInterp(x::Vector, y::Vector, w::Vector, h)

Construct a local barycentric Lagrange interpolant that forms a degree `degree`
local polynomial approximation of the data `y` on the equispace grid `x`, which
must be identical to a range with step size `h`. `w` are the equispace
interpolation weights.
"""
struct LocalEquiBaryInterp{Tx,Ty,Tw} <: Function
    x::Vector{Tx}
    y::Vector{Ty}
    w::Vector{Tw}
    h⁻¹::Tx
end
function LocalEquiBaryInterp(x::AbstractVector{Tx}, y::AbstractVector{Ty}, degree::Integer=8) where {Tx,Ty}
    if (n = length(x)) < degree+1
        throw(ArgumentError("Insufficient nodes to construct interpolant of requested degree"))
    else
        @assert n == length(y) "nodes and data are not the same length"
    end
    LocalEquiBaryInterp(convert(Vector{Tx}, x), convert(Vector{Ty}, y), equi_bary_weights(degree), inv(step(to_range(x))))
end
function (b::LocalEquiBaryInterp{T})(x_::Number) where T
    x = convert(T, x_)
    n = length(b.x)
    b.x[1] <= x <= b.x[n] || throw(ArgumentError("x is out of range of interpolant"))

    # find nodes nearest to the evaluation point to construct a local interpolant
    r = (x - b.x[1])*b.h⁻¹
    i = round(Int, r) # find index of nearest grid point (zero-indexed)
    p = length(b.w)
    q, rr = divrem(p, 2)
    i -= r > i ? q - 1 : q - 2 + rr # shift index to lowest of nearest p grid points (one-indexed)
    i += min(0, n+1-i-p) + max(1-i, 0) # offset to stay within grid
    idx = i:(i+p-1)
    @inbounds xs = view(b.x, idx)
    @inbounds ys = view(b.y, idx)

    bary_kernel(x, xs, ys, b.w)
end


"""
    to_range(x::AbstractVector)

Assert that `x` is numerically identical to an equispace range, and return an
equivalent range object.
"""
to_range(x::AbstractRange) = x
function to_range(x::AbstractVector)
    y = range(first(x), last(x), length=length(x))
    x == y || throw(InexactError(:to_range, typeof(y), x))
    y
end

end # module