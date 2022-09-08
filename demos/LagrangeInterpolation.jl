module LagrangeInterpolation

using StaticArrays: SVector, sacollect

export barycentric_weights, BaryPoly, LocalEquiBaryInterpolant

function barycentric_weights(x::AbstractVector{T}) where {T<:Real}
    N = length(x)
    sacollect(SVector{N,T}, inv(prod(x[i] - x[j] for j in 1:N if j != i)) for i in 1:N)
end
barycentric_weights(x::AbstractRange) = equi_bary_weights(length(x)-1)
"Return barycentric weights for `n+1` equispace nodes"
equi_bary_weights(n) = sacollect(SVector{n+1,Int64}, (-1)^j * binomial(n, j) for j in 0:n)

"""
    BaryPoly(x, y)
    BaryPoly(x, y, w)

Construct and store a Barycentric Lagrange polynomial from the data
"""
struct BaryPoly{N,Tx<:Real,Ty,Tw<:Real}
    x::SVector{N,Tx}
    y::SVector{N,Ty}
    w::SVector{N,Tw}
end

BaryPoly(x, y) = BaryPoly(x, y, barycentric_weights(x))
function (b::BaryPoly{N,Tx,Ty})(x::Real) where {N,Tx,Ty}
    first(b.x)<=x<=last(b.x) || ArgumentError("x is out of range of interpolant")
    bary_kernel(x, b.x, b.y, b.w, N, Tx, Ty)
end
function bary_kernel(x, xs, ys, ws, N, Tx, Ty)
    p = zero(Ty)
    q = zero(Tx)
    for i in 1:N
        Δx = x - xs[i]
        iszero(Δx) && return ys[i]
        l = ws[i]/Δx
        p += l*ys[i]
        q += l
    end
    p/q
end

"""
    LocalEquiBaryInterpolant(x, y, n::Integer)
    LocalEquiBaryInterpolant(x, y, w)

Construct a local Barycentric Lagrange interpolant that uses the `n` nearest
points to the evaluation point to form weights `w` of a Barycentric Lagrange
polynomial interpolant of the data for f.
"""
struct LocalEquiBaryInterpolant{N,Tx<:AbstractRange,Ty<:AbstractVector,Tw}
    x::Tx
    y::Ty
    w::SVector{N,Tw}
end
LocalEquiBaryInterpolant(x, y, n::Integer) = LocalEquiBaryInterpolant(to_range(x), y, n)
function LocalEquiBaryInterpolant(x::AbstractRange, y, n::Integer)
    length(x) >= n+1 || ArgumentError("Insufficient nodes to construct requested order interpolant")
    LocalEquiBaryInterpolant(x, y, equi_bary_weights(n))
end
function (b::LocalEquiBaryInterpolant{N,Tx,Ty})(x::Number) where {N,Tx,Ty}
    first(b.x)<=x<=last(b.x) || ArgumentError("x is out of range of interpolant")
    local_bary_kernel(x, b.x, b.y, b.w, N, Tx, Ty)
end
function local_bary_kernel(x, xs::AbstractRange, ys::AbstractVector, ws, N, Tx, Ty)
    r = (x - first(xs))/step(xs) + 1
    i = round(Int, r)
    if r > i
        il = i-ceil(Int, N/2)+1
        iu = i+floor(Int, N/2)
    else # r <= i
        il = i-floor(Int, N/2)
        iu = i+ceil(Int, N/2)-1
    end
    edge_offset = min(0, length(xs)-iu) + max(1-il, 0)
    il += edge_offset
    iu += edge_offset
    idx = SVector{N,Int}(il:iu)
    xs_ = xs[idx]
    ys_ = ys[idx]
    bary_kernel(x, xs_, ys_, ws, N, eltype(Tx), eltype(Ty))
end

function to_range(x::AbstractVector)
    y = range(first(x), last(x), length=length(x))
    x == y || error("input points could not be converted to a range")
    y
end

end # module